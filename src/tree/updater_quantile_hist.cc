/*!
 * Copyright 2017-2020 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <numeric>
#include <string>
#include <utility>

#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

#include "constraints.h"
#include "param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"
#include "../common/threading_utils.h"
#include <fstream>
#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

DMLC_REGISTER_PARAMETER(CPUHistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", tparam_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template<typename GradientSumT>
void QuantileHistMaker::SetBuilder(std::unique_ptr<Builder<GradientSumT>>* builder,
                                   DMatrix *dmat) {
  builder->reset(new Builder<GradientSumT>(
                param_,
                std::move(pruner_),
                int_constraint_, dmat));
  if (rabit::IsDistributed()) {
    (*builder)->SetHistSynchronizer(new DistributedHistSynchronizer<GradientSumT>());
    (*builder)->SetHistRowsAdder(new DistributedHistRowsAdder<GradientSumT>());
  } else {
    (*builder)->SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
    (*builder)->SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  }
}

template<typename GradientSumT>
void QuantileHistMaker::CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                                          HostDeviceVector<GradientPair> *gpair,
                                          DMatrix *dmat,
                                          const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, tree, numa1_bins.data(), numa2_bins.data(), &histograms_);
  }
}
void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  std::vector<GradientPair>& gpair_h = gpair->HostVector();
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    gmat_.Init(dmat, static_cast<uint32_t>(param_.max_bin));
    column_matrix_.Init(gmat_, param_.sparse_threshold);
    if (param_.enable_feature_grouping > 0) {
      gmatb_.Init(gmat_, column_matrix_, param_);
    }
    updater_monitor_.Stop("GmatInitialization");
    // A proper solution is puting cut matrix in DMatrix, see:
    // https://github.com/dmlc/xgboost/issues/5143
    is_gmat_initialized_ = true;
    const size_t n_threads = omp_get_max_threads();
    //    std::cout << "\nn_threads: " << n_threads << "\n";
    const size_t n_elements = gmat_.index.Size();
    const uint8_t* data = gmat_.index.data<uint8_t>();
    const size_t n_bins = gmat_.cut.Ptrs().back();
    histograms_.resize(n_threads);
    #pragma omp parallel num_threads(n_threads)
    {
      const size_t tid = omp_get_thread_num();
      if (tid == 0) {
        //std::cout << "\n\nNUMA1 was initiated!\n\n";
        this->numa1_bins.resize(n_elements,0);
        for (size_t i = 0; i < n_elements; ++i) {
          this->numa1_bins[i] = data[i];
        }
      }
      if (tid == (n_threads - 1)) {
        //std::cout << "\n\nNUMA2 was initiated!\n\n";
        this->numa2_bins.resize(n_elements,0);
        for (size_t i = 0; i < n_elements; ++i) {
          this->numa2_bins[i] = data[i];
        }
      }
      histograms_[tid].resize(n_bins*(1 << (param_.max_depth + 1)), 0);
    }
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  int_constraint_.Configure(param_, dmat->Info().num_col_);
  // build tree
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      SetBuilder(&float_builder_, dmat);
    }
    CallBuilderUpdate(float_builder_, gpair, dmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(&double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds) {
  if (param_.subsample < 1.0f) {
    return false;
  } else {
    if (hist_maker_param_.single_precision_histogram && float_builder_) {
        return float_builder_->UpdatePredictionCache(data, out_preds);
    } else if (double_builder_) {
        return double_builder_->UpdatePredictionCache(data, out_preds);
    } else {
       return false;
    }
  }
}

template <typename GradientSumT>
void BatchHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT *builder,
                                                         int,
                                                         int,
                                                         RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  // const size_t nbins = builder->hist_builder_.GetNumBins();
  // common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t) {
  //   return nbins;
  // }, 1024);

  // common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
  //   const auto& entry = builder->nodes_for_explicit_hist_build_[node];
  //   auto this_hist = builder->hist_[entry.nid];
  //   // Merging histograms from each thread into once
  //   builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());

  //   if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
  //     const size_t parent_id = (*p_tree)[entry.nid].Parent();
  //     auto parent_hist = builder->hist_[parent_id];
  //     auto sibling_hist = builder->hist_[entry.sibling_nid];
  //     SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
  //   }
  // });
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT* builder,
                                                 int starting_index,
                                                 int sync_count,
                                                 RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();
  common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t) {
    return nbins;
  }, 1024);
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Merging histograms from each thread into once
    builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());
    // Store posible parent node
    auto this_local = builder->hist_local_worker_[entry.nid];
    CopyHist(this_local, this_hist, r.begin(), r.end());

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_local_worker_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
      // Store posible parent node
      auto sibling_local = builder->hist_local_worker_[entry.sibling_nid];
      CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
    }
  });
  builder->builder_monitor_.Start("SyncHistogramsAllreduce");

  builder->histred_.Allreduce(builder->hist_[starting_index].data(),
                                    builder->hist_builder_.GetNumBins() * sync_count);

  builder->builder_monitor_.Stop("SyncHistogramsAllreduce");

  ParallelSubtractionHist(builder, space, builder->nodes_for_explicit_hist_build_, p_tree);

  common::BlockedSpace2d space2(builder->nodes_for_subtraction_trick_.size(), [&](size_t) {
    return nbins;
  }, 1024);
  ParallelSubtractionHist(builder, space2, builder->nodes_for_subtraction_trick_, p_tree);
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizer<GradientSumT>::ParallelSubtractionHist(
                                  BuilderT* builder,
                                  const common::BlockedSpace2d& space,
                                  const std::vector<ExpandEntryT>& nodes,
                                  const RegTree * p_tree) {
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = nodes[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = builder->hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
        auto parent_hist = builder->hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = builder->hist_[entry.sibling_nid];
        SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(), r.end());
      }
    }
  });
}

template <typename GradientSumT>
void BatchHistRowsAdder<GradientSumT>::AddHistRows(BuilderT *builder,
                                                   int *starting_index,
                                                   int *sync_count,
                                                   RegTree *) {
  builder->builder_monitor_.Start("AddHistRows");
  for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
    int nid = entry.nid;
    builder->hist_.AddHistRow(nid);
    (*starting_index) = std::min(nid, (*starting_index));
  }
  (*sync_count) = builder->nodes_for_explicit_hist_build_.size();

  for (auto const& node : builder->nodes_for_subtraction_trick_) {
    builder->hist_.AddHistRow(node.nid);
  }

  builder->hist_.AllocateAllData();
  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void DistributedHistRowsAdder<GradientSumT>::AddHistRows(BuilderT *builder,
                                                         int *starting_index,
                                                         int *sync_count,
                                                         RegTree *p_tree) {
  builder->builder_monitor_.Start("AddHistRows");
  const size_t explicit_size = builder->nodes_for_explicit_hist_build_.size();
  const size_t subtaction_size = builder->nodes_for_subtraction_trick_.size();
  std::vector<int> merged_node_ids(explicit_size + subtaction_size);
  for (size_t i = 0; i < explicit_size; ++i) {
    merged_node_ids[i] = builder->nodes_for_explicit_hist_build_[i].nid;
  }
  for (size_t i = 0; i < subtaction_size; ++i) {
    merged_node_ids[explicit_size + i] =
    builder->nodes_for_subtraction_trick_[i].nid;
  }
  std::sort(merged_node_ids.begin(), merged_node_ids.end());
  int n_left = 0;
  for (auto const& nid : merged_node_ids) {
    if ((*p_tree)[nid].IsLeftChild()) {
      builder->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
      n_left++;
      builder->hist_local_worker_.AddHistRow(nid);
    }
  }
  for (auto const& nid : merged_node_ids) {
    if (!((*p_tree)[nid].IsLeftChild())) {
      builder->hist_.AddHistRow(nid);
      builder->hist_local_worker_.AddHistRow(nid);
    }
  }
  builder->hist_.AllocateAllData();
  builder->hist_local_worker_.AllocateAllData();
  (*sync_count) = std::max(1, n_left);
  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistSynchronizer(
    HistSynchronizer<GradientSumT> *sync) {
  hist_synchronizer_.reset(sync);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistRowsAdder(
    HistRowsAdder<GradientSumT> *adder) {
  hist_rows_adder_.reset(adder);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildHistogramsLossGuide(
    ExpandEntry entry, const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb, RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  if (entry.sibling_nid > -1) {
    nodes_for_subtraction_trick_.emplace_back(entry.sibling_nid, entry.nid,
        p_tree->GetDepth(entry.sibling_nid), 0.0f, 0);
  }

  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
  BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h);
  hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
}
static size_t N_CALL = 0;
#include <sys/time.h>
#include <time.h>
uint64_t get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000 + t.tv_nsec;
}
#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)


struct Prefetch1 {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize =
      kPrefetchOffset + kCacheLineSize /
      sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch1::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch1::kNoPrefetchSize;

#define UNR(IDX, J)                                                                                    \
    const uint32_t idx_bin##IDX = two * (static_cast<uint32_t>(gr_index_local[13*J + IDX]) + offsets[13*J + IDX]); \
    hist_data[idx_bin##IDX]   += pgh[idx_gh]; \
    hist_data[idx_bin##IDX+1] += pgh[idx_gh+1];

#define UNR_TAIL(IDX)                                                                                    \
    const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[IDX]) + offsets[IDX]); \
    hist_data[idx_bin]   += pgh[idx_gh]; \
    hist_data[idx_bin+1] += pgh[idx_gh+1];

#define VECTOR_UNR(IDX, J)                                                                                 \
    const size_t offset##IDX = offsets64[IDX + 13*J] + ((size_t)(gr_index_local[IDX + 13*J])) * 16; \
    asm("vmovapd (%0), %%xmm1;" : : "r" ( offset##IDX ) : /*"%xmm1"*/);                 \
    asm("vaddpd %xmm2, %xmm1, %xmm3;");                                                             \
    asm("vmovapd %%xmm3, (%0);" : : "r" ( offset##IDX ) : /*"%xmm3"*/);                 \

template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<FPType> hist, const uint8_t* numa, uint16_t* nodes_ids) {
  //const size_t size = row_indices.Size();
  //const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

std::vector<uint64_t> offsets64_v(n_features);
uint64_t* offsets64 = &(offsets64_v[0]);

for(size_t i = 0; i < n_features; ++i) {
  offsets64[i] = (uint64_t)hist_data + 16*(uint64_t)(offsets[i]);
}

  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    nodes_ids[i] = 0;
    const size_t icol_start = i * n_features;
    const size_t idx_gh = two *i;
    const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

    const uint8_t* gr_index_local = gradient_index + icol_start;
   // if (nb >= 1) {
      VECTOR_UNR(0, 0);
      VECTOR_UNR(1, 0);
      VECTOR_UNR(2, 0);
      VECTOR_UNR(3, 0);
      VECTOR_UNR(4, 0);
      VECTOR_UNR(5, 0);
      VECTOR_UNR(6, 0);
      VECTOR_UNR(7, 0);
      VECTOR_UNR(8, 0);
      VECTOR_UNR(9, 0);
      VECTOR_UNR(10, 0);
      VECTOR_UNR(11, 0);
      VECTOR_UNR(12, 0);
      // VECTOR_UNR(13, 0);
      // VECTOR_UNR(14, 0);
      // VECTOR_UNR(15, 0);
      // VECTOR_UNR(16, 0);
      // VECTOR_UNR(17, 0);
      // VECTOR_UNR(18, 0);
      // VECTOR_UNR(19, 0);
      // VECTOR_UNR(20, 0);
      // VECTOR_UNR(21, 0);
      // VECTOR_UNR(22, 0);
      // VECTOR_UNR(23, 0);
      // VECTOR_UNR(24, 0);
      // VECTOR_UNR(25, 0);
      // VECTOR_UNR(26, 0);
      // VECTOR_UNR(27, 0);
  }
}


void JustPartition(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const uint8_t* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, uint64_t* mask) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
 uint32_t& t = *hist;
//  if((row_indices_end - row_indices_begin )> Prefetch1::kPrefetchOffset) {
//   for (size_t i = row_indices_begin; i < row_indices_end - Prefetch1::kPrefetchOffset; ++i) {
//     const size_t icol_start = i * n_features;
//     const uint8_t* gr_index_local = gradient_index + icol_start;
//     const uint32_t nid = nodes_ids[i];
//     PREFETCH_READ_T0(gradient_index + (i+Prefetch1::kPrefetchOffset)*n_features + (*split_ind)[nodes_ids[i +Prefetch1::kPrefetchOffset] + 1]);
//     const int32_t sc = (*split_conditions)[nid + 1];
//     const bst_uint si = (*split_ind)[nid + 1];
//     nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
//     //count += i%2;
//     if (((uint64_t)(1) << nodes_ids[i]) & mask) {
//       rows[++count] = i;
//      //t = i;
//     }
//   }

//   for (size_t i = row_indices_end - Prefetch1::kPrefetchOffset; i < row_indices_end; ++i) {
//     const size_t icol_start = i * n_features;
//     const uint8_t* gr_index_local = gradient_index + icol_start;
//     const uint32_t nid = nodes_ids[i];
//     const int32_t sc = (*split_conditions)[nid + 1];
//     const bst_uint si = (*split_ind)[nid + 1];
//     nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
//     // //count += i%2;
//     if (((uint64_t)(1) << nodes_ids[i]) & mask) {
//       rows[++count] = i;
//      //t = i;
//     }
//   }
//  } else {
  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    const uint32_t nid = nodes_ids[i];
    // if(((uint16_t)(1) << 15 & nid)) {
    // continue;
    // }
    // if(((uint64_t)(1) << nid & uint64_t(3145728))) {
    //   nodes_ids[i] = (uint16_t)(1) << 15;
    // }
      const size_t icol_start = i * n_features;
      const uint8_t* gr_index_local = gradient_index + icol_start;
      const int32_t sc = (*split_conditions)[nid + 1];
      const bst_uint si = (*split_ind)[nid + 1];
      nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
      // //count += i%2;
      if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask + nodes_ids[i]/64)) {
        rows[++count] = i;
      //t = i;
      }
//    }
  }
// }

}


void JustPartitionWithLeafsMask(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const uint8_t* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, uint64_t* mask, uint64_t* leafs_mask, std::vector<int>* prev_level_nodes) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
 uint32_t& t = *hist;

  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    const uint32_t nid = nodes_ids[i];
    if(((uint16_t)(1) << 15 & nid)) {
      continue;
    }
    if((((uint64_t)(1) << (nid%64)) & *(leafs_mask + nid/64))) {
      nodes_ids[i] = (uint16_t)(1) << 15;
      nodes_ids[i] |= (uint16_t)((*prev_level_nodes)[nid]);
      continue;
    }
      const size_t icol_start = i * n_features;
      const uint8_t* gr_index_local = gradient_index + icol_start;
      const int32_t sc = (*split_conditions)[nid + 1];
      const bst_uint si = (*split_ind)[nid + 1];
      nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
      if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask+nodes_ids[i]/64)) {
        rows[++count] = i;
      }
  }
}


void JustPartitionLastLayer(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const uint8_t* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind,
                          std::vector<int>* curr_level_nodes, uint64_t* leafs_mask, std::vector<int>* prev_level_nodes) {
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    const uint32_t nid = nodes_ids[i];
    if(((uint16_t)(1) << 15 & nid)) {
    continue;
    }
    if((((uint64_t)(1) << (nid%64)) & *(leafs_mask + (nid/64)))) {
      nodes_ids[i] = (uint16_t)(1) << 15;
      nodes_ids[i] |= (uint16_t)((*prev_level_nodes)[nid]);
      continue;
    }
      const size_t icol_start = i * n_features;
      const uint8_t* gr_index_local = gradient_index + icol_start;
      const int32_t sc = (*split_conditions)[nid + 1];
      const bst_uint si = (*split_ind)[nid + 1];
      nodes_ids[i] = (uint16_t)(1) << 15;

      nodes_ids[i] |= (uint16_t)((*curr_level_nodes)[2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc)]);
  }

}


void JustPartitionColumnar(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const uint8_t* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, const ColumnMatrix *column_matrix) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
 uint32_t& t = *hist;
const ColumnMatrix& column_matrix_ref = *column_matrix;
const auto column_ptr = column_matrix_ref.GetColumn<uint8_t>(0);
//const common::DenseColumn<uint8_t>& column = static_cast<const common::DenseColumn<uint8_t>& >(*(column_ptr.get()));
  const uint8_t* idx = column_ptr->GetFeatureBinIdxPtr().data();

  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    const uint32_t nid = nodes_ids[i];
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
const auto column_ptr = column_matrix_ref.GetColumn<uint8_t>(si);
//const common::DenseColumn<uint8_t>& column = static_cast<const common::DenseColumn<uint8_t>& >(*(column_ptr.get()));
  const uint32_t idx = column_ptr->GetFeatureBinIdxPtr().data()[i];
    //t += 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
    // //count += i%2;
    if (i%2) {
     rows[++count] = idx;
    }
  }

}


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const uint32_t* rows,
                          const uint32_t row_size,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<FPType> hist, const uint8_t* numa, uint16_t* nodes_ids, const uint32_t n_nodes) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint8_t* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  const uint32_t n_bins = gmat.cut.Ptrs().back();
  FPType* hist_data0 = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
std::vector<uint64_t> offsets64_v(n_features*n_nodes);
uint64_t* offsets640 = &(offsets64_v[0]);

for(size_t nid = 0; nid < n_nodes; ++nid) {
  for(size_t i = 0; i < n_features; ++i) {
    offsets640[nid*n_features + i] = (uint64_t)(hist_data0 + nid*n_bins*2) + 16*(uint64_t)(offsets[i]);
  }
}

if (row_size > Prefetch1::kPrefetchOffset) {
  for (size_t ri = 0; ri < row_size - Prefetch1::kPrefetchOffset; ++ri) {
    const size_t i = rows[ri];
    const size_t icol_start = i * n_features;
    const uint8_t* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;
    const uint32_t nid = nodes_ids[i];

    const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

    PREFETCH_READ_T0(pgh + two * rows[ri + Prefetch1::kPrefetchOffset]);
    PREFETCH_READ_T0(nodes_ids + rows[ri + Prefetch1::kPrefetchOffset]);
    for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
        j += Prefetch1::GetPrefetchStep<BinIdxType>()) {
      PREFETCH_READ_T0(gradient_index + j);
    }

    const uint64_t* offsets64 = offsets640 + nid*n_features;
    const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

//    const int32_t sc = (*split_conditions)[nid + 1];
//    const bst_uint si = (*split_ind)[nid + 1];
//    nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
    FPType* hist_data = hist_data0 + nid*n_bins*2;
   // nodes_ids[i] = ;
   // if (nb >= 1) {
      VECTOR_UNR(0, 0);
      VECTOR_UNR(1, 0);
      VECTOR_UNR(2, 0);
      VECTOR_UNR(3, 0);
      VECTOR_UNR(4, 0);
      VECTOR_UNR(5, 0);
      VECTOR_UNR(6, 0);
      VECTOR_UNR(7, 0);
      VECTOR_UNR(8, 0);
      VECTOR_UNR(9, 0);
      VECTOR_UNR(10, 0);
      VECTOR_UNR(11, 0);
      VECTOR_UNR(12, 0);
      // VECTOR_UNR(13, 0);
      // VECTOR_UNR(14, 0);
      // VECTOR_UNR(15, 0);
      // VECTOR_UNR(16, 0);
      // VECTOR_UNR(17, 0);
      // VECTOR_UNR(18, 0);
      // VECTOR_UNR(19, 0);
      // VECTOR_UNR(20, 0);
      // VECTOR_UNR(21, 0);
      // VECTOR_UNR(22, 0);
      // VECTOR_UNR(23, 0);
      // VECTOR_UNR(24, 0);
      // VECTOR_UNR(25, 0);
      // VECTOR_UNR(26, 0);
      // VECTOR_UNR(27, 0);
  }

for (size_t ri = row_size - Prefetch1::kPrefetchOffset; ri < row_size; ++ri) {
    const size_t i = rows[ri];
    const size_t icol_start = i * n_features;
    const uint8_t* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;
    const uint32_t nid = nodes_ids[i];

    const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

    const uint64_t* offsets64 = offsets640 + nid*n_features;
    const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

    FPType* hist_data = hist_data0 + nid*n_bins*2;
    VECTOR_UNR(0, 0);
    VECTOR_UNR(1, 0);
    VECTOR_UNR(2, 0);
    VECTOR_UNR(3, 0);
    VECTOR_UNR(4, 0);
    VECTOR_UNR(5, 0);
    VECTOR_UNR(6, 0);
    VECTOR_UNR(7, 0);
    VECTOR_UNR(8, 0);
    VECTOR_UNR(9, 0);
    VECTOR_UNR(10, 0);
    VECTOR_UNR(11, 0);
    VECTOR_UNR(12, 0);
      // VECTOR_UNR(13, 0);
      // VECTOR_UNR(14, 0);
      // VECTOR_UNR(15, 0);
      // VECTOR_UNR(16, 0);
      // VECTOR_UNR(17, 0);
      // VECTOR_UNR(18, 0);
      // VECTOR_UNR(19, 0);
      // VECTOR_UNR(20, 0);
      // VECTOR_UNR(21, 0);
      // VECTOR_UNR(22, 0);
      // VECTOR_UNR(23, 0);
      // VECTOR_UNR(24, 0);
      // VECTOR_UNR(25, 0);
      // VECTOR_UNR(26, 0);
      // VECTOR_UNR(27, 0);
  }

} else {
    for (size_t ri = 0; ri < row_size; ++ri) {
    const size_t i = rows[ri];
    const size_t icol_start = i * n_features;
    const uint8_t* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;
    const uint32_t nid = nodes_ids[i];
    const uint64_t* offsets64 = offsets640 + nid*n_features;
    const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

//    const int32_t sc = (*split_conditions)[nid + 1];
//    const bst_uint si = (*split_ind)[nid + 1];
//    nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
    FPType* hist_data = hist_data0 + nid*n_bins*2;
   // nodes_ids[i] = ;
   // if (nb >= 1) {
  VECTOR_UNR(0, 0);
  VECTOR_UNR(1, 0);
  VECTOR_UNR(2, 0);
  VECTOR_UNR(3, 0);
  VECTOR_UNR(4, 0);
  VECTOR_UNR(5, 0);
  VECTOR_UNR(6, 0);
  VECTOR_UNR(7, 0);
  VECTOR_UNR(8, 0);
  VECTOR_UNR(9, 0);
  VECTOR_UNR(10, 0);
  VECTOR_UNR(11, 0);
  VECTOR_UNR(12, 0);
      // VECTOR_UNR(13, 0);
      // VECTOR_UNR(14, 0);
      // VECTOR_UNR(15, 0);
      // VECTOR_UNR(16, 0);
      // VECTOR_UNR(17, 0);
      // VECTOR_UNR(18, 0);
      // VECTOR_UNR(19, 0);
      // VECTOR_UNR(20, 0);
      // VECTOR_UNR(21, 0);
      // VECTOR_UNR(22, 0);
      // VECTOR_UNR(23, 0);
      // VECTOR_UNR(24, 0);
      // VECTOR_UNR(25, 0);
      // VECTOR_UNR(26, 0);
      // VECTOR_UNR(27, 0);
  }
}

}

struct AddrBeginEnd {
  uint32_t* addr;
  uint32_t b;
  uint32_t e;
};

struct NodesBeginEnd {
  uint32_t node_id;
  uint32_t b;
  uint32_t e;
};

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildLocalHistograms(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, int depth, const uint8_t* numa1, const uint8_t* numa2,
    std::vector<std::vector<double>>* histograms, uint16_t* nodes_ids, std::vector<int32_t>* split_conditions,
    std::vector<bst_uint>* split_ind, const ColumnMatrix *column_matrix, uint64_t* mask, uint64_t* leaf_mask, int max_depth) {
      builder_monitor_.Start("BuildLocalHistograms FULL");
      std::string timer_name = "BuildLocalHistograms:";
      timer_name += std::to_string(depth);

    // if(1 << depth != qexpand_depth_wise_.size()) {
    //   std::cout << "\n\n\n\n2NOT COMPLEATED TREEE!!!: " << depth << " leaf_mask:" << *leaf_mask<< "\n\n\n\n";
    // }
      //CHECK_EQ(1 << depth, qexpand_depth_wise_.size());
  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  const size_t n_bins = gmat.cut.Ptrs().back();
  const size_t n_features = gmat.cut.Ptrs().size() - 1;
static size_t summs[] = {0,0,0,0,0,0,0,0,0};
static size_t average_dist[] = {0,0,0,0,0,0,0,0,0};

//size_t summ_size = 0;
//size_t dist = 0;
//
//for (size_t i = 0; i < n_nodes; ++i) {
//    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
//  summ_size += row_set_collection_[nid].Size();
//  for(size_t j = 0; j < row_set_collection_[nid].Size() - 1; ++j) {
//    CHECK_GE(*(row_set_collection_[nid].begin + j + 1), *(row_set_collection_[nid].begin + j));
//    dist += *(row_set_collection_[nid].begin + j + 1) - *(row_set_collection_[nid].begin + j);
//  }
//}
//summs[depth] += summ_size;
//average_dist[depth] += dist/(summ_size-n_nodes);
  // create space of size (# rows in each node)
  //std::cout << "gmat.row_ptr.size() - 1: " << gmat.row_ptr.size() - 1 << "\n";
  common::BlockedSpace2d space(1, [&](size_t node) {
    //const int32_t nid = nodes_for_explicit_hist_build_[node].nid;
    //return row_set_collection_[nid].Size();
    return gmat.row_ptr.size() - 1;
  }, 4096);
  std::vector<GHistRowT> target_hists(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    target_hists[i] = hist_[nid];
//    const bst_uint fid = tree[nid].SplitIndex();
//    split_inds[i] = fid;
  }

  hist_buffer_.Reset(this->nthread_, n_nodes, space, target_hists);
  int nthreads = this->nthread_;
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);

#pragma omp parallel num_threads(nthreads)
{
 const size_t tid = omp_get_thread_num();
 for (size_t i = 0; i < (1 << depth); ++i) {
   for (size_t bin_id = 0; bin_id <  n_bins*2; ++bin_id) {
     (*histograms)[tid][2*i*n_bins + bin_id] = 0;
   }
 }
}
builder_monitor_.Start("JustPartition!!!!!!");
std::vector<std::vector<uint32_t>> vec(nthreads);
static std::vector<std::vector<uint32_t>> vec_rows(nthreads);
static bool is_compleate_tree = true;
if (depth == 0) {
  is_compleate_tree = true;
}

is_compleate_tree = is_compleate_tree * (1 << depth == qexpand_depth_wise_.size());

std::vector<std::vector<AddrBeginEnd>> threads_addr(nthreads);

static std::vector<int> prev_level_nodes;
std::vector<int> curr_level_nodes(1 << depth, 0);

for(size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
  curr_level_nodes[compleate_trees_depth_wise_[i]] = qexpand_depth_wise_[i].nid;
}
if(depth > 0) {
if(depth < max_depth) {

  if (is_compleate_tree) {
    #pragma omp parallel num_threads(nthreads)
      {
          size_t tid = omp_get_thread_num();
          const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
          size_t chunck_size =
              num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

          size_t begin = chunck_size * tid;
          size_t end = std::min(begin + chunck_size, num_blocks_in_space);
          uint64_t local_time_alloc = 0;
          vec[tid].resize(64);
          vec_rows[tid].resize(4096*(end - begin));
          uint32_t count = 0;
          for (auto i = begin; i < end; i++) {
            common::Range1d r = space.GetRange(i);
            JustPartition(r.begin(), r.end(), gmat, n_features,
                          vec[tid].data(), vec_rows[tid].data(), count, numa,
                          nodes_ids, split_conditions, split_ind, mask);//, column_matrix);
          }
          vec_rows[tid][0] = count;
      }
  } else {
    #pragma omp parallel num_threads(nthreads)
      {
          size_t tid = omp_get_thread_num();
          const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
          size_t chunck_size =
              num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

          size_t begin = chunck_size * tid;
          size_t end = std::min(begin + chunck_size, num_blocks_in_space);
          uint64_t local_time_alloc = 0;
          vec[tid].resize(64);
          vec_rows[tid].resize(4096*(end - begin));
          uint32_t count = 0;
          for (auto i = begin; i < end; i++) {
            common::Range1d r = space.GetRange(i);
            JustPartitionWithLeafsMask(r.begin(), r.end(), gmat, n_features,
                          vec[tid].data(), vec_rows[tid].data(), count, numa,
                          nodes_ids, split_conditions, split_ind, mask, leaf_mask, &prev_level_nodes);

          }
          vec_rows[tid][0] = count;
      }
  }
    uint32_t summ_size1 = 0;

    for(uint32_t i = 0; i < nthreads; ++i) {
      summ_size1 += vec_rows[i][0];
    }

    // std::cout << "summ_size1: " << summ_size1 << std::endl;
    uint32_t block_size = summ_size1/nthreads + !!(summ_size1%nthreads);
    // std::cout << "block_size: " << block_size << std::endl;
    uint32_t curr_vec_rows_id = 0;
    uint32_t curr_vec_rows_size = vec_rows[curr_vec_rows_id][0];
    uint32_t curr_thread_size = block_size;
    for(uint32_t i = 0; i < nthreads; ++i) {
      // std::cout << "curr_thread_size: " << curr_thread_size << std::endl;
      while (curr_thread_size != 0) {
        // std::cout << "  curr_vec_rows_size: " << curr_vec_rows_size << std::endl;
        // std::cout << "  curr_vec_rows_id: " << curr_vec_rows_id << std::endl;
        // std::cout << "  curr_thread_size: " << curr_thread_size << std::endl;
        if(curr_vec_rows_size > curr_thread_size) {
          threads_addr[i].push_back({vec_rows[curr_vec_rows_id].data(),
                                    1 + vec_rows[curr_vec_rows_id][0] - curr_vec_rows_size,
                                    1 + vec_rows[curr_vec_rows_id][0] - curr_vec_rows_size + curr_thread_size});
          curr_vec_rows_size -= curr_thread_size;
          curr_thread_size = 0;
        } else if (curr_vec_rows_size == curr_thread_size) {
          threads_addr[i].push_back({vec_rows[curr_vec_rows_id].data(),
                                    1 + vec_rows[curr_vec_rows_id][0] - curr_vec_rows_size,
                                    1 + vec_rows[curr_vec_rows_id][0] - curr_vec_rows_size + curr_thread_size});
          curr_vec_rows_id += (curr_vec_rows_id < (nthreads - 1));
          curr_vec_rows_size = vec_rows[curr_vec_rows_id][0];
          curr_thread_size = 0;
        } else {
          threads_addr[i].push_back({vec_rows[curr_vec_rows_id].data(),
                                    1 + vec_rows[curr_vec_rows_id][0] - curr_vec_rows_size,
                                    1 + vec_rows[curr_vec_rows_id][0]});
          curr_thread_size -= curr_vec_rows_size;
          curr_vec_rows_id += (curr_vec_rows_id < (nthreads - 1));
          curr_vec_rows_size = vec_rows[curr_vec_rows_id][0];
        }
      }
      // std::cout << "thread: " << i << ":";
      // for (size_t j = 0; j < threads_addr[i].size(); ++j) {
      //   std::cout << threads_addr[i][j].e - threads_addr[i][j].b << "   ";
      // }
      //std::cout << "\n";
      curr_thread_size = std::min(block_size, summ_size1 - block_size*(i+1));
    }
} else {
    #pragma omp parallel num_threads(nthreads)
      {
          size_t tid = omp_get_thread_num();
          const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
          size_t chunck_size =
              num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

          size_t begin = chunck_size * tid;
          size_t end = std::min(begin + chunck_size, num_blocks_in_space);
          vec[tid].resize(64);
          vec_rows[tid].resize(4096*(end - begin));
          uint64_t local_time_alloc = 0;
          uint32_t count = 0;
          for (auto i = begin; i < end; i++) {
            common::Range1d r = space.GetRange(i);
            JustPartitionLastLayer(r.begin(), r.end(), gmat, n_features,
                          vec[tid].data(), vec_rows[tid].data(), count, numa,
                          nodes_ids, split_conditions, split_ind, &curr_level_nodes, leaf_mask, &prev_level_nodes);
          }
          vec_rows[tid][0] = count;
      }

}


}
prev_level_nodes = curr_level_nodes;

  builder_monitor_.Stop("JustPartition!!!!!!");

if(depth < max_depth) {
  builder_monitor_.Start(timer_name);

//  // Parallel processing by nodes and data in each node
//  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
//    const auto tid = static_cast<unsigned>(omp_get_thread_num());
//    const int32_t nid = nodes_for_explicit_hist_build_[nid_in_set].nid;
//
//    auto start_of_row_set = row_set_collection_[nid].begin;
//    auto rid_set = RowSetCollection::Elem(start_of_row_set + r.begin(),
//                                      start_of_row_set + r.end(),
//                                      nid);
//    BuildHist(gpair_h, rid_set, gmat, gmatb, hist_buffer_.GetInitializedHist(tid, nid_in_set));
//  });




auto func = [&](size_t nid_in_set, common::Range1d r) {
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
      const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
    const int32_t nid = nodes_for_explicit_hist_build_[nid_in_set].nid;

    auto start_of_row_set = row_set_collection_[nid].begin;
    auto rid_set = RowSetCollection::Elem(start_of_row_set + r.begin(),
                                      start_of_row_set + r.end(),
                                      nid);
    BuildHist(gpair_h, rid_set, gmat, gmatb, hist_buffer_.GetInitializedHist(tid, nid_in_set), gmat.index.data<uint8_t>());
  };

std::vector<std::vector<uint64_t>> treads_times(nthreads);
const uint64_t t1 = get_time();

//  dmlc::OMPException omp_exc;
  //for(size_t tid = 0; tid < nthreads; ++tid)
  if(depth == 0) {
#pragma omp parallel num_threads(nthreads)
  {
    //std::cout << "\nTID1!\n";
/*    omp_exc.Run(
        [&treads_times](size_t num_blocks_in_space, const common::BlockedSpace2d& space, int nthreads, auto& func) {*/
      size_t tid = omp_get_thread_num();
      const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
      treads_times[tid].resize(1,0);
      uint64_t& local_time = treads_times[tid][0];
      //const uint64_t t1 = get_time();
      size_t chunck_size =
          num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

      size_t begin = chunck_size * tid;
      size_t end = std::min(begin + chunck_size, num_blocks_in_space);
      uint64_t local_time_alloc = 0;
      for (auto i = begin; i < end; i++) {
        size_t nid_in_set = space.GetFirstDimension(i); common::Range1d r = space.GetRange(i);
        //const auto tid = static_cast<unsigned>(omp_get_thread_num());
        const int32_t nid = nodes_for_explicit_hist_build_[nid_in_set].nid;
        auto start_of_row_set = row_set_collection_[nid].begin;
        auto rid_set = RowSetCollection::Elem(start_of_row_set + r.begin(),
                                              start_of_row_set + r.end(),
                                              nid);
    //std::cout << "\nTID2!\n";
    GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
//     if(depth != 0) {
// //        BuildHistKernel<GradientSumT, true, uint8_t>(gpair_h, rid_set1, gmat, n_features,  hist_buffer_.GetInitializedHist(tid, nid_in_set), numa, nodes_ids);
//    // std::cout << "\nTID3!\n";
//         BuildHistKernel<GradientSumT, false, uint8_t>(gpair_h, r.begin(), r.end(), gmat, n_features,
//         local_hist, numa, nodes_ids, split_conditions, split_ind);
//     } else {
        BuildHistKernel<GradientSumT, false, uint8_t>(gpair_h, r.begin(), r.end(), gmat, n_features,  local_hist, numa, nodes_ids);
//    }
    //std::cout << "\nTID4!\n";
        //GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data() + 2*nid_in_set*n_bins), n_bins);
        //BuildHist(gpair_h, rid_set, gmat, gmatb,  hist_buffer_.GetInitializedHist(tid, nid_in_set), numa);
      }
      local_time = get_time();
//    }, num_blocks_in_space, space, nthreads, func);
  }
  } else {

#pragma omp parallel num_threads(nthreads)
  {
      size_t tid = omp_get_thread_num();
      const uint8_t* numa = tid < nthreads/2 ? numa1 : numa2;
      treads_times[tid].resize(1,0);
      uint64_t& local_time = treads_times[tid][0];
      std::vector<AddrBeginEnd>& local_thread_addr = threads_addr[tid];
      GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
      for(uint32_t block_id = 0; block_id < local_thread_addr.size(); ++block_id) {
        const uint32_t* rows = local_thread_addr[block_id].addr + local_thread_addr[block_id].b;
        const uint32_t size_r = local_thread_addr[block_id].e - local_thread_addr[block_id].b;
        BuildHistKernel<GradientSumT, false, uint8_t>(gpair_h, rows, size_r, gmat, n_features,
                                                      local_hist, numa, nodes_ids, 1 << depth);

      }
      local_time = get_time();
  }

  }

//  omp_exc.Rethrow();

static std::vector<std::vector<uint64_t> > times(9, std::vector<uint64_t>(nthreads, 0));
static int n_call = 0;

for(int i = 0; i < nthreads; ++i) {
  times[depth][i] += treads_times[i][0] - t1;
}

if(++n_call == N_CALL/5) {
  std::cout << "\n";
  for(int di = 0; di < 9; ++di) {
    std::cout << "!depth "  << di << ": " << summs[di] << "\n";
  }
  for(int di = 0; di < 9; ++di) {
    std::cout << "!average_dist "  << di << ": " << average_dist[di] << "\n";
  }
  std::cout << "\nBuildLocalHist: " << N_CALL <<"\n";
  for(int di = 0; di < 9; ++di) {
    std::cout << "depth " << di << ": ";
    for(int i = 0; i < nthreads; ++i) {
      std::cout << (double)(times[di][i])/1000000000 << "\t";
    }
    std::cout << std::endl;
  }
}
  builder_monitor_.Stop(timer_name);

if(depth == 0) {
  for (size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
   const int32_t nid = qexpand_depth_wise_[i].nid;
   //target_hists[i] = hist_[nid];
   GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[nid].data());
   for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
     dest_hist[bin_id] = (*histograms)[0][2*i*n_bins + bin_id];
   }
   for (size_t tid = 1; tid < nthreads; ++tid) {
     for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
       dest_hist[bin_id] += (*histograms)[tid][2*i*n_bins + bin_id];
     }
   }
  }
} else {
  std::vector<size_t> smallest;
  std::vector<size_t> largest;
  for (size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
   const int32_t nid_c = compleate_trees_depth_wise_[i];
   if(((uint64_t)(1) << (nid_c%64)) & *(mask + nid_c/64)) {
     smallest.push_back(i);
   } else {
     largest.push_back(i);
   }
  }

// if(smallest.size() == 1) {
//   for (size_t i = 0; i < smallest.size(); ++i) {
//    const int32_t nid_c = compleate_trees_depth_wise_[smallest[i]];
//    const int32_t nid = qexpand_depth_wise_[smallest[i]].nid;
//     GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[nid].data());
//     for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
//       dest_hist[bin_id] = (*histograms)[0][2*nid_c*n_bins + bin_id];
//     }
//     for (size_t tid = 1; tid < nthreads; ++tid) {
//       for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
//         dest_hist[bin_id] += (*histograms)[tid][2*nid_c*n_bins + bin_id];
//       }
//     }
//   }
// } else {
    const uint32_t summ_size_bin = n_bins*smallest.size();
    uint32_t block_size = summ_size_bin/nthreads + !!(summ_size_bin%nthreads);
    std::vector<std::vector<NodesBeginEnd>> threads_work(nthreads);
    // std::cout << "block_size: " << block_size << std::endl;
    // std::cout << "n_bins: " << n_bins << std::endl;
    const uint32_t node_full_size = n_bins;
    uint32_t curr_node_id = 0;
    uint32_t curr_node_size = node_full_size;
    uint32_t curr_thread_size = block_size;
    for(uint32_t i = 0; i < nthreads; ++i) {
      // std::cout << "i : " << i << std::endl;
      // std::cout << "curr_thread_size : " << curr_thread_size << std::endl;
      // std::cout << "curr_node_size : " << curr_node_size << std::endl;
      while (curr_thread_size != 0) {
        if(curr_node_size > curr_thread_size) {
          CHECK_LT(curr_node_id, smallest.size());
          threads_work[i].push_back({curr_node_id, node_full_size - curr_node_size,
                                    node_full_size - curr_node_size + curr_thread_size});
          curr_node_size -= curr_thread_size;
          curr_thread_size = 0;
        } else if (curr_node_size == curr_thread_size) {
          CHECK_LT(curr_node_id, smallest.size());
          threads_work[i].push_back({curr_node_id,
                                    node_full_size - curr_node_size,
                                    node_full_size - curr_node_size + curr_thread_size});
          curr_node_id++;//= (curr_node_id < (nthreads - 1));
          curr_node_size = node_full_size;
          curr_thread_size = 0;
        } else {
          CHECK_LT(curr_node_id, smallest.size());
          threads_work[i].push_back({curr_node_id,
                                    node_full_size - curr_node_size,
                                    node_full_size});
          curr_thread_size -= curr_node_size;
          curr_node_id++;//(curr_node_id < (nthreads - 1));
          curr_node_size = node_full_size;
        }
      }
      curr_thread_size = std::min(block_size, summ_size_bin > block_size*(i+1) ? summ_size_bin - block_size*(i+1) : 0);
    }
#pragma omp parallel num_threads(nthreads)
  {
      size_t tid = omp_get_thread_num();
      for(size_t i = 0; i < threads_work[tid].size(); ++i) {
        const size_t begin = threads_work[tid][i].b * 2;
        const size_t end = threads_work[tid][i].e * 2;

        const int32_t nid_c = compleate_trees_depth_wise_[smallest[threads_work[tid][i].node_id]];
        const int32_t nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].nid;
        GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[nid].data());
        for (size_t bin_id = begin; bin_id < end; ++bin_id) {
          dest_hist[bin_id] = (*histograms)[0][2*nid_c*n_bins + bin_id];
        }
        for (size_t tid = 1; tid < nthreads; ++tid) {
          for (size_t bin_id = begin; bin_id < end; ++bin_id) {
            dest_hist[bin_id] += (*histograms)[tid][2*nid_c*n_bins + bin_id];
          }
        }
      }
  }
//}


  // for (size_t i = 0; i < smallest.size(); ++i) {
  //  const int32_t nid_c = compleate_trees_depth_wise_[smallest[i]];
  //  const int32_t nid = qexpand_depth_wise_[smallest[i]].nid;
  //   GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[nid].data());
  //   for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
  //     dest_hist[bin_id] = (*histograms)[0][2*nid_c*n_bins + bin_id];
  //   }
  //   for (size_t tid = 1; tid < nthreads; ++tid) {
  //     for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
  //       dest_hist[bin_id] += (*histograms)[tid][2*nid_c*n_bins + bin_id];
  //     }
  //   }
  // }
#pragma omp parallel num_threads(nthreads)
  {
      size_t tid = omp_get_thread_num();
      for(size_t i = 0; i < threads_work[tid].size(); ++i) {
        const size_t begin = threads_work[tid][i].b * 2;
        const size_t end = threads_work[tid][i].e * 2;

        const int32_t small_nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].nid;
        const int32_t largest_nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].sibling_nid;
        const size_t parent_id = (*p_tree)[small_nid].Parent();

        GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[largest_nid].data());
        GradientSumT* parent_hist = reinterpret_cast< GradientSumT*>(hist_[parent_id].data());
        GradientSumT* small_hist = reinterpret_cast< GradientSumT*>(hist_[small_nid].data());
        for (size_t bin_id = begin; bin_id < end; ++bin_id) {
          dest_hist[bin_id] = parent_hist[bin_id] - small_hist[bin_id];
        }
      }
  }


CHECK_EQ(smallest.size(), largest.size());
// for(size_t i = 0; i < largest.size(); ++i) {
//   const int32_t small_nid = qexpand_depth_wise_[smallest[i]].nid;
//   const int32_t largest_nid = qexpand_depth_wise_[smallest[i]].sibling_nid;
//   const size_t parent_id = (*p_tree)[small_nid].Parent();

//   GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[largest_nid].data());
//   GradientSumT* parent_hist = reinterpret_cast< GradientSumT*>(hist_[parent_id].data());
//   GradientSumT* small_hist = reinterpret_cast< GradientSumT*>(hist_[small_nid].data());
//   for (size_t bin_id = 0; bin_id < n_bins*2; ++bin_id) {
//     dest_hist[bin_id] = parent_hist[bin_id] - small_hist[bin_id];
//   }
// }

}
}

builder_monitor_.Stop("BuildLocalHistograms FULL");

}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildNodeStats(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, uint64_t* mask, int n_call) {
  builder_monitor_.Start("BuildNodeStats");
  int i = 0;
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gmat, gpair_h, *p_fmat, *p_tree, compleate_trees_depth_wise_[i], mask);
    ++i;
    // add constraints
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      // it's a right child
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_feature_id = snode_[parent_id].best.SplitIndex();
      // if (n_call == 175 || n_call == 174 || n_call == 173) {
      //   std::cout << "parent_id: " << parent_id << " rnid: " << nid << " parent_split_feature_id: " <<
      //   parent_split_feature_id << " left.weight: " << snode_[left_sibling_id].weight << std::endl;
      // }

      tree_evaluator_.AddSplit(
          parent_id, left_sibling_id, nid, parent_split_feature_id,
          snode_[left_sibling_id].weight, snode_[nid].weight);
      interaction_constraints_.Split(parent_id, parent_split_feature_id,
                                     left_sibling_id, nid);
    }
  }
  builder_monitor_.Stop("BuildNodeStats");
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const GHistIndexMatrix &gmat,
          RegTree *p_tree,
          int *num_leaves,
          int depth,
          unsigned *timestamp,
          std::vector<ExpandEntry>* nodes_for_apply_split,
          std::vector<ExpandEntry>* temp_qexpand_depth, std::vector<uint16_t>* compleate_tmp, uint64_t* leaf_mask, int n_call, std::vector<uint16_t>* compleate_splits ) {
        //    std::cout << "\nAddSplitsToTree: ";
  auto evaluator = tree_evaluator_.GetEvaluator();
  size_t i = 0;
  CHECK_EQ(compleate_trees_depth_wise_.size(), qexpand_depth_wise_.size());
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
     // std::cout << " construct leaf :(" << i << ") ";
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      *(leaf_mask + i/64) |= ((uint64_t)(1) << (compleate_trees_depth_wise_[i] % 64));
    } else {
    //  std::cout << " construct split :{" << i << ") ";
      nodes_for_apply_split->push_back(entry);
      compleate_splits->push_back(compleate_trees_depth_wise_[i]);
      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());
// if (n_call == 175 || n_call == 174 || n_call == 173)
// std::cout << "nid: " << nid << " e.best.SplitIndex(): " << e.best.SplitIndex() << " e.best.split_value: "
//  << e.best.split_value << " e.weight: " <<  e.weight << " left_leaf_weight:" << left_leaf_weight
//   << " right_leaf_weight: " << right_leaf_weight << std::endl;
      int left_id = (*p_tree)[nid].LeftChild();
      int right_id = (*p_tree)[nid].RightChild();
      temp_qexpand_depth->push_back(ExpandEntry(left_id, right_id,
                                                p_tree->GetDepth(left_id), 0.0, (*timestamp)++));
      temp_qexpand_depth->push_back(ExpandEntry(right_id, left_id,
                                                p_tree->GetDepth(right_id), 0.0, (*timestamp)++));
      compleate_tmp->push_back(2*compleate_trees_depth_wise_[i]);
      compleate_tmp->push_back(2*compleate_trees_depth_wise_[i] + 1);
      // - 1 parent + 2 new children
      (*num_leaves)++;
    }
    ++i;
  }
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::EvaluateAndApplySplits(
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry> *temp_qexpand_depth,
    std::vector<uint16_t>* compleate_tmp, uint64_t* leaf_mask,
    std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, int n_call) {
//std::cout << "!EvaluateAndApplySplits start " << std::endl;


//std::cout << "\nEvaluateSplits start " << std::endl;
  EvaluateSplits(qexpand_depth_wise_, gmat, hist_, *p_tree);
//std::cout << "EvaluateSplits finished " << std::endl;

  std::vector<ExpandEntry> nodes_for_apply_split;
  std::vector<uint16_t> compleate_apply_split;
  AddSplitsToTree(gmat, p_tree, num_leaves, depth, timestamp,
                  &nodes_for_apply_split, temp_qexpand_depth, compleate_tmp, leaf_mask, n_call, &compleate_apply_split);
//std::cout << "AddSplitsToTree finished " << std::endl;
  ApplySplit(nodes_for_apply_split, gmat, column_matrix, hist_, p_tree, depth, split_conditions, split_ind, &compleate_apply_split);
//std::cout << "ApplySplit finished " << std::endl;
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SplitSiblings(
    const std::vector<ExpandEntry> &nodes,
    std::vector<ExpandEntry> *small_siblings,
    std::vector<ExpandEntry> *big_siblings, RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  for (auto const& entry : nodes) {
    int nid = entry.nid;
    RegTree::Node &node = (*p_tree)[nid];
    if (node.IsRoot()) {
      small_siblings->push_back(entry);
    } else {
      const int32_t left_id = (*p_tree)[node.Parent()].LeftChild();
      const int32_t right_id = (*p_tree)[node.Parent()].RightChild();

      if (nid == left_id && row_set_collection_[left_id ].Size() <
                            row_set_collection_[right_id].Size()) {
        small_siblings->push_back(entry);
      } else if (nid == right_id && row_set_collection_[right_id].Size() <=
                                    row_set_collection_[left_id ].Size()) {
        small_siblings->push_back(entry);
      } else {
        big_siblings->push_back(entry);
      }
    }
  }
  builder_monitor_.Stop("SplitSiblings");
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ExpandWithDepthWise(
  const GHistIndexMatrix &gmat,
  const GHistIndexBlockMatrix &gmatb,
  const ColumnMatrix &column_matrix,
  DMatrix *p_fmat,
  RegTree *p_tree,
  const std::vector<GradientPair> &gpair_h, const uint8_t* numa1, const uint8_t* numa2, std::vector<std::vector<double>>* histograms) {
  unsigned timestamp = 0;
  int num_leaves = 0;
      qexpand_depth_wise_.clear();
      compleate_trees_depth_wise_.clear();
  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(ExpandEntry::kRootNid), 0.0, timestamp++));
  compleate_trees_depth_wise_.emplace_back(0);
  ++num_leaves;
  node_ids.resize(row_set_collection_[0].Size(),0);
  std::vector<bst_uint> split_indexs(1 << param_.max_depth + 1);
  std::vector<int32_t> split_values(1 << param_.max_depth + 1);
//std::cout << "split_conditions.size(): " << split_values.size() << " split_ind.size(): " << split_indexs.size() << std::endl;

    uint64_t leafs_mask[] = {0,0,0,0};

static uint64_t n_call = 0;
++n_call;
  // if(n_call == 175) {
  //   std::cout << "764 gh: " << gpair_h[764] << std::endl;
  // }
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    std::vector<ExpandEntry> temp_qexpand_depth;
    std::vector<uint16_t> tmp_compleate_trees_depth;
//    std::cout << "SplitSiblings started!" << std::endl;
    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
//    std::cout << "SplitSiblings finished!!!" << std::endl;
    hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
//    std::cout << "AddHistRows finished!" << std::endl;
    uint64_t mask[] = {0,0,0,0};
if(depth > 0) {
//if(depth < param_.max_depth) {

    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h, mask, n_call);
    // if(1 << depth != qexpand_depth_wise_.size()) {
    //   std::cout << "\n\n\n\nNOT COMPLEATED TREEE!!!" << n_call << " depth: " << depth << " mask: " << (long long)mask << " qexpand_depth_wise_.size(): " << qexpand_depth_wise_.size() << "\n\n\n\n";
    // }
    // std::cout << "mask: " << (long long)mask << std::endl;
    //if(depth <  param_.max_depth) {
    BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h, depth, numa1, numa2, histograms, node_ids.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth);
leafs_mask[0] = 0;
leafs_mask[1] = 0;
leafs_mask[2] = 0;
leafs_mask[3] = 0;

    hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
    //}
// } else {
//     BuildNodeStats(gmat, p_fmat, p_tree, gpair_h);
//     nodes_for_subtraction_trick_.clear();
//     nodes_for_explicit_hist_build_.clear();
//     qexpand_depth_wise_.clear();
//     temp_qexpand_depth.clear();
//     break;
// }
} else {
    BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h, depth, numa1, numa2, histograms, node_ids.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth);

//    std::cout << "BuildLocalHistograms finished!" << std::endl;
    hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
    // for (size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
    //     const int32_t nid = qexpand_depth_wise_[i].nid;
    //     std::cout << i << ":" << nid << " : ";
    //     for (size_t j = 0; j < hist_[nid].size(); ++j)
    //       std::cout << hist_[nid][j] << "  ";
    //     std::cout << std::endl;
    // }

    //std::cout << "SyncHistograms finished!" << std::endl;
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h);
}
//    std::cout << "BuildNodeStats finished!" << std::endl;


// if (depth == param_.max_depth) {
//     nodes_for_subtraction_trick_.clear();
//     nodes_for_explicit_hist_build_.clear();
//     //qexpand_depth_wise_.clear();
//     temp_qexpand_depth.clear();
//     break;
// }
    EvaluateAndApplySplits(gmat, column_matrix, p_tree, &num_leaves, depth, &timestamp,
                   &temp_qexpand_depth, &tmp_compleate_trees_depth, leafs_mask, &split_values, &split_indexs, n_call);
    // clean up
    //qexpand_depth_wise_.clear();
    //std::cout << "    qexpand_depth_wise_.clear() finished! " << std::endl;
    nodes_for_subtraction_trick_.clear();
    //std::cout << "    nodes_for_subtraction_trick_.clear(); finished! " << std::endl;
    nodes_for_explicit_hist_build_.clear();
    //std::cout << "    nodes_for_explicit_hist_build_.clear(); finished! " << std::endl;
    if (temp_qexpand_depth.empty()) {
      qexpand_depth_wise_.clear();
      compleate_trees_depth_wise_.clear();
      break;
    } else {
      qexpand_depth_wise_.clear();
      qexpand_depth_wise_ = temp_qexpand_depth;
      compleate_trees_depth_wise_.clear();
      compleate_trees_depth_wise_ = tmp_compleate_trees_depth;
      tmp_compleate_trees_depth.clear();
      temp_qexpand_depth.clear();
    }
    //std::cout << "    temp_qexpand_depth.clear(); finished! " << std::endl;
//    std::cout << "\nIteration compleated" << std::endl;
  }
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ExpandWithLossGuide(
    const GHistIndexMatrix& gmat,
    const GHistIndexBlockMatrix& gmatb,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandWithLossGuide");
  unsigned timestamp = 0;
  int num_leaves = 0;

  ExpandEntry node(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(0), 0.0f, timestamp++);
  BuildHistogramsLossGuide(node, gmat, gmatb, p_tree, gpair_h);

  this->InitNewNode(ExpandEntry::kRootNid, gmat, gpair_h, *p_fmat, *p_tree);

  this->EvaluateSplits({node}, gmat, hist_, *p_tree);
  node.loss_chg = snode_[ExpandEntry::kRootNid].best.loss_chg;

  qexpand_loss_guided_->push(node);
  ++num_leaves;

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int nid = candidate.nid;
    qexpand_loss_guided_->pop();
    if (candidate.IsValid(param_, num_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      auto evaluator = tree_evaluator_.GetEvaluator();
      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      this->ApplySplit({candidate}, gmat, column_matrix, hist_, p_tree);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();

      ExpandEntry left_node(cleft, cright, p_tree->GetDepth(cleft),
                            0.0f, timestamp++);
      ExpandEntry right_node(cright, cleft, p_tree->GetDepth(cright),
                            0.0f, timestamp++);

      if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
        BuildHistogramsLossGuide(left_node, gmat, gmatb, p_tree, gpair_h);
      } else {
        BuildHistogramsLossGuide(right_node, gmat, gmatb, p_tree, gpair_h);
      }

      this->InitNewNode(cleft, gmat, gpair_h, *p_fmat, *p_tree);
      this->InitNewNode(cright, gmat, gpair_h, *p_fmat, *p_tree);
      bst_uint featureid = snode_[nid].best.SplitIndex();
      tree_evaluator_.AddSplit(nid, cleft, cright, featureid,
                               snode_[cleft].weight, snode_[cright].weight);
      interaction_constraints_.Split(nid, featureid, cleft, cright);

      this->EvaluateSplits({left_node, right_node}, gmat, hist_, *p_tree);
      left_node.loss_chg = snode_[cleft].best.loss_chg;
      right_node.loss_chg = snode_[cright].best.loss_chg;

      qexpand_loss_guided_->push(left_node);
      qexpand_loss_guided_->push(right_node);

      ++num_leaves;  // give two and take one, as parent is no longer a leaf
    }
  }
  builder_monitor_.Stop("ExpandWithLossGuide");
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(
    const GHistIndexMatrix &gmat, const GHistIndexBlockMatrix &gmatb,
    const ColumnMatrix &column_matrix, HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat, RegTree *p_tree, const uint8_t* numa1, const uint8_t* numa2, std::vector<std::vector<double>>* histograms) {
  builder_monitor_.Start("Update");

  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();

  tree_evaluator_ =
      TreeEvaluator(param_, p_fmat->Info().num_col_, GenericParameter::kCpuId);
  interaction_constraints_.Reset();

  this->InitData(gmat, gpair_h, *p_fmat, *p_tree);
  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  } else {
    N_CALL = (param_.max_depth + 1) * 500;
    ExpandWithDepthWise(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h, numa1, numa2, histograms);
  }

  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
    p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.GetHess());
  }
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});
  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* p_out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");
static int n_call = 0;
++n_call;
  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

  if (leaf_value_cache_.empty()) {
    leaf_value_cache_.resize(p_last_tree_->param.num_nodes,
                             std::numeric_limits<float>::infinity());
  }

  CHECK_GT(out_preds.size(), 0U);

  size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();
  common::BlockedSpace2d space(1, [&](size_t node) {
    return row_set_collection_[0].Size();
  }, 1024);

  common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
    const RowSetCollection::Elem rowset = row_set_collection_[0];
    for (size_t it = r.begin(); it <  r.end(); ++it) {
      bst_float leaf_value;
      // if a node is marked as deleted by the pruner, traverse upward to locate
      // a non-deleted leaf.
      int nid = (~((uint16_t)(1) << 15)) & node_ids[it];
      if ((*p_last_tree_)[nid].IsDeleted()) {
        while ((*p_last_tree_)[nid].IsDeleted()) {
          nid = (*p_last_tree_)[nid].Parent();
        }
        CHECK((*p_last_tree_)[nid].IsLeaf());
      }
      leaf_value = (*p_last_tree_)[nid].LeafValue();
// if(it == 764 && n_call == 174) {
//   std::cout << "leaf_value: " << leaf_value << " nid: " << nid << " old out_preds[it]: " << out_preds[it] << std::endl;
// }
      out_preds[it] += leaf_value;
     // gpair_h_ptr[it] = common::Sigmoid(out_preds[it]) - labels[it];
      // + 1*common::Sigmoid(out_preds[it]);
// if(it == 764 && n_call == 174) {
//   std::cout << " new out_preds[it]: " << out_preds[it] << std::endl;
// }
    }
    // if (rowset.begin != nullptr && rowset.end != nullptr) {
    //   int nid = rowset.node_id;
    //   //int nid = qexpand_depth_wise_[0].nid;
    //  // CHECK(nid == 0);
    //   bst_float leaf_value;
    //   // if a node is marked as deleted by the pruner, traverse upward to locate
    //   // a non-deleted leaf.
    //   if ((*p_last_tree_)[nid].IsDeleted()) {
    //     while ((*p_last_tree_)[nid].IsDeleted()) {
    //       nid = (*p_last_tree_)[nid].Parent();
    //     }
    //     CHECK((*p_last_tree_)[nid].IsLeaf());
    //   }
    //   leaf_value = (*p_last_tree_)[nid].LeafValue();

    //   for (const size_t* it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
    //     out_preds[*it] += 0;
    //   }
    // }
  });

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const std::vector<GradientPair>& gpair,
                                                const DMatrix& fmat,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<size_t>& row_indices_local = *row_indices;
  size_t* p_row_indices = row_indices_local.data();
#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  size_t j = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
      p_row_indices[j++] = i;
    }
  }
  /* resize row_indices to reduce memory */
  row_indices_local.resize(j);
#else
  const size_t nthread = this->nthread_;
  std::vector<size_t> row_offsets(nthread, 0);
  /* usage of mt19937_64 give 2x speed up for subsampling */
  std::vector<std::mt19937> rnds(nthread);
  /* create engine for each thread */
  for (std::mt19937& r : rnds) {
    r = rnd;
  }
  const size_t discard_size = info.num_row_ / nthread;
  auto upper_border = static_cast<float>(std::numeric_limits<uint32_t>::max());
  uint32_t coin_flip_border = static_cast<uint32_t>(upper_border * param_.subsample);
  #pragma omp parallel num_threads(nthread)
  {
    const size_t tid = omp_get_thread_num();
    const size_t ibegin = tid * discard_size;
    const size_t iend = (tid == (nthread - 1)) ?
                        info.num_row_ : ibegin + discard_size;

    rnds[tid].discard(discard_size * tid);
    for (size_t i = ibegin; i < iend; ++i) {
      if (gpair[i].GetHess() >= 0.0f && rnds[tid]() < coin_flip_border) {
        p_row_indices[ibegin + row_offsets[tid]++] = i;
      }
    }
  }
  /* discard global engine */
  rnd = rnds[nthread - 1];
  size_t prefix_sum = row_offsets[0];
  for (size_t i = 1; i < nthread; ++i) {
    const size_t ibegin = i * discard_size;

    for (size_t k = 0; k < row_offsets[i]; ++k) {
      row_indices_local[prefix_sum + k] = row_indices_local[ibegin + k];
    }
    prefix_sum += row_offsets[i];
  }
  /* resize row_indices to reduce memory */
  row_indices_local.resize(prefix_sum);
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const DMatrix& fmat,
                                          const RegTree& tree) {
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  {
    // initialize the row set
    row_set_collection_.Clear();
    // clear local prediction cache
    leaf_value_cache_.clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(nbins);
    hist_local_worker_.Init(nbins);
    hist_buffer_.Init(nbins);

    // initialize histogram builder
#pragma omp parallel
    {
      this->nthread_ = omp_get_num_threads();
    }
    hist_builder_ = GHistBuilder<GradientSumT>(this->nthread_, nbins);

    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(info.num_row_);
    size_t* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, fmat, &row_indices);
    } else {
      MemStackAllocator<bool, 128> buff(this->nthread_);
      bool* p_buff = buff.Get();
      std::fill(p_buff, p_buff + this->nthread_, false);

      const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

      // #pragma omp parallel num_threads(this->nthread_)
      // {
      //   const size_t tid = omp_get_thread_num();
      //   const size_t ibegin = tid * block_size;
      //   const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
      //       static_cast<size_t>(info.num_row_));

      //   for (size_t i = ibegin; i < iend; ++i) {
      //     if (gpair[i].GetHess() < 0.0f) {
      //       p_buff[tid] = true;
      //       break;
      //     }
      //   }
      // }

      // bool has_neg_hess = false;
      // for (int32_t tid = 0; tid < this->nthread_; ++tid) {
      //   if (p_buff[tid]) {
      //     has_neg_hess = true;
      //   }
      // }

      // if (has_neg_hess) {
      //   size_t j = 0;
      //   for (size_t i = 0; i < info.num_row_; ++i) {
      //     if (gpair[i].GetHess() >= 0.0f) {
      //       p_row_indices[j++] = i;
      //     }
      //   }
      //   row_indices.resize(j);
      // } else {
      //   #pragma omp parallel num_threads(this->nthread_)
      //   {
      //     const size_t tid = omp_get_thread_num();
      //     const size_t ibegin = tid * block_size;
      //     const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
      //         static_cast<size_t>(info.num_row_));
      //     for (size_t i = ibegin; i < iend; ++i) {
      //      p_row_indices[i] = i;
      //     }
      //   }
      // }
    }
  }

  row_set_collection_.Init();

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
    if (nrow * ncol == nnz) {
      // dense data with zero-based indexing
      data_layout_ = DataLayout::kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = DataLayout::kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = DataLayout::kSparseData;
    }
  }
  // store a pointer to the tree
  p_last_tree_ = &tree;
  if (data_layout_ == DataLayout::kDenseDataOneBased) {
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, true);
  } else {
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);
  }
  if (data_layout_ == DataLayout::kDenseDataZeroBased
      || data_layout_ == DataLayout::kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
    const auto nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
    uint32_t min_nbins_per_feature = 0;
    for (bst_uint i = 0; i < nfeature; ++i) {
      const uint32_t nbins = row_ptr[i + 1] - row_ptr[i];
      if (nbins > 0) {
        if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
          min_nbins_per_feature = nbins;
          fid_least_bins_ = i;
        }
      }
    }
    CHECK_GT(min_nbins_per_feature, 0U);
  }
  {
    snode_.reserve(256);
    snode_.clear();
  }
  {
    if (param_.grow_policy == TrainParam::kLossGuide) {
      qexpand_loss_guided_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_depth_wise_.clear();
    }
  }
  builder_monitor_.Stop("InitData");
}

// if sum of statistics for non-missing values in the node
// is equal to sum of statistics for all values:
// then - there are no missing values
// else - there are missing values
template <typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::SplitContainsMissingValues(
    const GradStats e, const NodeEntry &snode) {
  if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
    return false;
  } else {
    return true;
  }
}

// nodes_set - set of nodes to be processed in parallel
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::EvaluateSplits(
                                               const std::vector<ExpandEntry>& nodes_set,
                                               const GHistIndexMatrix& gmat,
                                               const HistCollection<GradientSumT>& hist,
                                               const RegTree& tree) {
  builder_monitor_.Start("EvaluateSplits");
//std::cout << "EvaluateSplits 1 \n";
  const size_t n_nodes_in_set = nodes_set.size();
  const size_t nthread = std::max(1, this->nthread_);
//std::cout << "EvaluateSplits 1.1" << std::endl ;
  //best_split_tloc_.resize(224);

  using FeatureSetType = std::shared_ptr<HostDeviceVector<bst_feature_t>>;
//std::cout << "EvaluateSplits 1.2 " << std::endl ;
  std::vector<FeatureSetType> features_sets(n_nodes_in_set);
//std::cout << "EvaluateSplits 1.3 " << nthread * n_nodes_in_set << std::endl ;
  best_split_tloc_.resize(nthread * n_nodes_in_set);
//std::cout << "EvaluateSplits 2 " << std::endl ;

  // Generate feature set for each tree node
  for (size_t nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    features_sets[nid_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nid));

    for (unsigned tid = 0; tid < nthread; ++tid) {
      best_split_tloc_[nthread*nid_in_set + tid] = snode_[nid].best;
    }
  }
//std::cout << "EvaluateSplits 3 \n";

  // Create 2D space (# of nodes to process x # of features to process)
  // to process them in parallel
  const size_t grain_size = std::max<size_t>(1, features_sets[0]->Size() / nthread);
  common::BlockedSpace2d space(n_nodes_in_set, [&](size_t nid_in_set) {
      return features_sets[nid_in_set]->Size();
  }, grain_size);
//std::cout << "EvaluateSplits 4 \n";

  auto evaluator = tree_evaluator_.GetEvaluator();
  // Start parallel enumeration for all tree nodes in the set and all features
  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
    GHistRowT node_hist = hist[nid];

    for (auto idx_in_feature_set = r.begin(); idx_in_feature_set < r.end(); ++idx_in_feature_set) {
      const auto fid = features_sets[nid_in_set]->ConstHostVector()[idx_in_feature_set];
      if (interaction_constraints_.Query(nid, fid)) {
        auto grad_stats = this->EnumerateSplit<+1>(
            gmat, node_hist, snode_[nid],
            &best_split_tloc_[nthread * nid_in_set + tid], fid, nid, evaluator);
        if (SplitContainsMissingValues(grad_stats, snode_[nid])) {
          this->EnumerateSplit<-1>(
              gmat, node_hist, snode_[nid],
              &best_split_tloc_[nthread * nid_in_set + tid], fid, nid,
              evaluator);
        }
      }
    }
  });
//std::cout << "EvaluateSplits 5 \n";

  // Find Best Split across threads for each node in nodes set
  for (unsigned nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    for (unsigned tid = 0; tid < nthread; ++tid) {
      snode_[nid].best.Update(best_split_tloc_[nthread*nid_in_set + tid]);
    }
  }
//std::cout << "EvaluateSplits 6 \n";

  builder_monitor_.Stop("EvaluateSplits");
}

// split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
// Analog of std::stable_partition, but in no-inplace manner
template <bool default_left, bool any_missing, bool is_root, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionDenseKernel(const common::DenseColumn<BinIdxType>& column,
      common::Span<const size_t> rid_span, const int32_t split_cond,
      common::Span<size_t> left_part, common::Span<size_t> right_part) {
  const int32_t offset = column.GetBaseIdx();
  const BinIdxType* idx = column.GetFeatureBinIdxPtr().data();
  size_t* p_left_part = left_part.data();
  size_t* p_right_part = right_part.data();
  size_t nleft_elems = 0;
  size_t nright_elems = 0;

  const size_t* rid_span_ptr = rid_span.data();
  const size_t rid_span_size = rid_span.size();

if (is_root) {
    for (size_t i = rid_span_ptr[0]; i < rid_span_ptr[0] + rid_span_size; ++i)  {
        const size_t cond = size_t(bool((static_cast<int32_t>(idx[i]) + offset) <= split_cond));
        //if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        //  p_left_part[nleft_elems++] = rid;
        //} else {
        //  p_right_part[nright_elems++] = rid;
        //}
        p_left_part[nleft_elems] = i;
        p_right_part[nright_elems] = i;
        nleft_elems += cond;
        nright_elems += !cond;
    }
}
else {
  if (any_missing) {
    for (auto rid : rid_span) {
      if (column.IsMissing(rid)) {
        if (default_left) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      } else {
        if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      }
    }
  } else {
    if (rid_span_size >= 64) {
      for (size_t i = 0; i < rid_span_size - 64; ++i)  {
        const size_t rid = rid_span_ptr[i];
        PREFETCH_READ_T0(idx + rid_span_ptr[i + 64]);
        const size_t cond = size_t(bool((static_cast<int32_t>(idx[rid]) + offset) <= split_cond));
        //if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        //  p_left_part[nleft_elems++] = rid;
        //} else {
        //  p_right_part[nright_elems++] = rid;
        //}
        p_left_part[nleft_elems] = rid;
        p_right_part[nright_elems] = rid;
        nleft_elems += cond;
        nright_elems += !cond;
      }
      for (size_t i = rid_span_size - 64; i < rid_span_size; ++i)  {
        const size_t rid = rid_span_ptr[i];
        const size_t cond = size_t(bool((static_cast<int32_t>(idx[rid]) + offset) <= split_cond));
        //if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        //  p_left_part[nleft_elems++] = rid;
        //} else {
        //  p_right_part[nright_elems++] = rid;
        //}
        p_left_part[nleft_elems] = rid;
        p_right_part[nright_elems] = rid;
        nleft_elems += cond;
        nright_elems += !cond;
      }
    } else {
      for (size_t i = 0; i < rid_span_size; ++i)  {
        const size_t rid = rid_span_ptr[i];
        const size_t cond = size_t(bool((static_cast<int32_t>(idx[rid]) + offset) <= split_cond));
        //if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        //  p_left_part[nleft_elems++] = rid;
        //} else {
        //  p_right_part[nright_elems++] = rid;
        //}
        p_left_part[nleft_elems] = rid;
        p_right_part[nright_elems] = rid;
        nleft_elems += cond;
        nright_elems += !cond;
      }
    }
  }
}
  return {nleft_elems, nright_elems};
}

// Split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond).
// Handle sparse columns
template<bool default_left, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionSparseKernel(
  common::Span<const size_t> rid_span, const int32_t split_cond,
  const common::SparseColumn<BinIdxType>& column, common::Span<size_t> left_part,
  common::Span<size_t> right_part) {
  size_t* p_left_part  = left_part.data();
  size_t* p_right_part = right_part.data();

  size_t nleft_elems = 0;
  size_t nright_elems = 0;
  const size_t* row_data = column.GetRowData();
  const size_t column_size = column.Size();
  if (rid_span.size()) {  // ensure that rid_span is nonempty range
    // search first nonzero row with index >= rid_span.front()
    const size_t* p = std::lower_bound(row_data, row_data + column_size,
                                       rid_span.front());

    if (p != row_data + column_size && *p <= rid_span.back()) {
      size_t cursor = p - row_data;

      for (auto rid : rid_span) {
        while (cursor < column_size
               && column.GetRowIdx(cursor) < rid
               && column.GetRowIdx(cursor) <= rid_span.back()) {
          ++cursor;
        }
        if (cursor < column_size && column.GetRowIdx(cursor) == rid) {
          if (static_cast<int32_t>(column.GetGlobalBinIdx(cursor)) <= split_cond) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
          ++cursor;
        } else {
          // missing value
          if (default_left) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
        }
      }
    } else {  // all rows in rid_span have missing values
      if (default_left) {
        std::copy(rid_span.begin(), rid_span.end(), p_left_part);
        nleft_elems = rid_span.size();
      } else {
        std::copy(rid_span.begin(), rid_span.end(), p_right_part);
        nright_elems = rid_span.size();
      }
    }
  }

  return {nleft_elems, nright_elems};
}

template <typename GradientSumT>
template <typename BinIdxType, bool is_root>
void QuantileHistMaker::Builder<GradientSumT>::PartitionKernel(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree) {
  const size_t* rid = row_set_collection_[nid].begin;

  common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
  common::Span<size_t> left  = partition_builder_.GetLeftBuffer(node_in_set,
                                                                range.begin(), range.end());
  common::Span<size_t> right = partition_builder_.GetRightBuffer(node_in_set,
                                                                 range.begin(), range.end());
  const bst_uint fid = tree[nid].SplitIndex();
  const bool default_left = tree[nid].DefaultLeft();
  const auto column_ptr = column_matrix.GetColumn<BinIdxType>(fid);

  std::pair<size_t, size_t> child_nodes_sizes;

  if (column_ptr->GetType() == xgboost::common::kDenseColumn) {
    const common::DenseColumn<BinIdxType>& column =
          static_cast<const common::DenseColumn<BinIdxType>& >(*(column_ptr.get()));
    if (default_left) {
      if (column_matrix.AnyMissing()) {
        child_nodes_sizes = PartitionDenseKernel<true, true, is_root>(column, rid_span, split_cond,
                                                             left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<true, false, is_root>(column, rid_span, split_cond,
                                                              left, right);
      }
    } else {
      if (column_matrix.AnyMissing()) {
        child_nodes_sizes = PartitionDenseKernel<false, true, is_root>(column, rid_span, split_cond,
                                                              left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<false, false, is_root>(column, rid_span, split_cond,
                                                               left, right);
      }
    }
  } else {
    const common::SparseColumn<BinIdxType>& column
      = static_cast<const common::SparseColumn<BinIdxType>& >(*(column_ptr.get()));
    if (default_left) {
      child_nodes_sizes = PartitionSparseKernel<true>(rid_span, split_cond, column, left, right);
    } else {
      child_nodes_sizes = PartitionSparseKernel<false>(rid_span, split_cond, column, left, right);
    }
  }

  const size_t n_left  = child_nodes_sizes.first;
  const size_t n_right = child_nodes_sizes.second;

  partition_builder_.SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
  partition_builder_.SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::FindSplitConditions(
                                                     const std::vector<ExpandEntry>& nodes,
                                                     const RegTree& tree,
                                                     const GHistIndexMatrix& gmat,
                                                     std::vector<int32_t>* split_conditions, std::vector<uint16_t>* tmp) {
  const size_t n_nodes = nodes.size();
  //std::cout<< "n_nodes: " << n_nodes << "\n";
  (*split_conditions)[0] = n_nodes;
  //std::cout<< "(*split_conditions)[0]: " << (*split_conditions)[0] << "\n";
  //split_conditions->resize(n_nodes);

  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const bst_uint fid = tree[nid].SplitIndex();
    const bst_float split_pt = tree[nid].SplitCond();
    const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
    const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
    int32_t split_cond = -1;
    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    CHECK_LT(upper_bound,
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
      if (split_pt == gmat.cut.Values()[bound]) {
        split_cond = static_cast<int32_t>(bound);
      }
    }
    (*split_conditions)[(*tmp)[i] + 1] = split_cond;
  }
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToRowSet(
                                               const std::vector<ExpandEntry>& nodes,
                                               RegTree* p_tree) {
  const size_t n_nodes = nodes.size();
  for (unsigned int i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);
    CHECK_EQ((*p_tree)[nid].LeftChild() + 1, (*p_tree)[nid].RightChild());
    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(),
        (*p_tree)[nid].RightChild(), n_left, n_right);
  }
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ApplySplit(const std::vector<ExpandEntry> nodes,
                                            const GHistIndexMatrix& gmat,
                                            const ColumnMatrix& column_matrix,
                                            const HistCollection<GradientSumT>& hist,
                                            RegTree* p_tree, int depth, std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind,
                                            std::vector<uint16_t>* compleate_splits ) {
  std::string timer_name = "Partition:";
  timer_name += std::to_string(depth);
  builder_monitor_.Start("ApplySplit");
  // 1. Find split condition for each split
  const size_t n_nodes = nodes.size();
  //std::cout << "\nn_nodes: " << n_nodes << "\n";
  //std::vector<int32_t> split_conditions;
  FindSplitConditions(nodes, *p_tree, gmat, split_conditions, compleate_splits);
  //std::cout << "\n FindSplitConditions finished \n";
(*split_ind)[0] = n_nodes;
for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const bst_uint fid = (*p_tree)[nid].SplitIndex();
    (*split_ind)[(*compleate_splits)[i] + 1] = fid;
}
  //std::cout << "\n ApplySplit finished \n";
//   // 2.1 Create a blocked space of size SUM(samples in each node)
//   common::BlockedSpace2d space(n_nodes, [&](size_t node_in_set) {
//     int32_t nid = nodes[node_in_set].nid;
//     return row_set_collection_[nid].Size();
//   }, kPartitionBlockSize);
//   // 2.2 Initialize the partition builder
//   // allocate buffers for storage intermediate results by each thread
//   partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
//     const int32_t nid = nodes[node_in_set].nid;
//     const size_t size = row_set_collection_[nid].Size();
//     const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
//     return n_tasks;
//   });
//   // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
//   // Store results in intermediate buffers from partition_builder_
//   builder_monitor_.Start(timer_name);
//   common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
//     size_t begin = r.begin();
//     const int32_t nid = nodes[node_in_set].nid;
//     const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
//     partition_builder_.AllocateForTask(task_id);
// if (depth == 0) {
//         switch (column_matrix.GetTypeSize()) {
//       case common::kUint8BinsTypeSize:
//         PartitionKernel<uint8_t, true>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       case common::kUint16BinsTypeSize:
//         PartitionKernel<uint16_t, true>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       case common::kUint32BinsTypeSize:
//         PartitionKernel<uint32_t, true>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       default:
//         CHECK(false);  // no default behavior
//     }
// } else {
//       switch (column_matrix.GetTypeSize()) {
//       case common::kUint8BinsTypeSize:
//         PartitionKernel<uint8_t, false>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       case common::kUint16BinsTypeSize:
//         PartitionKernel<uint16_t, false>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       case common::kUint32BinsTypeSize:
//         PartitionKernel<uint32_t, false>(node_in_set, nid, r,
//                   split_conditions[node_in_set], column_matrix, *p_tree);
//         break;
//       default:
//         CHECK(false);  // no default behavior
//     }
// }
//     });

//   // 3. Compute offsets to copy blocks of row-indexes
//   // from partition_builder_ to row_set_collection_
//   partition_builder_.CalculateRowOffsets();

//   // 4. Copy elements from partition_builder_ to row_set_collection_ back
//   // with updated row-indexes for each tree-node
//   common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
//     const int32_t nid = nodes[node_in_set].nid;
//     partition_builder_.MergeToArray(node_in_set, r.begin(),
//         const_cast<size_t*>(row_set_collection_[nid].begin));
//   });
//     builder_monitor_.Stop(timer_name);

//   // 5. Add info about splits into row_set_collection_
//   AddSplitsToRowSet(nodes, p_tree);
  builder_monitor_.Stop("ApplySplit");
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitNewNode(int nid,
                                             const GHistIndexMatrix& gmat,
                                             const std::vector<GradientPair>& gpair,
                                             const DMatrix& fmat,
                                             const RegTree& tree, int i, uint64_t* mask) {
  builder_monitor_.Start("InitNewNode");
  {
    snode_.resize(tree.param.num_nodes, NodeEntry(param_));
  }

  {
    GHistRowT hist = hist_[nid];
    GradientPairT grad_stat;
    if (tree[nid].IsRoot()) {
      if (data_layout_ == DataLayout::kDenseDataZeroBased
          || data_layout_ == DataLayout::kDenseDataOneBased) {
        const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
        const uint32_t ibegin = row_ptr[fid_least_bins_];
        const uint32_t iend = row_ptr[fid_least_bins_ + 1];
        auto begin = hist.data();
        for (uint32_t i = ibegin; i < iend; ++i) {
          const GradientPairT et = begin[i];
          grad_stat.Add(et.GetGrad(), et.GetHess());
        }
      } else {
        const RowSetCollection::Elem e = row_set_collection_[nid];
        for (const size_t* it = e.begin; it < e.end; ++it) {
          grad_stat.Add(gpair[*it].GetGrad(), gpair[*it].GetHess());
        }
      }
      histred_.Allreduce(&grad_stat, 1);
      snode_[nid].stats = tree::GradStats(grad_stat.GetGrad(), grad_stat.GetHess());
    } else {
      int parent_id = tree[nid].Parent();
      if (snode_[parent_id].best.left_sum.GetHess() < snode_[parent_id].best.right_sum.GetHess() && tree[nid].IsLeftChild()) {
        //std::cout << "l:" << i << " ";
        *(mask + i/64) |= ((uint64_t)(1) << (i%64));
      }
      if (snode_[parent_id].best.right_sum.GetHess() < snode_[parent_id].best.left_sum.GetHess() && !(tree[nid].IsLeftChild())) {
        //std::cout << "r:" << i << " ";
        *(mask + i/64) |= ((uint64_t)(1) << (i%64));
      }
      if (tree[nid].IsLeftChild()) {
        snode_[nid].stats = snode_[parent_id].best.left_sum;
      } else {
        snode_[nid].stats = snode_[parent_id].best.right_sum;
      }
    }
  }

  // calculating the weights
  {
    auto evaluator = tree_evaluator_.GetEvaluator();
    bst_uint parentid = tree[nid].Parent();
    snode_[nid].weight = static_cast<float>(
        evaluator.CalcWeight(parentid, param_, GradStats{snode_[nid].stats}));
    snode_[nid].root_gain = static_cast<float>(
        evaluator.CalcGain(parentid, param_, GradStats{snode_[nid].stats}));
  }
  builder_monitor_.Stop("InitNewNode");
}

// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template <typename GradientSumT>
template <int d_step>
GradStats QuantileHistMaker::Builder<GradientSumT>::EnumerateSplit(
    const GHistIndexMatrix &gmat, const GHistRowT &hist, const NodeEntry &snode,
    SplitEntry *p_best, bst_uint fid, bst_uint nodeID,
    TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
  CHECK(d_step == +1 || d_step == -1);

  // aliases
  const std::vector<uint32_t>& cut_ptr = gmat.cut.Ptrs();
  const std::vector<bst_float>& cut_val = gmat.cut.Values();

  // statistics on both sides of split
  GradStats c;
  GradStats e;
  // best split so far
  SplitEntry best;

  // bin boundaries
  CHECK_LE(cut_ptr[fid],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(cut_ptr[fid + 1],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  // imin: index (offset) of the minimum value for feature fid
  //       need this for backward enumeration
  const auto imin = static_cast<int32_t>(cut_ptr[fid]);
  // ibegin, iend: smallest/largest cut points for feature fid
  // use int to allow for value -1
  int32_t ibegin, iend;
  if (d_step > 0) {
    ibegin = static_cast<int32_t>(cut_ptr[fid]);
    iend = static_cast<int32_t>(cut_ptr[fid + 1]);
  } else {
    ibegin = static_cast<int32_t>(cut_ptr[fid + 1]) - 1;
    iend = static_cast<int32_t>(cut_ptr[fid]) - 1;
  }

  for (int32_t i = ibegin; i != iend; i += d_step) {
    // start working
    // try to find a split
    e.Add(hist[i].GetGrad(), hist[i].GetHess());
    if (e.GetHess() >= param_.min_child_weight) {
      c.SetSubstract(snode.stats, e);
      if (c.GetHess() >= param_.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param_, nodeID, fid, GradStats{e},
                                      GradStats{c}) -
              snode.root_gain);
          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        } else {
          // backward enumeration: split at left bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param_, nodeID, fid, GradStats{c},
                                      GradStats{e}) -
              snode.root_gain);
          if (i == imin) {
            // for leftmost bin, left bound is the smallest feature value
            split_pt = gmat.cut.MinValues()[fid];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
        }
      }
    }
  }
  p_best->Update(best);

  return e;
}

template struct QuantileHistMaker::Builder<float>;
template struct QuantileHistMaker::Builder<double>;
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint8_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint16_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint32_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint8_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint16_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint32_t, true>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);

template void QuantileHistMaker::Builder<float>::PartitionKernel<uint8_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint16_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint32_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint8_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint16_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint32_t, false>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("(Deprecated, use grow_quantile_histmaker instead.)"
          " Grow tree using quantized histogram.")
.set_body(
    []() {
      LOG(WARNING) << "grow_fast_histmaker is deprecated, "
                   << "use grow_quantile_histmaker instead.";
      return new QuantileHistMaker();
    });

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    []() {
      return new QuantileHistMaker();
    });
}  // namespace tree
}  // namespace xgboost
