/*!
 * Copyright 2017-2021 by Contributors
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
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

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
#include <sys/time.h>
#include <time.h>
#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

uint64_t get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000 + t.tv_nsec;
}


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
  const bool is_optimized_branch = (dmat->IsDense() && param_.enable_feature_grouping <= 0 && param_.grow_policy == TrainParam::kDepthWise);
  builder->reset(new Builder<GradientSumT>(
                param_,
                std::move(pruner_),
                int_constraint_, dmat, is_optimized_branch));
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
    builder->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, tree);
  }
}
void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  uint64_t t1 = 0;
  t1 = get_time();
  std::vector<GradientPair>& gpair_h = gpair->HostVector();
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    gmat_.Init(dmat, static_cast<uint32_t>(param_.max_bin));
    column_matrix_.Init(gmat_, param_.sparse_threshold);
    if (param_.enable_feature_grouping > 0) {
      gmatb_.Init(gmat_, column_matrix_, param_);
    }
    updater_monitor_.Stop("GmatInitialization");
    time_GmatInitialization += get_time() - t1;
    // A proper solution is puting cut matrix in DMatrix, see:
    // https://github.com/dmlc/xgboost/issues/5143
    is_gmat_initialized_ = true;
    const size_t n_threads = omp_get_max_threads();
    //    std::cout << "\nn_threads: " << n_threads << "\n";
    const size_t n_elements = gmat_.index.Size();
    const uint8_t* data = gmat_.index.data<uint8_t>();
    const size_t n_bins = gmat_.cut.Ptrs().back();
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
  time_FullUpdate += get_time() - t1;
  N_CALL_++;
  if(N_CALL_ % 100 == 0) {
    std::cout << "[TIMER]:FullUpdate time,s: " <<  (double)(time_FullUpdate)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    GmatInitialization time,s: " <<  (double)(time_GmatInitialization)/(double)(1000000000) << std::endl;
  }
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data, HostDeviceVector<bst_float>* out_preds) {
    if (hist_maker_param_.single_precision_histogram && float_builder_) {
      if (data->IsDense() &&  param_.enable_feature_grouping <= 0 && param_.grow_policy == TrainParam::kDepthWise) {
        return float_builder_->UpdatePredictionCacheDense(data, out_preds, 0, 1, &gmat_);
      } else{
        return float_builder_->UpdatePredictionCache(data, out_preds);
      }
    } else if (double_builder_) {
      if (data->IsDense() &&  param_.enable_feature_grouping <= 0 && param_.grow_policy == TrainParam::kDepthWise) {
        return double_builder_->UpdatePredictionCacheDense(data, out_preds, 0, 1, &gmat_);
      } else{
        return double_builder_->UpdatePredictionCache(data, out_preds);
      }
    } else {
       return false;
    }
  }

bool QuantileHistMaker::UpdatePredictionCacheMulticlass(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds, const int gid, const int ngroup) {
    if (hist_maker_param_.single_precision_histogram && float_builder_) {
      if (data->IsDense() && param_.enable_feature_grouping <= 0 && param_.grow_policy == TrainParam::kDepthWise) {
        return float_builder_->UpdatePredictionCacheDense(data, out_preds, gid, ngroup, &gmat_);
      } else{
        return float_builder_->UpdatePredictionCache(data, out_preds, gid, ngroup);
      }
    } else if (double_builder_) {
      if (data->IsDense() && param_.enable_feature_grouping <= 0 && param_.grow_policy == TrainParam::kDepthWise) {
        return double_builder_->UpdatePredictionCacheDense(data, out_preds, gid, ngroup, &gmat_);
      } else{
        return double_builder_->UpdatePredictionCache(data, out_preds, gid, ngroup);
      }
    } else {
       return false;
    }
}


template <typename GradientSumT>
void BatchHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT *builder,
                                                         int,
                                                         int,
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

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
    }
  });
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
  for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
    int nid = entry.nid;
    builder->hist_.AddHistRow(nid);
    (*starting_index) = std::min(nid, (*starting_index));
  }
  (*sync_count) = builder->nodes_for_explicit_hist_build_.size();

  for (auto const& node : builder->nodes_for_subtraction_trick_) {
    builder->hist_.AddHistRow(node.nid);
  }
  builder->builder_monitor_.Start("AddHistRows");

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
uint64_t get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000 + t.tv_nsec;
}

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

template<typename BinIdxType, bool no_sampling, bool read_by_column>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<double> hist, const BinIdxType* numa, uint16_t* nodes_ids, uint64_t* offsets64, size_t* rows_ptr, const ColumnMatrix *column_matrix) {
if(read_by_column) {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint32_t* offsets = gmat.index.Offset();
  double* hist_data = reinterpret_cast<double*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
      const BinIdxType* gr_index_local = (*column_matrix).GetColumn<BinIdxType>(0)->
                                           GetFeatureBinIdxPtr().data();
      double* hist_data_local = hist_data + two*(offsets[0]);
      for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
        const size_t row_id = no_sampling ? ii : rows_ptr[ii];
        nodes_ids[row_id] = 0;
        const size_t idx_gh = row_id << 1;
        const uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[row_id]) << 1;
        hist_data_local[idx_bin]   += pgh[idx_gh];
        hist_data_local[idx_bin+1] += pgh[idx_gh+1];
      }
    for (size_t cid = 1; cid < n_features; ++cid) {
      gr_index_local = (*column_matrix).GetColumn<BinIdxType>(cid)->
                                           GetFeatureBinIdxPtr().data();
      hist_data_local = hist_data + two*(offsets[cid]);
      for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
        const size_t row_id = no_sampling ? ii : rows_ptr[ii];
        const size_t idx_gh = row_id << 1;
        const uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[row_id]) << 1;
        hist_data_local[idx_bin]   += pgh[idx_gh];
        hist_data_local[idx_bin+1] += pgh[idx_gh+1];
      }
    }
} else {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  double* hist_data = reinterpret_cast<double*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

  for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
    const size_t i = no_sampling ? ii : rows_ptr[ii];
    nodes_ids[i] = 0;
    const size_t icol_start = i * n_features;
    const size_t idx_gh = two *i;
    const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t ib = 0; ib < nb; ++ib) {
      VECTOR_UNR(0, ib);
      VECTOR_UNR(1, ib);
      VECTOR_UNR(2, ib);
      VECTOR_UNR(3, ib);
      VECTOR_UNR(4, ib);
      VECTOR_UNR(5, ib);
      VECTOR_UNR(6, ib);
      VECTOR_UNR(7, ib);
      VECTOR_UNR(8, ib);
      VECTOR_UNR(9, ib);
      VECTOR_UNR(10, ib);
      VECTOR_UNR(11, ib);
      VECTOR_UNR(12, ib);
    }
    for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
        VECTOR_UNR(jb,0);
    }
  }
}
}


template<typename BinIdxType, bool no_sampling, bool read_by_column>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<float> hist, const BinIdxType* numa, uint16_t* nodes_ids, uint64_t* offsets64, size_t* rows_ptr, const ColumnMatrix *column_matrix) {
if(read_by_column) {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint32_t* offsets = gmat.index.Offset();
  float* hist_data = reinterpret_cast<float*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
      const BinIdxType* gr_index_local = (*column_matrix).GetColumn<BinIdxType>(0)->
                                           GetFeatureBinIdxPtr().data();
      float* hist_data_local = hist_data + two*(offsets[0]);
      for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
        const size_t row_id = no_sampling ? ii : rows_ptr[ii];
        nodes_ids[row_id] = 0;
        const size_t idx_gh = row_id << 1;
        const uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[row_id]) << 1;
        hist_data_local[idx_bin]   += pgh[idx_gh];
        hist_data_local[idx_bin+1] += pgh[idx_gh+1];
      }
  for (size_t cid = 1; cid < n_features; ++cid) {
      gr_index_local = (*column_matrix).GetColumn<BinIdxType>(cid)->
                                           GetFeatureBinIdxPtr().data();
      hist_data_local = hist_data + two*(offsets[cid]);
      for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
        const size_t row_id = no_sampling ? ii : rows_ptr[ii];
        const size_t idx_gh = row_id << 1;
        const uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[row_id]) << 1;
        hist_data_local[idx_bin]   += pgh[idx_gh];
        hist_data_local[idx_bin+1] += pgh[idx_gh+1];
      }
    }
} else {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  float* hist_data = reinterpret_cast<float*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

  for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
    const size_t i = no_sampling ? ii : rows_ptr[ii];
    nodes_ids[i] = 0;
    const size_t icol_start = i * n_features;
    const size_t idx_gh = two *i;
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t ib = 0; ib < nb; ++ib) {
      UNR(0, ib);
      UNR(1, ib);
      UNR(2, ib);
      UNR(3, ib);
      UNR(4, ib);
      UNR(5, ib);
      UNR(6, ib);
      UNR(7, ib);
      UNR(8, ib);
      UNR(9, ib);
      UNR(10, ib);
      UNR(11, ib);
      UNR(12, ib);
    }
    for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
      UNR(jb,0);
    }
  }
}
}

template<typename BinIdxType>
void JustPartition(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, uint64_t* mask, uint32_t* nodes_count) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  for (size_t i = row_indices_begin; i < row_indices_end; ++i) {
    const uint32_t nid = nodes_ids[i];
    const size_t icol_start = i * n_features;
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
    nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
    if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask + nodes_ids[i]/64)) {
      rows[++count] = i;
      ++nodes_count[nodes_ids[i]];
    }
  }
}


template<typename BinIdxType>
void JustPartitionWithLeafsMask(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, uint64_t* mask,
                          uint64_t* leafs_mask, std::vector<int>* prev_level_nodes, uint32_t* nodes_count) {
  // const size_t size = row_indices.Size();
  // const size_t* rid = row_indices.begin;
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();

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
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
    nodes_ids[i] = 2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc);
    if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask+nodes_ids[i]/64)) {
      rows[++count] = i;
      ++nodes_count[nodes_ids[i]];
    }
  }
}


template<typename BinIdxType>
void JustPartitionLastLayer(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind,
                          std::vector<int>* curr_level_nodes, uint64_t* leafs_mask, std::vector<int>* prev_level_nodes) {
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
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
      const BinIdxType* gr_index_local = gradient_index + icol_start;
      const int32_t sc = (*split_conditions)[nid + 1];
      const bst_uint si = (*split_ind)[nid + 1];
      nodes_ids[i] = (uint16_t)(1) << 15;
      nodes_ids[i] |= (uint16_t)((*curr_level_nodes)[2*nid + !(((int32_t)(gr_index_local[si]) + (int32_t)(offsets[si])) <= sc)]);
  }
}

template<typename BinIdxType, bool no_sampling>
void JustPartitionWithLeafsMaskColumn(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind, uint64_t* mask,
                          uint64_t* leafs_mask, std::vector<int>* prev_level_nodes, uint32_t* nodes_count, const ColumnMatrix *column_matrix, const size_t* row_indices_ptr) {
  const uint32_t rows_offset = gmat.row_ptr.size() - 1;
  const BinIdxType* columnar_data = reinterpret_cast<const BinIdxType*>(column_matrix->GetIndexData());
  const uint32_t* offsets = gmat.index.Offset();

  for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
    const uint32_t i = no_sampling ? ii : row_indices_ptr[ii];
    const uint32_t nid = nodes_ids[i];
    if(((uint16_t)(1) << 15 & nid)) {
      continue;
    }
    if((((uint64_t)(1) << (nid%64)) & *(leafs_mask + nid/64))) {
      nodes_ids[i] = (uint16_t)(1) << 15;
      nodes_ids[i] |= (uint16_t)((*prev_level_nodes)[nid]);
      continue;
    }
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
    const int32_t cmp_value = ((int32_t)(columnar_data[si*rows_offset + i]) + (int32_t)(offsets[si]));

    nodes_ids[i] = 2*nid + !(cmp_value <= sc);
    if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask+nodes_ids[i]/64)) {
      rows[++count] = i;
      ++nodes_count[nodes_ids[i]];
    }
  }
}


template<typename BinIdxType, bool no_sampling>
void JustPartitionLastLayerColumn(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind,
                          std::vector<int>* curr_level_nodes, uint64_t* leafs_mask, std::vector<int>* prev_level_nodes,
                          const ColumnMatrix *column_matrix, const size_t* row_indices_ptr) {
  const uint32_t rows_offset = gmat.row_ptr.size() - 1;
  const BinIdxType* columnar_data = reinterpret_cast<const BinIdxType*>(column_matrix->GetIndexData());
  const uint32_t* offsets = gmat.index.Offset();
  for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
    const uint32_t i = no_sampling ? ii : row_indices_ptr[ii];
    const uint32_t nid = nodes_ids[i];
    if(((uint16_t)(1) << 15 & nid)) {
      continue;
    }
    if((((uint64_t)(1) << (nid%64)) & *(leafs_mask + (nid/64)))) {
      nodes_ids[i] = (uint16_t)(1) << 15;
      nodes_ids[i] |= (uint16_t)((*prev_level_nodes)[nid]);
      continue;
    }
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
    nodes_ids[i] = (uint16_t)(1) << 15;
    const int32_t cmp_value = ((int32_t)(columnar_data[si*rows_offset + i]) + (int32_t)(offsets[si]));
    nodes_ids[i] |= (uint16_t)((*curr_level_nodes)[2*nid + !(cmp_value <= sc)]);
  }
}

// sloow
template<typename BinIdxType, bool no_sampling>
void JustPartitionColumnar(const size_t row_indices_begin,
                          const size_t row_indices_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          uint32_t* hist, uint32_t* rows, uint32_t& count, const BinIdxType* numa, uint16_t* nodes_ids,
                          std::vector<int32_t>* split_conditions, std::vector<bst_uint>* split_ind,
                          uint64_t* mask, uint32_t* nodes_count,
                          const ColumnMatrix *column_matrix, const size_t* row_indices_ptr) {
  const uint32_t* offsets = gmat.index.Offset();
  const uint32_t rows_offset = gmat.row_ptr.size() - 1;
  const BinIdxType* columnar_data = reinterpret_cast<const BinIdxType*>(column_matrix->GetIndexData());

  for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
    const uint32_t i = no_sampling ? ii : row_indices_ptr[ii];
    const uint32_t nid = nodes_ids[i];
    const int32_t sc = (*split_conditions)[nid + 1];
    const bst_uint si = (*split_ind)[nid + 1];
    const int32_t cmp_value = ((int32_t)(columnar_data[si*rows_offset + i]) + (int32_t)(offsets[si]));
    nodes_ids[i] = 2*nid + !(cmp_value <= sc);
    if (((uint64_t)(1) << (nodes_ids[i]%64)) & *(mask + nodes_ids[i]/64)) {
      rows[++count] = i;
      ++nodes_count[nodes_ids[i]];
    }
  }

}


template<bool do_prefetch, typename BinIdxType, int depth0, bool read_by_column, bool feature_blocking>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const uint32_t* rows,
                          const uint32_t row_size,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<double> hist, const BinIdxType* numa, uint16_t* nodes_ids, const uint32_t n_nodes, uint64_t* offsets640, const ColumnMatrix *column_matrix, const size_t n_features_in_block) {
if (read_by_column) {
  const uint32_t n_bins = gmat.cut.Ptrs().back();
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint32_t* offsets = gmat.index.Offset();
  double* hist_data = reinterpret_cast<double*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains

  for (size_t cid = 0; cid < n_features; ++cid) {
      const BinIdxType* gr_index_local = (*column_matrix).GetColumn<BinIdxType>(cid)->
                                           GetFeatureBinIdxPtr().data();
      double* hist_data_local = hist_data + two*(offsets[cid]);
      for (size_t ii = 0; ii < row_size; ++ii) {
        const size_t row_id = rows[ii];
        const uint32_t nid = nodes_ids[row_id];
        const size_t idx_gh = row_id << 1;
        const uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[row_id]) << 1;
        hist_data_local[idx_bin + nid*2*n_bins]   += pgh[idx_gh];
        hist_data_local[idx_bin+1  + nid*2*n_bins] += pgh[idx_gh+1];
      }
    }
} else if (feature_blocking) {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  const uint32_t n_bins = gmat.cut.Ptrs().back();
  double* hist_data0 = reinterpret_cast<double*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

  for (size_t ib = 0; ib < nb; ++ib) {
  for (size_t ri = 0; ri < row_size; ++ri) {
      const size_t i = rows[ri];
      const size_t icol_start = i * n_features;
      const BinIdxType* gr_index_local = gradient_index + icol_start;
      const size_t idx_gh = two * i;
      const uint32_t nid = nodes_ids[i];

      const uint64_t* offsets64 = offsets640 + nid*n_features;
      const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
      asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

      double* hist_data = hist_data0 + nid*n_bins*2;
        VECTOR_UNR(0, ib);
        VECTOR_UNR(1, ib);
        VECTOR_UNR(2, ib);
        VECTOR_UNR(3, ib);
        VECTOR_UNR(4, ib);
        VECTOR_UNR(5, ib);
        VECTOR_UNR(6, ib);
        VECTOR_UNR(7, ib);
        VECTOR_UNR(8, ib);
        VECTOR_UNR(9, ib);
        VECTOR_UNR(10, ib);
        VECTOR_UNR(11, ib);
        VECTOR_UNR(12, ib);
    }
}

  for (size_t ri = 0; ri < row_size; ++ri) {
      const size_t i = rows[ri];
      const size_t icol_start = i * n_features;
      const BinIdxType* gr_index_local = gradient_index + icol_start;
      const size_t idx_gh = two * i;
      const uint32_t nid = nodes_ids[i];
      const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

      const uint64_t* offsets64 = offsets640 + nid*n_features;
      const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
      asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

      double* hist_data = hist_data0 + nid*n_bins*2;
      for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
          VECTOR_UNR(jb,0);
      }
    }

  } else {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  const uint32_t n_bins = gmat.cut.Ptrs().back();
  double* hist_data0 = reinterpret_cast<double*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

  const size_t size_with_prefetch = row_size > Prefetch1::kPrefetchOffset ? row_size - Prefetch1::kPrefetchOffset : 0;
    for (size_t ri = 0; ri < size_with_prefetch; ++ri) {
      const size_t i = rows[ri];
      const size_t icol_start = i * n_features;
      const BinIdxType* gr_index_local = gradient_index + icol_start;
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
      double* hist_data = hist_data0 + nid*n_bins*2;
      for (size_t ib = 0; ib < nb; ++ib) {
        VECTOR_UNR(0, ib);
        VECTOR_UNR(1, ib);
        VECTOR_UNR(2, ib);
        VECTOR_UNR(3, ib);
        VECTOR_UNR(4, ib);
        VECTOR_UNR(5, ib);
        VECTOR_UNR(6, ib);
        VECTOR_UNR(7, ib);
        VECTOR_UNR(8, ib);
        VECTOR_UNR(9, ib);
        VECTOR_UNR(10, ib);
        VECTOR_UNR(11, ib);
        VECTOR_UNR(12, ib);
      }
      for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
          VECTOR_UNR(jb,0);
      }
    }

  for (size_t ri = size_with_prefetch; ri < row_size; ++ri) {
      const size_t i = rows[ri];
      const size_t icol_start = i * n_features;
      const BinIdxType* gr_index_local = gradient_index + icol_start;
      const size_t idx_gh = two * i;
      const uint32_t nid = nodes_ids[i];
      const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

      const uint64_t* offsets64 = offsets640 + nid*n_features;
      const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1]};
      asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : );

      double* hist_data = hist_data0 + nid*n_bins*2;
      for (size_t ib = 0; ib < nb; ++ib) {
        VECTOR_UNR(0, ib);
        VECTOR_UNR(1, ib);
        VECTOR_UNR(2, ib);
        VECTOR_UNR(3, ib);
        VECTOR_UNR(4, ib);
        VECTOR_UNR(5, ib);
        VECTOR_UNR(6, ib);
        VECTOR_UNR(7, ib);
        VECTOR_UNR(8, ib);
        VECTOR_UNR(9, ib);
        VECTOR_UNR(10, ib);
        VECTOR_UNR(11, ib);
        VECTOR_UNR(12, ib);
      }
      for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
          VECTOR_UNR(jb,0);
      }
    }
  }
}



template<bool do_prefetch, typename BinIdxType, int depth0, bool read_by_column, bool feature_blocking>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const uint32_t* rows,
                          const uint32_t row_size,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<float> hist, const BinIdxType* numa, uint16_t* nodes_ids, const uint32_t n_nodes, uint64_t* offsets640, const ColumnMatrix *column_matrix, const size_t n_features_in_block) {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa;//gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  const uint32_t n_bins = gmat.cut.Ptrs().back();
  float* hist_data0 = reinterpret_cast<float*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nb = n_features / 13;
  const size_t tail_size = n_features - nb*13;

  const size_t size_with_prefetch = row_size > Prefetch1::kPrefetchOffset ? row_size - Prefetch1::kPrefetchOffset : 0;

  // TODO need template do_prefetch
  for (size_t ri = 0; ri < size_with_prefetch; ++ri) {
    const size_t i = rows[ri];
    const size_t icol_start = i * n_features;
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;
    const uint32_t nid = nodes_ids[i];
    const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

    PREFETCH_READ_T0(pgh + two * rows[ri + Prefetch1::kPrefetchOffset]);
    PREFETCH_READ_T0(nodes_ids + rows[ri + Prefetch1::kPrefetchOffset]);
    for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
        j += Prefetch1::GetPrefetchStep<BinIdxType>()) {
      PREFETCH_READ_T0(gradient_index + j);
    }

    float* hist_data = hist_data0 + nid*n_bins*2;
    for (size_t ib = 0; ib < nb; ++ib) {
      UNR(0, ib);
      UNR(1, ib);
      UNR(2, ib);
      UNR(3, ib);
      UNR(4, ib);
      UNR(5, ib);
      UNR(6, ib);
      UNR(7, ib);
      UNR(8, ib);
      UNR(9, ib);
      UNR(10, ib);
      UNR(11, ib);
      UNR(12, ib);
    }
    for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
        UNR(jb,0);
    }
  }

  for (size_t ri = size_with_prefetch; ri < row_size; ++ri) {
      const size_t i = rows[ri];
      const size_t icol_start = i * n_features;
      const BinIdxType* gr_index_local = gradient_index + icol_start;
      const size_t idx_gh = two * i;
      const uint32_t nid = nodes_ids[i];
      const size_t icol_start_prefetch = rows[ri + Prefetch1::kPrefetchOffset] * n_features;

      float* hist_data = hist_data0 + nid*n_bins*2;
      for (size_t ib = 0; ib < nb; ++ib) {
        UNR(0, ib);
        UNR(1, ib);
        UNR(2, ib);
        UNR(3, ib);
        UNR(4, ib);
        UNR(5, ib);
        UNR(6, ib);
        UNR(7, ib);
        UNR(8, ib);
        UNR(9, ib);
        UNR(10, ib);
        UNR(11, ib);
        UNR(12, ib);
      }
      for(size_t jb = n_features - tail_size;  jb < n_features; ++jb) {
          UNR(jb,0);
      }
    }
}


template<typename GradientSumT>
template<typename BinIdxType>
void QuantileHistMaker::Builder<GradientSumT>::DensePartition(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, int depth,
    std::vector<std::vector<GradientSumT>>* histograms, uint16_t* nodes_ids, std::vector<int32_t>* split_conditions,
    std::vector<bst_uint>* split_ind, const ColumnMatrix *column_matrix, uint64_t* mask, uint64_t* leaf_mask, int max_depth, common::BlockedSpace2d* space_ptr) {
  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  const size_t n_bins = gmat.cut.Ptrs().back();
  std::vector<size_t>& row_indices = *row_set_collection_.Data();
  const size_t* row_indices_ptr = row_indices.data();
  common::BlockedSpace2d& space = *space_ptr;
  int nthreads = this->nthread_;
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);
  std::string depth_str = std::to_string(depth);
builder_monitor_.Start("JustPartition!!!!!!" + depth_str);
            vec_rows_.resize(nthreads);
            if (depth == 0) {
              is_compleate_tree_ = true;
            }

            is_compleate_tree_ = is_compleate_tree_ * (1 << depth == qexpand_depth_wise_.size());

            threads_addr_.resize(nthreads);

            std::vector<int> curr_level_nodes(1 << depth, 0);
            std::vector<std::vector<uint32_t>> threads_nodes_count(nthreads);

            for(size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
              curr_level_nodes[compleate_trees_depth_wise_[i]] = qexpand_depth_wise_[i].nid;
            }
            // std::cout << "curr_" << depth <<" level_nodes: ";
            // for(size_t i = 0; i < curr_level_nodes.size(); ++i) {
            //   std::cout << curr_level_nodes[i] << "  ";
            // }
            // std::cout << std::endl;

            if(depth > 0) {
            if(depth < max_depth) {

              if (is_compleate_tree_) {
                #pragma omp parallel num_threads(nthreads)
                  {
                      size_t tid = omp_get_thread_num();
                      threads_nodes_count[tid].resize(1 << depth, 0);
                      const BinIdxType* numa = tid < nthreads/2 ? gmat.index.data<BinIdxType>() :  gmat.index.data2<BinIdxType>();
                      size_t chunck_size =
                          num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

                      size_t begin = chunck_size * tid;
                      size_t end = std::min(begin + chunck_size, num_blocks_in_space);
                      uint64_t local_time_alloc = 0;
                    const size_t th_size = end > begin ? end - begin : 0;
                      vec_rows_[tid].resize(4096*th_size + 1, 0);
                      uint32_t count = 0;
                    if(row_indices.size() == 0) {
                      for (auto i = begin; i < end; i++) {
                        common::Range1d r = space.GetRange(i);
                        JustPartitionColumnar<BinIdxType, true>(r.begin(), r.end(), gmat, n_features,
                                      nullptr, vec_rows_[tid].data(), count, numa,
                                      nodes_ids, split_conditions, split_ind, mask, threads_nodes_count[tid].data(), column_matrix, row_indices_ptr);//, column_matrix);
                      }
                    } else {
                      for (auto i = begin; i < end; i++) {
                        common::Range1d r = space.GetRange(i);
                        JustPartitionColumnar<BinIdxType, false>(r.begin(), r.end(), gmat, n_features,
                                      nullptr, vec_rows_[tid].data(), count, numa,
                                      nodes_ids, split_conditions, split_ind, mask, threads_nodes_count[tid].data(), column_matrix, row_indices_ptr);//, column_matrix);
                      }
                    }
                      vec_rows_[tid][0] = count;
                  }
              } else {
                #pragma omp parallel num_threads(nthreads)
                  {
                      size_t tid = omp_get_thread_num();
                      threads_nodes_count[tid].resize(1 << depth, 0);
                      const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
                      size_t chunck_size =
                          num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

                      size_t begin = chunck_size * tid;
                      size_t end = std::min(begin + chunck_size, num_blocks_in_space);
                      uint64_t local_time_alloc = 0;
                    const size_t th_size = end > begin ? end - begin : 0;
                      vec_rows_[tid].resize(4096*th_size + 1, 0);
                      uint32_t count = 0;
                      if(row_indices.size() == 0) {
                        for (auto i = begin; i < end; i++) {
                          common::Range1d r = space.GetRange(i);
                          JustPartitionWithLeafsMaskColumn<BinIdxType, true>(r.begin(), r.end(), gmat, n_features,
                                        nullptr, vec_rows_[tid].data(), count, numa,
                                        nodes_ids, split_conditions, split_ind, mask, leaf_mask, &prev_level_nodes_, threads_nodes_count[tid].data(), column_matrix, row_indices_ptr);
                        }
                      } else {
                        for (auto i = begin; i < end; i++) {
                          common::Range1d r = space.GetRange(i);
                          JustPartitionWithLeafsMaskColumn<BinIdxType, false>(r.begin(), r.end(), gmat, n_features,
                                        nullptr, vec_rows_[tid].data(), count, numa,
                                        nodes_ids, split_conditions, split_ind, mask, leaf_mask, &prev_level_nodes_, threads_nodes_count[tid].data(), column_matrix, row_indices_ptr);
                        }
                      }
                      //std::cout << "count: " << count << std::endl;
                      vec_rows_[tid][0] = count;
                  }
              }
                uint32_t summ_size1 = 0;

                for(uint32_t i = 0; i < nthreads; ++i) {
                  summ_size1 += vec_rows_[i][0];
                }
threads_rows_nodes_wise_.resize(nthreads);
const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();

if (n_features*summ_size1 / nthreads < (1 << (depth + 2))*n_bins || (depth > 2 && !hist_fit_to_l2) || (n_features == 61) && depth > 4) {
  threads_id_for_nodes_.resize(1 << max_depth);
 // std::cout << "\n no reason to read sequentialy!: " << depth << ":" <<  n_features*summ_size1 / nthreads << std::endl;
  std::vector<std::vector<int>> nodes_count(nthreads);
  #pragma omp parallel num_threads(nthreads)
  {
    size_t tid = omp_get_thread_num();
    threads_rows_nodes_wise_[tid].resize(vec_rows_[tid][0],0);

    nodes_count[tid].resize((1 << depth) + 1, 0);
    for(size_t i = 1; i < (1<<depth); ++i){
      nodes_count[tid][i + 1] += nodes_count[tid][i] + threads_nodes_count[tid][i-1];
    }
    for(size_t i = 0; i < vec_rows_[tid][0]; ++i) {
      const uint32_t row_id = vec_rows_[tid][i + 1];
      const uint32_t nod_id = nodes_ids[row_id];
      // CHECK_LT(nod_id, 1<<depth);
      // CHECK_LT(nodes_count[nod_id], vec_rows_[tid][0]);
      threads_rows_nodes_wise_[tid][nodes_count[tid][nod_id + 1]++] = row_id;
    }
    std::copy(threads_rows_nodes_wise_[tid].data(), threads_rows_nodes_wise_[tid].data() + vec_rows_[tid][0], vec_rows_[tid].data()+1);
  }
  // std::cout << depth << " - threads_nodes_count: " << std::endl;
//   for (size_t i = 0; i < nthreads; ++i) {
//     std::cout << "i-" << i << ": ";
//     for (size_t j = 0; j < (1 << depth); ++j) {
//       std::cout << threads_nodes_count[i][j] << "  ";
//     }
//     std::cout << "\n";
//   }
//   std::cout << std::endl;

//   std::cout << "nodes_count: " << std::endl;
//   for (size_t i = 0; i < nthreads; ++i) {
//     std::cout << "i-" << i << ": ";
//     for (size_t j = 0; j < (1 << depth) + 1; ++j) {
//       std::cout << nodes_count[i][j] << "  ";
//     }
//     std::cout << "\n";
//   }
//   std::cout << std::endl;
// std::cout << nodes_count.size() << "  " << nodes_count[0].size() << std::endl;

                uint32_t block_size = summ_size1/nthreads + !!(summ_size1%nthreads);
                uint32_t node_id = 0;
                uint32_t curr_thread_size = block_size;
//                std::cout << "summ_size1: " << summ_size1 << "\n";
//                std::cout << "curr_thread_size: " << curr_thread_size << "\n";
                uint32_t curr_node_disp = 0;
                uint32_t curr_thread_id = 0;
                for(uint32_t i = 0; i < nthreads; ++i) {
                  //std::cout << i << std::endl;
                  while (curr_thread_size != 0) {
                    const uint32_t curr_thread_node_size = threads_nodes_count[curr_thread_id%nthreads][node_id];
//                    std::cout << "curr_thread_node_size: " << curr_thread_node_size << std::endl;
//                    std::cout << "curr_thread_size: " << curr_thread_size << std::endl;
                    if (curr_thread_node_size == 0) {
                      ++curr_thread_id;
                      node_id = curr_thread_id / nthreads;
                    } else if (curr_thread_node_size > 0 && curr_thread_node_size <= curr_thread_size) {
//                      std::cout << "2.1-curr_thread_size: " << curr_thread_size << std::endl;
                      const uint32_t begin = 1 + nodes_count[curr_thread_id%nthreads][node_id];
                      CHECK_EQ(nodes_count[curr_thread_id%nthreads][node_id] + curr_thread_node_size,
                               nodes_count[curr_thread_id%nthreads][node_id+1]);
//                      std::cout << "begin: " << begin << std::endl;
                      threads_addr_[i].push_back({vec_rows_[curr_thread_id%nthreads].data(), begin,
                        begin + curr_thread_node_size});
                      //  std::cout << "node_id: " << node_id << " thr-i: " << i << " threads_id_for_nodes_.size()" << threads_id_for_nodes_.size() << " threads_id_for_nodes_[node_id].size():" << threads_id_for_nodes_[node_id].size() <<  std::endl;
                      if (threads_id_for_nodes_[node_id].size() != 0) {
                       if (threads_id_for_nodes_[node_id].back() != i) {
                         threads_id_for_nodes_[node_id].push_back(i);
                       }
                      } else {
                       threads_id_for_nodes_[node_id].push_back(i);
                      }
                      threads_nodes_count[curr_thread_id%nthreads][node_id] = 0;
                      curr_thread_size -= curr_thread_node_size;
//                      std::cout << "2.2-curr_thread_size: " << curr_thread_size << std::endl;
                      ++curr_thread_id;
                      node_id = curr_thread_id / nthreads;
                    } else {
                      const uint32_t begin = 1 + nodes_count[curr_thread_id%nthreads][node_id];
                      CHECK_EQ(nodes_count[curr_thread_id%nthreads][node_id] + curr_thread_node_size,
                               nodes_count[curr_thread_id%nthreads][node_id+1]);

                      threads_addr_[i].push_back({vec_rows_[curr_thread_id%nthreads].data(), begin,
                        begin + curr_thread_size});
                      if (threads_id_for_nodes_[node_id].size() != 0) {
                       if (threads_id_for_nodes_[node_id].back() != i) {
                         threads_id_for_nodes_[node_id].push_back(i);
                       }
                      } else {
                       threads_id_for_nodes_[node_id].push_back(i);
                      }
                      threads_nodes_count[curr_thread_id%nthreads][node_id] -= curr_thread_size;
                      nodes_count[curr_thread_id%nthreads][node_id] += curr_thread_size;
                      curr_thread_size = 0;
                    }
                  }
                  curr_thread_size = std::min(block_size, summ_size1 > block_size*(i+1) ? summ_size1 - block_size*(i+1) : 0);
                }
//std::cout << "\nNEW WORK PREPARATION was DONE!\n";
// std::cout << "threads_addr_.sizes:";
// for(size_t i = 0; i < nthreads; ++i) {
//   std::cout << threads_addr_[i].size() << "  ";
// }
// std::cout << "\n";
}  else {
                //std::cout << "depth: " << depth << " summ_size1: " <<  summ_size1 << std::endl;
                uint32_t block_size = summ_size1/nthreads + !!(summ_size1%nthreads);
                uint32_t curr_vec_rows_id = 0;
                uint32_t curr_vec_rows_size = vec_rows_[curr_vec_rows_id][0];
                uint32_t curr_thread_size = block_size;
                for(uint32_t i = 0; i < nthreads; ++i) {
                  while (curr_thread_size != 0) {
                    if(curr_vec_rows_size > curr_thread_size) {
                      threads_addr_[i].push_back({vec_rows_[curr_vec_rows_id].data(),
                                                1 + vec_rows_[curr_vec_rows_id][0] - curr_vec_rows_size,
                                                1 + vec_rows_[curr_vec_rows_id][0] - curr_vec_rows_size + curr_thread_size});
                      curr_vec_rows_size -= curr_thread_size;
                      curr_thread_size = 0;
                    } else if (curr_vec_rows_size == curr_thread_size) {
                      threads_addr_[i].push_back({vec_rows_[curr_vec_rows_id].data(),
                                                1 + vec_rows_[curr_vec_rows_id][0] - curr_vec_rows_size,
                                                1 + vec_rows_[curr_vec_rows_id][0] - curr_vec_rows_size + curr_thread_size});
                      curr_vec_rows_id += (curr_vec_rows_id < (nthreads - 1));
                      curr_vec_rows_size = vec_rows_[curr_vec_rows_id][0];
                      curr_thread_size = 0;
                    } else {
                      threads_addr_[i].push_back({vec_rows_[curr_vec_rows_id].data(),
                                                1 + vec_rows_[curr_vec_rows_id][0] - curr_vec_rows_size,
                                                1 + vec_rows_[curr_vec_rows_id][0]});
                      curr_thread_size -= curr_vec_rows_size;
                      curr_vec_rows_id += (curr_vec_rows_id < (nthreads - 1));
                      curr_vec_rows_size = vec_rows_[curr_vec_rows_id][0];
                    }
                  }
                  curr_thread_size = std::min(block_size, summ_size1 > block_size*(i+1) ? summ_size1 - block_size*(i+1) : 0);
                }
              }
            } else {
                #pragma omp parallel num_threads(nthreads)
                  {
                      size_t tid = omp_get_thread_num();
                      const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
                      size_t chunck_size =
                          num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

                      size_t begin = chunck_size * tid;
                      size_t end = std::min(begin + chunck_size, num_blocks_in_space);
                      const size_t th_size = end > begin ? end - begin : 0;
                      vec_rows_[tid].resize(4096*th_size + 1, 0);
                      uint64_t local_time_alloc = 0;
                      uint32_t count = 0;
                      if(row_indices.size() == 0) {
                        for (auto i = begin; i < end; i++) {
                          common::Range1d r = space.GetRange(i);
                          JustPartitionLastLayerColumn<BinIdxType, true>(r.begin(), r.end(), gmat, n_features,
                                        nullptr, vec_rows_[tid].data(), count, numa,
                                        nodes_ids, split_conditions, split_ind, &curr_level_nodes, leaf_mask, &prev_level_nodes_, column_matrix, row_indices_ptr);
                        }
                      } else {
                        for (auto i = begin; i < end; i++) {
                          common::Range1d r = space.GetRange(i);
                          JustPartitionLastLayerColumn<BinIdxType, false>(r.begin(), r.end(), gmat, n_features,
                                        nullptr, vec_rows_[tid].data(), count, numa,
                                        nodes_ids, split_conditions, split_ind, &curr_level_nodes, leaf_mask, &prev_level_nodes_, column_matrix, row_indices_ptr);
                        }
                      }
                      vec_rows_[tid][0] = count;
                  }
            }


            }
            prev_level_nodes_ = curr_level_nodes;

builder_monitor_.Stop("JustPartition!!!!!!" + depth_str);
}


template<typename GradientSumT>
template<bool is_distributed>
void QuantileHistMaker::Builder<GradientSumT>::DenseSync(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, int depth,
    std::vector<std::vector<GradientSumT>>* histograms, uint16_t* nodes_ids, std::vector<int32_t>* split_conditions,
    std::vector<bst_uint>* split_ind, const ColumnMatrix *column_matrix, uint64_t* mask, uint64_t* leaf_mask, int max_depth, common::BlockedSpace2d* space_ptr,
    int starting_index, int sync_count) {
  const size_t n_bins = gmat.cut.Ptrs().back();
  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  common::BlockedSpace2d& space = *space_ptr;
  int nthreads = this->nthread_;
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);

  builder_monitor_.Start("BuildHistSync!!");

if(depth == 0) {
  builder_monitor_.Start("BuildHistSync: depth0");
  //for (size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
   const int32_t nid = 0;//qexpand_depth_wise_[i].nid;
   GradientSumT* dest_hist = reinterpret_cast<GradientSumT*>(hist_[nid].data());
   const size_t block_size = 2*n_bins / nthreads + !!(2*n_bins % nthreads);
#pragma omp parallel num_threads(nthreads)
  {
   size_t tid = omp_get_thread_num();
   const size_t begin = tid*block_size;
   const size_t end = (begin + block_size) < 2*n_bins ? (begin + block_size) : 2*n_bins;

   for (size_t bin_id = begin; bin_id < end; ++bin_id) {
     dest_hist[bin_id] = (*histograms)[0][bin_id];
     (*histograms)[0][bin_id] = 0;
   }
   for (size_t tid = 1; tid < nthreads; ++tid) {
     for (size_t bin_id = begin; bin_id < end; ++bin_id) {
       dest_hist[bin_id] += (*histograms)[tid][bin_id];
       (*histograms)[tid][bin_id] = 0;
     }
   }
  if(is_distributed) {
  GradientSumT* this_local = reinterpret_cast<GradientSumT*>(hist_local_worker_[nid].data());
    for (size_t bin_id = begin; bin_id < end; ++bin_id) {
       this_local[bin_id] = dest_hist[bin_id];
     }
  }
  }
  //}
  builder_monitor_.Stop("BuildHistSync: depth0");

} else if (depth < max_depth){
  builder_monitor_.Start("BuildHistSync: preporation");

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
    const uint32_t summ_size_bin = n_bins*smallest.size();
    uint32_t block_size = summ_size_bin/nthreads + !!(summ_size_bin%nthreads);
    std::vector<std::vector<NodesBeginEnd>> threads_work(nthreads);
    const uint32_t node_full_size = n_bins;
    uint32_t curr_node_id = 0;
    uint32_t curr_node_size = node_full_size;
    uint32_t curr_thread_size = block_size;
    for(uint32_t i = 0; i < nthreads; ++i) {
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
  builder_monitor_.Stop("BuildHistSync: preporation");

std::string depth_str = std::to_string(depth);
  builder_monitor_.Start("BuildHistSync: depth " + depth_str);

if (threads_id_for_nodes_.size() == 0) {
#pragma omp parallel num_threads(nthreads)
  {
    const size_t block_size1 = 2*n_bins / nthreads + !!(2*n_bins % nthreads);
    size_t tid = omp_get_thread_num();
    const size_t begin = tid*block_size1;
    const size_t end = (begin + block_size1) < 2*n_bins ? (begin + block_size1) : 2*n_bins;
    for (size_t i = 0; i < smallest.size(); ++i) {
      const int32_t nid = qexpand_depth_wise_[smallest[i]].nid;
      const int32_t nid_c = compleate_trees_depth_wise_[smallest[i]];

      GradientSumT* dest_hist = reinterpret_cast<GradientSumT*>(hist_[nid].data());

      for (size_t bin_id = begin; bin_id < end; ++bin_id) {
        dest_hist[bin_id] = (*histograms)[0][nid_c*2*n_bins + bin_id];
        (*histograms)[0][nid_c*2*n_bins + bin_id] = 0;
      }
      for (size_t tid = 1; tid < nthreads; ++tid) {
        for (size_t bin_id = begin; bin_id < end; ++bin_id) {
          dest_hist[bin_id] += (*histograms)[tid][nid_c*2*n_bins + bin_id];
          (*histograms)[tid][nid_c*2*n_bins + bin_id] = 0;
        }
      }
    }
  }
} else {
#pragma omp parallel num_threads(nthreads)
  {
    const size_t block_size1 = 2*n_bins / nthreads + !!(2*n_bins % nthreads);
    size_t tid = omp_get_thread_num();
    const size_t begin = tid*block_size1;
    const size_t end = (begin + block_size1) < 2*n_bins ? (begin + block_size1) : 2*n_bins;
    for (size_t i = 0; i < smallest.size(); ++i) {
      const int32_t nid = qexpand_depth_wise_[smallest[i]].nid;
      const int32_t nid_c = compleate_trees_depth_wise_[smallest[i]];

      GradientSumT* dest_hist = reinterpret_cast<GradientSumT*>(hist_[nid].data());
      if (threads_id_for_nodes_[nid_c].size() != 0) {
        const size_t first_thread_id = threads_id_for_nodes_[nid_c][0];
        for (size_t bin_id = begin; bin_id < end; ++bin_id) {
          dest_hist[bin_id] = (*histograms)[first_thread_id][nid_c*2*n_bins + bin_id];
          (*histograms)[first_thread_id][nid_c*2*n_bins + bin_id] = 0;
        }
        for (size_t tid = 1; tid < threads_id_for_nodes_[nid_c].size(); ++tid) {
          const size_t thread_id = threads_id_for_nodes_[nid_c][tid];
          for (size_t bin_id = begin; bin_id < end; ++bin_id) {
            dest_hist[bin_id] += (*histograms)[thread_id][nid_c*2*n_bins + bin_id];
            (*histograms)[thread_id][nid_c*2*n_bins + bin_id] = 0;
          }
        }
      } else /*if (is_distributed) */ {
        const size_t first_thread_id = 0;
        for (size_t bin_id = begin; bin_id < end; ++bin_id) {
          dest_hist[bin_id] = 0;
          (*histograms)[first_thread_id][nid_c*2*n_bins + bin_id] = 0;
        }
      }
    }
  }
}
threads_id_for_nodes_.clear();
CHECK_EQ(threads_id_for_nodes_.size(), 0);
// #pragma omp parallel num_threads(nthreads)
//   {
//       size_t tid = omp_get_thread_num();
//       for(size_t i = 0; i < threads_work[tid].size(); ++i) {
//         const size_t begin = threads_work[tid][i].b * 2;
//         const size_t end = threads_work[tid][i].e * 2;

//         const int32_t nid_c = compleate_trees_depth_wise_[smallest[threads_work[tid][i].node_id]];
//         const int32_t nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].nid;
//         GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[nid].data());
//         const size_t block_size = 1024;
//         const size_t size = end > begin ? end - begin : 0;
//         const size_t n_blocks = size / block_size;
//         const size_t tail_size = size - n_blocks*block_size;
//         for (size_t block_id = 0; block_id < n_blocks; ++block_id) {
//           for (size_t bin_id = begin + block_id*block_size; bin_id < begin + (block_id+1)*block_size; ++bin_id) {
//             dest_hist[bin_id] = (*histograms)[0][2*nid_c*n_bins + bin_id];
//             (*histograms)[0][2*nid_c*n_bins + bin_id] = 0;
//           }
//           for (size_t tid = 1; tid < nthreads; ++tid) {
//             for (size_t bin_id = begin + block_id*block_size; bin_id < begin + (block_id+1)*block_size; ++bin_id) {
//               dest_hist[bin_id] += (*histograms)[tid][2*nid_c*n_bins + bin_id];
//               (*histograms)[tid][2*nid_c*n_bins + bin_id] = 0;
//             }
//           }
//         }
//         if (tail_size != 0) {
//           for (size_t bin_id = begin + n_blocks*block_size; bin_id < end; ++bin_id) {
//             dest_hist[bin_id] = (*histograms)[0][2*nid_c*n_bins + bin_id];
//             (*histograms)[0][2*nid_c*n_bins + bin_id] = 0;
//           }
//           for (size_t tid = 1; tid < nthreads; ++tid) {
//             for (size_t bin_id = begin + n_blocks*block_size; bin_id < end; ++bin_id) {
//               dest_hist[bin_id] += (*histograms)[tid][2*nid_c*n_bins + bin_id];
//               (*histograms)[tid][2*nid_c*n_bins + bin_id] = 0;
//             }
//           }
//         }
//       }
//   }
  builder_monitor_.Stop("BuildHistSync: depth " + depth_str);
  builder_monitor_.Start("Subtrick: depth " + depth_str);

#pragma omp parallel num_threads(nthreads)
  {
      size_t tid = omp_get_thread_num();
      for(size_t i = 0; i < threads_work[tid].size(); ++i) {
        const size_t begin = threads_work[tid][i].b * 2;
        const size_t end = threads_work[tid][i].e * 2;

        const int32_t small_nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].nid;
        const int32_t largest_nid = qexpand_depth_wise_[smallest[threads_work[tid][i].node_id]].sibling_nid;
        const size_t parent_id = (*p_tree)[small_nid].Parent();
        if (largest_nid > -1) {
          GradientSumT* dest_hist = reinterpret_cast< GradientSumT*>(hist_[largest_nid].data());
          GradientSumT* parent_hist = nullptr;
          if (is_distributed) {
            parent_hist = reinterpret_cast< GradientSumT*>(hist_local_worker_[parent_id].data());
          } else {
            parent_hist = reinterpret_cast< GradientSumT*>(hist_[parent_id].data());
          }
          GradientSumT* small_hist = reinterpret_cast< GradientSumT*>(hist_[small_nid].data());
          if (is_distributed) {
            auto this_local = hist_local_worker_[small_nid];
            auto this_hist = hist_[small_nid];
            CopyHist(this_local, this_hist, threads_work[tid][i].b, threads_work[tid][i].e);
          }
          for (size_t bin_id = begin; bin_id < end; ++bin_id) {
            dest_hist[bin_id] = parent_hist[bin_id] - small_hist[bin_id];
          }
          if (is_distributed) {
            auto sibling_local = hist_local_worker_[largest_nid];
            auto sibling_hist = hist_[largest_nid];
            CopyHist(sibling_local, sibling_hist, threads_work[tid][i].b, threads_work[tid][i].e);
          }
        }
      }
  }
  builder_monitor_.Stop("Subtrick: depth " + depth_str);


CHECK_EQ(smallest.size(), largest.size());
CHECK_EQ(nodes_for_explicit_hist_build_.size(), 0);

}
if (is_distributed) {
  uint64_t t1=get_time();
  builder_monitor_.Start("SyncHistogramsAllreduce");

  histred_.Allreduce(hist_[starting_index].data(),
                      hist_builder_.GetNumBins() * sync_count);

  builder_monitor_.Stop("SyncHistogramsAllreduce");
  time_AllReduce += get_time() - t1;
  common::BlockedSpace2d space1(nodes_for_explicit_hist_build_.size(), [&](size_t) {
    return n_bins;
  }, 1024);

  //ParallelSubtractionHist(this, space1, nodes_for_explicit_hist_build_, p_tree);
common::ParallelFor2d(space1, nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = nodes_for_explicit_hist_build_[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
        auto parent_hist = hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = hist_[entry.sibling_nid];
        SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(), r.end());
      }
    }
  });

  common::BlockedSpace2d space2(nodes_for_subtraction_trick_.size(), [&](size_t) {
    return n_bins;
  }, 1024);
//  ParallelSubtractionHist(this, space2, nodes_for_subtraction_trick_, p_tree);
common::ParallelFor2d(space2, nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = nodes_for_subtraction_trick_[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
        auto parent_hist = hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = hist_[entry.sibling_nid];
        SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(), r.end());
      }
    }
  });
}
builder_monitor_.Stop("BuildHistSync!!");

}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildLocalHistograms(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h) {
  builder_monitor_.Start("BuildLocalHistograms");

  const size_t n_nodes = nodes_for_explicit_hist_build_.size();

  // create space of size (# rows in each node)
  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    const int32_t nid = nodes_for_explicit_hist_build_[node].nid;
    return row_set_collection_[nid].Size();
  }, 256);

  std::vector<GHistRowT> target_hists(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    target_hists[i] = hist_[nid];
  }

  hist_buffer_.Reset(this->nthread_, n_nodes, space, target_hists);

  // Parallel processing by nodes and data in each node
  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
    const int32_t nid = nodes_for_explicit_hist_build_[nid_in_set].nid;

    auto start_of_row_set = row_set_collection_[nid].begin;
    auto rid_set = RowSetCollection::Elem(start_of_row_set + r.begin(),
                                      start_of_row_set + r.end(),
                                      nid);
    BuildHist(gpair_h, rid_set, gmat, gmatb, hist_buffer_.GetInitializedHist(tid, nid_in_set));
  });

  builder_monitor_.Stop("BuildLocalHistograms");
}

template<typename GradientSumT>
template<typename BinIdxType>
void QuantileHistMaker::Builder<GradientSumT>::BuildLocalHistogramsDense(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, int depth,
    std::vector<std::vector<GradientSumT>>* histograms, uint16_t* nodes_ids, std::vector<int32_t>* split_conditions,
    std::vector<bst_uint>* split_ind, const ColumnMatrix *column_matrix, uint64_t* mask, uint64_t* leaf_mask, int max_depth, common::BlockedSpace2d* space_ptr) {

builder_monitor_.Start("BuildLocalHistograms FULL");
  std::string timer_name = "BuildLocalHistograms:";
  timer_name += std::to_string(depth);
  const size_t n_bins = gmat.cut.Ptrs().back();
  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  common::BlockedSpace2d& space = *space_ptr;
  int nthreads = this->nthread_;
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);


  // std::vector<size_t> smallest;
  // std::vector<size_t> largest;
  // for (size_t i = 0; i < qexpand_depth_wise_.size(); ++i) {
  //  const int32_t nid_c = compleate_trees_depth_wise_[i];
  //  if(((uint64_t)(1) << (nid_c%64)) & *(mask + nid_c/64)) {
  //    smallest.push_back(i);
  //  } else {
  //    largest.push_back(i);
  //  }
  // }
  const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();

if(depth < max_depth) {
  builder_monitor_.Start(timer_name);

  if(depth == 0) {
    if (!hist_fit_to_l2) {
      #pragma omp parallel num_threads(nthreads)
        {
            size_t tid = omp_get_thread_num();
            GradientSumT* hist = (*histograms)[tid].data();// + nid_c*2*n_bins;
            // for (size_t bin_id = 0; bin_id < 2*n_bins; ++bin_id) {
            //   hist[bin_id] = 0;
            // }
            const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
            size_t chunck_size =
                num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

            size_t begin = chunck_size * tid;
            size_t end = std::min(begin + chunck_size, num_blocks_in_space);
            uint64_t local_time_alloc = 0;
            if ((*row_set_collection_.Data()).size() == 0) {
              for (auto i = begin; i < end; i++) {
                common::Range1d r = space.GetRange(i);
                GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
                BuildHistKernel<BinIdxType, true, true>(gpair_h, r.begin(), r.end(), gmat, n_features,  local_hist, numa, nodes_ids, offsets64_[tid].data(), nullptr, column_matrix);
              }
            } else {
              for (auto i = begin; i < end; i++) {
                common::Range1d r = space.GetRange(i);
                GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
                BuildHistKernel<BinIdxType, false, true>(gpair_h, r.begin(), r.end(), gmat, n_features,  local_hist, numa, nodes_ids, offsets64_[tid].data(), (*row_set_collection_.Data()).data(), column_matrix);
              }
            }
        }
    } else {
      #pragma omp parallel num_threads(nthreads)
        {
            size_t tid = omp_get_thread_num();
            GradientSumT* hist = (*histograms)[tid].data();// + nid_c*2*n_bins;
            // for (size_t bin_id = 0; bin_id < 2*n_bins; ++bin_id) {
            //   hist[bin_id] = 0;
            // }
            const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
            size_t chunck_size =
                num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

            size_t begin = chunck_size * tid;
            size_t end = std::min(begin + chunck_size, num_blocks_in_space);
            uint64_t local_time_alloc = 0;
            if ((*row_set_collection_.Data()).size() == 0) {
              for (auto i = begin; i < end; i++) {
                common::Range1d r = space.GetRange(i);
                GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
                BuildHistKernel<BinIdxType, true, false>(gpair_h, r.begin(), r.end(), gmat, n_features,  local_hist, numa, nodes_ids, offsets64_[tid].data(), nullptr, column_matrix);
              }
            } else {
              for (auto i = begin; i < end; i++) {
                common::Range1d r = space.GetRange(i);
                GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
                BuildHistKernel<BinIdxType, false, false>(gpair_h, r.begin(), r.end(), gmat, n_features,  local_hist, numa, nodes_ids, offsets64_[tid].data(), (*row_set_collection_.Data()).data(), column_matrix);
              }
            }
        }
    }
  } else {
if(!hist_fit_to_l2) {
  if(depth <= 2) {
    #pragma omp parallel num_threads(nthreads)
    {
          size_t tid = omp_get_thread_num();
          const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
          std::vector<AddrBeginEnd>& local_thread_addr = threads_addr_[tid];
          GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
            for(uint32_t block_id = 0; block_id < local_thread_addr.size(); ++block_id) {
              const uint32_t* rows = local_thread_addr[block_id].addr + local_thread_addr[block_id].b;
              const uint32_t size_r = local_thread_addr[block_id].e - local_thread_addr[block_id].b;
              BuildHistKernel<false, BinIdxType, 1, true, false>(gpair_h, rows, size_r, gmat, n_features,
                                                            local_hist, numa, nodes_ids, 1 << depth, offsets64_[tid].data(), column_matrix, 0);
            }
    }
  } else {
    #pragma omp parallel num_threads(nthreads)
    {
          size_t tid = omp_get_thread_num();
          const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
          std::vector<AddrBeginEnd>& local_thread_addr = threads_addr_[tid];
          GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
            for(uint32_t block_id = 0; block_id < local_thread_addr.size(); ++block_id) {
              const uint32_t* rows = local_thread_addr[block_id].addr + local_thread_addr[block_id].b;
              const uint32_t size_r = local_thread_addr[block_id].e - local_thread_addr[block_id].b;
              BuildHistKernel<false, BinIdxType, 1, false, true>(gpair_h, rows, size_r, gmat, n_features,
                                                            local_hist, numa, nodes_ids, 1 << depth, offsets64_[tid].data(), column_matrix, 0);
            }
    }
  }
} else {
    #pragma omp parallel num_threads(nthreads)
    {
          size_t tid = omp_get_thread_num();
          const BinIdxType* numa = tid < nthreads/2 ?  gmat.index.data<BinIdxType>() : gmat.index.data2<BinIdxType>();
          std::vector<AddrBeginEnd>& local_thread_addr = threads_addr_[tid];
          GHistRow<GradientSumT> local_hist(reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>((*histograms)[tid].data()), n_bins);
          for(uint32_t block_id = 0; block_id < local_thread_addr.size(); ++block_id) {
              const uint32_t* rows = local_thread_addr[block_id].addr + local_thread_addr[block_id].b;
              const uint32_t size_r = local_thread_addr[block_id].e - local_thread_addr[block_id].b;
              BuildHistKernel<false, BinIdxType, 1, false, false>(gpair_h, rows, size_r, gmat, n_features,
                                                            local_hist, numa, nodes_ids, 1 << depth, offsets64_[tid].data(), column_matrix, 0);
          }
    }

}
  }

  builder_monitor_.Stop(timer_name);

}

builder_monitor_.Stop("BuildLocalHistograms FULL");
threads_addr_.clear();
CHECK_EQ(threads_addr_.size(), 0);
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
void QuantileHistMaker::Builder<GradientSumT>::BuildNodeStats(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h) {
  builder_monitor_.Start("BuildNodeStats");
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gmat, gpair_h, *p_fmat, *p_tree);
    // add constraints
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      // it's a right child
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_feature_id = snode_[parent_id].best.SplitIndex();
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
  auto evaluator = tree_evaluator_.GetEvaluator();
  size_t i = 0;
  CHECK_EQ(compleate_trees_depth_wise_.size(), qexpand_depth_wise_.size());
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
//          std::cout << "node " << nid << " is leaf!: " << compleate_trees_depth_wise_[i] << " depth: " << depth <<  std::endl;

      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      *(leaf_mask + compleate_trees_depth_wise_[i]/64) |= ((uint64_t)(1) << (compleate_trees_depth_wise_[i] % 64));
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
// if (n_call == 92 || n_call == 93 || n_call == 94)
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
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const GHistIndexMatrix &gmat,
          RegTree *p_tree,
          int *num_leaves,
          int depth,
          unsigned *timestamp,
          std::vector<ExpandEntry>* nodes_for_apply_split,
          std::vector<ExpandEntry>* temp_qexpand_depth) {
  auto evaluator = tree_evaluator_.GetEvaluator();
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      int left_id = (*p_tree)[nid].LeftChild();
      int right_id = (*p_tree)[nid].RightChild();
      temp_qexpand_depth->push_back(ExpandEntry(left_id, right_id,
                                                p_tree->GetDepth(left_id), 0.0, (*timestamp)++));
      temp_qexpand_depth->push_back(ExpandEntry(right_id, left_id,
                                                p_tree->GetDepth(right_id), 0.0, (*timestamp)++));
      // - 1 parent + 2 new children
      (*num_leaves)++;
    }
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
    std::vector<ExpandEntry> *temp_qexpand_depth) {
  EvaluateSplits(qexpand_depth_wise_, gmat, hist_, *p_tree);

  std::vector<ExpandEntry> nodes_for_apply_split;
  AddSplitsToTree(gmat, p_tree, num_leaves, depth, timestamp,
                  &nodes_for_apply_split, temp_qexpand_depth);
  ApplySplit(nodes_for_apply_split, gmat, column_matrix, hist_, p_tree);
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
template<bool isDense>
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
      if (isDense) {
        big_siblings->push_back(entry);
      } else {
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
  const std::vector<GradientPair> &gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(ExpandEntry::kRootNid), 0.0, timestamp++));
  ++num_leaves;
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    std::vector<ExpandEntry> temp_qexpand_depth;
    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
    hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
    BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h);
    hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h);

    EvaluateAndApplySplits(gmat, column_matrix, p_tree, &num_leaves, depth, &timestamp,
                   &temp_qexpand_depth);

    // clean up
    qexpand_depth_wise_.clear();
    nodes_for_subtraction_trick_.clear();
    nodes_for_explicit_hist_build_.clear();
    if (temp_qexpand_depth.empty()) {
      break;
    } else {
      qexpand_depth_wise_ = temp_qexpand_depth;
      temp_qexpand_depth.clear();
    }
  }
}

template<typename GradientSumT>
template<typename BinIdxType>
void QuantileHistMaker::Builder<GradientSumT>::ExpandWithDepthWiseDense(
  const GHistIndexMatrix &gmat,
  const GHistIndexBlockMatrix &gmatb,
  const ColumnMatrix &column_matrix,
  DMatrix *p_fmat,
  RegTree *p_tree,
  const std::vector<GradientPair> &gpair_h) {
uint64_t time_ExpandWithDepthWiseDense_t1 = get_time();
  saved_split_ind_.clear();
  saved_split_ind_.resize(1 << (param_.max_depth + 1), 0);
  if (histograms_.size() == 0) {
    const size_t n_threads = omp_get_max_threads();
    const size_t n_bins = gmat.cut.Ptrs().back();
    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    const uint32_t* offsets = gmat.index.Offset();
    histograms_.resize(n_threads);
    offsets64_.resize(n_threads);
    #pragma omp parallel num_threads(n_threads)
    {
      const size_t tid = omp_get_thread_num();
      histograms_[tid].resize(n_bins*(1 << (param_.max_depth)), 0);
      offsets64_[tid].resize(n_features*(1 << (param_.max_depth - 1)), 0);
      uint64_t* offsets640 = offsets64_[tid].data();

      for(size_t nid = 0; nid < (1 << (param_.max_depth - 1)); ++nid) {
        for(size_t i = 0; i < n_features; ++i) {
          offsets640[nid*n_features + i] = (uint64_t)(histograms_[tid].data() + nid*n_bins*2) + 16*(uint64_t)(offsets[i]);
        }
      }
    }


  }

  unsigned timestamp = 0;
  int num_leaves = 0;
      qexpand_depth_wise_.clear();
      compleate_trees_depth_wise_.clear();
  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(ExpandEntry::kRootNid), 0.0, timestamp++));
  compleate_trees_depth_wise_.emplace_back(0);
  ++num_leaves;
  node_ids_.resize(gmat.row_ptr.size() - 1,0);
  std::vector<bst_uint> split_indexs(1 << param_.max_depth + 1);
  std::vector<int32_t> split_values(1 << param_.max_depth + 1);
//std::cout << "split_conditions.size(): " << split_values.size() << " split_ind.size(): " << split_indexs.size() << std::endl;

    uint64_t leafs_mask[128] = {};

uint64_t n_call = 0;
++n_call;
//std::cout << "n_call: " << n_call << std::endl;
  // if(n_call == 94) {
  //   std::cout << std::endl;
  //     std::cout << gpair_h[43] << "   ";// 0.67698/0.218678
  //   std::cout << std::endl;
  // }
  std::vector<size_t>& row_indices = *row_set_collection_.Data();
  const size_t size_threads = row_indices.size() == 0 ? (gmat.row_ptr.size() - 1) : row_indices.size();
  if (param_.subsample >= 1.0f) {
    CHECK_EQ(row_indices.size(), 0);
  }
  //std::cout << "size_threads: " << size_threads << "\n";
  common::BlockedSpace2d space(1, [&](size_t node) {
     // return gmat.row_ptr.size() - 1;
     return size_threads;
  }, 4096);
uint64_t t1 = 0;
++N_CALL;
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    std::vector<ExpandEntry> temp_qexpand_depth;
    std::vector<uint16_t> tmp_compleate_trees_depth;
    SplitSiblings</*isDense*/ true>(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
    hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
    uint64_t mask[128] = {};
if(depth > 0) {
  t1 = get_time();
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h, mask, n_call);
  time_BuildNodeStats += get_time() - t1;
  //  std::cout << "\n BuildNodeStats finished" << std::endl;
  t1 = get_time();
    DensePartition<BinIdxType>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space);
  time_DensePartition += get_time() - t1;
  //  std::cout << "\n DensePartition finished" << std::endl;
  t1 = get_time();
    BuildLocalHistogramsDense<BinIdxType>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space);
  time_BuildLocalHistogramsDense += get_time() - t1;
  //  std::cout << "\n BuildLocalHistogramsDense finished" << std::endl;
  t1 = get_time();
    if(rabit::IsDistributed()) {
      DenseSync<true>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space, starting_index, sync_count);
    } else {
      DenseSync<false>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space, starting_index, sync_count);
    }
  time_DenseSync += get_time() - t1;
 //   std::cout << "\n DenseSync finished" << std::endl;
    for(uint32_t i = 0; i < 128; ++i) {
      leafs_mask[i] = 0;
    }
    // leafs_mask[0] = 0; leafs_mask[1] = 0; leafs_mask[2] = 0; leafs_mask[3] = 0; leafs_mask[4] = 0; leafs_mask[5] = 0; leafs_mask[6] = 0; leafs_mask[7] = 0;
    // leafs_mask[8] = 0; leafs_mask[9] = 0; leafs_mask[10] = 0; leafs_mask[11] = 0; leafs_mask[12] = 0; leafs_mask[13] = 0; leafs_mask[14] = 0; leafs_mask[15] = 0;
} else {
  t1 = get_time();
    DensePartition<BinIdxType>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space);
  time_DensePartition += get_time() - t1;
  //  std::cout << "\n 0DensePartition finished" << std::endl;
  t1 = get_time();
    BuildLocalHistogramsDense<BinIdxType>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space);
  time_BuildLocalHistogramsDense += get_time() - t1;
  //  std::cout << "\n 0BuildLocalHistogramsDense finished" << std::endl;
  t1 = get_time();
    if(rabit::IsDistributed()) {
      DenseSync<true>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space, starting_index, sync_count);
    } else {
      DenseSync<false>(gmat, gmatb, p_tree, gpair_h, depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space, starting_index, sync_count);
    }
  time_DenseSync += get_time() - t1;
  //  std::cout << "\n 0DenseSync finished" << std::endl;
  t1 = get_time();
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h);
  time_BuildNodeStats += get_time() - t1;
  //  std::cout << "\n 0BuildNodeStats finished" << std::endl;
}
  t1 = get_time();
    EvaluateAndApplySplits(gmat, column_matrix, p_tree, &num_leaves, depth, &timestamp,
                   &temp_qexpand_depth, &tmp_compleate_trees_depth, leafs_mask, &split_values, &split_indexs, n_call);
  time_EvaluateAndApplySplits += get_time() - t1;
   // std::cout << "\n EvaluateAndApplySplits finished depth: " << depth << std::endl;
    // clean up
    nodes_for_subtraction_trick_.clear();
    nodes_for_explicit_hist_build_.clear();
    if (temp_qexpand_depth.empty()) {
      if (depth != param_.max_depth) {
        BuildNodeStats(gmat, p_fmat, p_tree, gpair_h, mask, n_call);
        DensePartition<BinIdxType>(gmat, gmatb, p_tree, gpair_h, param_.max_depth, &histograms_, node_ids_.data(), &split_values, &split_indexs, &column_matrix, mask, leafs_mask, param_.max_depth, &space);
      }
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
  }
  //std::cout << "ExpandWithDepth finished" << std::endl;
  time_ExpandWithDepthWiseDense += get_time() - time_ExpandWithDepthWiseDense_t1;
if(N_CALL % 100 == 0) {
    std::cout << "[TIMER]:ExpandWithDepthWiseDense time,s: " <<  (double)(time_ExpandWithDepthWiseDense)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    BuildLocalHistogramsDense time,s: " <<  (double)(time_BuildLocalHistogramsDense)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    DenseSync time,s: " <<  (double)(time_DenseSync)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:        AllReduce time,s: " <<  (double)(time_AllReduce)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    DensePartition time,s: " <<  (double)(time_DensePartition)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    BuildNodeStats time,s: " <<  (double)(time_BuildNodeStats)/(double)(1000000000) << std::endl;
    std::cout << "[TIMER]:    EvaluateAndApplySplits time,s: " <<  (double)(time_EvaluateAndApplySplits)/(double)(1000000000) << std::endl;
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
    DMatrix *p_fmat, RegTree *p_tree) {
  builder_monitor_.Start("Update");
  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();
  tree_evaluator_ =
      TreeEvaluator(param_, p_fmat->Info().num_col_, GenericParameter::kCpuId);
  interaction_constraints_.Reset();
  p_last_fmat_mutable_ = p_fmat;

  this->InitData(gmat, gpair_h, *p_fmat, *p_tree);
  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  } else {
    //N_CALL = (param_.max_depth + 1) * 500;
    if (is_optimized_branch_) {
      switch (gmat.index.GetBinTypeSize()) {
        case common::kUint8BinsTypeSize: {
          ExpandWithDepthWiseDense<uint8_t>(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
          break;
        }
        case common::kUint16BinsTypeSize: {
          ExpandWithDepthWiseDense<uint16_t>(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
          break;
        }
        case common::kUint32BinsTypeSize: {
          ExpandWithDepthWiseDense<uint32_t>(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
          break;
        }
        default: {
          CHECK(false);  // no default behavior
        }
      }
    } else {
      ExpandWithDepthWise(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
    }
  }
//std::cout << "Pstats update??? : " << p_tree->param.num_nodes << std::endl;
  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
//std::cout << "nid: " << nid << std::endl;
    p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.GetHess());
  }
//std::cout << "Pstats update??? finished" << std::endl;
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});
//std::cout << "Pruner finished finished" << std::endl;
  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* p_out_preds, const int gid, const int ngroup) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_ ||
      p_last_fmat_ != p_last_fmat_mutable_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");

  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

  CHECK_GT(out_preds.size(), 0U);

  size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();

  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    return row_set_collection_[node].Size();
  }, 1024);

  common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
    const RowSetCollection::Elem rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr) {
      int nid = rowset.node_id;
      bst_float leaf_value;
      // if a node is marked as deleted by the pruner, traverse upward to locate
      // a non-deleted leaf.
      if ((*p_last_tree_)[nid].IsDeleted()) {
        while ((*p_last_tree_)[nid].IsDeleted()) {
          nid = (*p_last_tree_)[nid].Parent();
        }
        CHECK((*p_last_tree_)[nid].IsLeaf());
      }
      leaf_value = (*p_last_tree_)[nid].LeafValue();

      for (const size_t* it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
        out_preds[*it * ngroup + gid] += leaf_value;
      }
    }
  });

  if (param_.subsample < 1.0f) {
    // Making a real prediction for the remaining rows
    size_t fvecs_size = feat_vecs_.size();
    feat_vecs_.resize(omp_get_max_threads(), RegTree::FVec());
    while (fvecs_size < feat_vecs_.size()) {
      feat_vecs_[fvecs_size++].Init(data->Info().num_col_);
    }
    for (auto&& batch : p_last_fmat_mutable_->GetBatches<SparsePage>()) {
      HostSparsePageView page_view = batch.GetView();
      const auto num_parallel_ops = static_cast<bst_omp_uint>(unused_rows_.size());
      common::ParallelFor(num_parallel_ops, [&](bst_omp_uint block_id) {
        RegTree::FVec &feats = feat_vecs_[omp_get_thread_num()];
        const SparsePage::Inst inst = page_view[unused_rows_[block_id]];
        feats.Fill(inst);

        const size_t row_num = unused_rows_[block_id] + batch.base_rowid;
        const int lid = feats.HasMissing() ? p_last_tree_->GetLeafIndex<true>(feats) :
                                            p_last_tree_->GetLeafIndex<false>(feats);
        out_preds[row_num * ngroup + gid] += (*p_last_tree_)[lid].LeafValue();

        feats.Drop(inst);
      });
    }
  }
  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}


template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCacheDense(
    const DMatrix* data,
    HostDeviceVector<bst_float>* p_out_preds, const int gid, const int ngroup, const GHistIndexMatrix* gmat_ptr) {
  uint64_t t1 = 0;
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");
// int n_call = 0;
// ++n_call;
  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

  CHECK_GT(out_preds.size(), 0U);
  std::vector<size_t>& row_indices = *row_set_collection_.Data();

  if (row_indices.size() == 0) {
    CHECK_GE(param_.subsample, 1);
    common::BlockedSpace2d space(1, [&](size_t node) {
      return node_ids_.size();
    }, 1024);

    common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
      for (size_t it = r.begin(); it <  r.end(); ++it) {
        bst_float leaf_value;
        // if a node is marked as deleted by the pruner, traverse upward to locate
        // a non-deleted leaf.
        int nid = (~((uint16_t)(1) << 15)) & node_ids_[it];
        if ((*p_last_tree_)[nid].IsDeleted()) {
          while ((*p_last_tree_)[nid].IsDeleted()) {
            nid = (*p_last_tree_)[nid].Parent();
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
        }
        leaf_value = (*p_last_tree_)[nid].LeafValue();
        out_preds[it*ngroup + gid] += leaf_value;
      }
    });
  } else {
    common::BlockedSpace2d space(1, [&](size_t node) {
      return row_indices.size();
    }, 1024);
    common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
      for (size_t it = r.begin(); it <  r.end(); ++it) {
        bst_float leaf_value;
        // if a node is marked as deleted by the pruner, traverse upward to locate
        // a non-deleted leaf.
        const size_t row_id = row_indices[it];
        int nid = (~((uint16_t)(1) << 15)) & node_ids_[row_id];
        if ((*p_last_tree_)[nid].IsDeleted()) {
          while ((*p_last_tree_)[nid].IsDeleted()) {
            nid = (*p_last_tree_)[nid].Parent();
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
        }
        leaf_value = (*p_last_tree_)[nid].LeafValue();
        out_preds[row_id*ngroup + gid] += leaf_value;
      }
    });
  builder_monitor_.Start("UpdatePredictionCachePredict");
//std::cout << "UpdatePredictionCachePredict started!!!"  << std::endl;
//std::cout << "saved_split_ind_.data(): " << saved_split_ind_.data() << std::endl;
//std::cout << "gmat_ptr: " << gmat_ptr << std::endl;
    if (param_.subsample < 1.0f) {
        CHECK_LE(param_.max_bin, 256);
        const uint8_t* data = (*gmat_ptr).index.data<uint8_t>();
//std::cout << "data(): " << data << std::endl;
        const size_t n_features = (*gmat_ptr).cut.Ptrs().size() - 1;
//std::cout << "n_features(): " << n_features << std::endl;
        const uint32_t* offsets = (*gmat_ptr).index.Offset();
//std::cout << "offsets(): " << offsets << std::endl;
      // Making a real prediction for the remaining rows
        common::ParallelFor(unused_rows_.size(), [&](bst_omp_uint block_id) {
          const size_t row_id = unused_rows_[block_id];
          const uint8_t* feat = data + row_id * n_features;
          const int lid = p_last_tree_->GetLeafIndex(feat, offsets, saved_split_ind_.data());
          out_preds[row_id * ngroup + gid] += (*p_last_tree_)[lid].LeafValue();
        });
      }
//std::cout << "UpdatePredictionCachePredict started!!!"  << std::endl;
  builder_monitor_.Stop("UpdatePredictionCachePredict");
    }

  builder_monitor_.Stop("UpdatePredictionCache");
  time_UpdatePredictionCacheDense += get_time() - t1;
  if ((N_CALL + 1) % 100 == 0) {
    std::cout << "[TIMER]:UpdatePredictionCacheDense time,s: " <<  (double)(time_UpdatePredictionCacheDense)/(double)(1000000000) << std::endl;
  }
  return true;
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const std::vector<GradientPair>& gpair,
                                                const DMatrix& fmat,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  unused_rows_.resize(info.num_row_);
  size_t* p_row_indices_used = row_indices->data();
  size_t* p_row_indices_unused = unused_rows_.data();
#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  size_t used = 0, unused = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
      p_row_indices_used[used++] = i;
    } else {
      p_row_indices_unused[unused++] = i;
    }
  }
  /* resize row_indices to reduce memory */
  row_indices->resize(used);
  unused_rows_.resize(unused);
#else
  const size_t nthread = this->nthread_;
  std::vector<size_t> row_offsets_used(nthread, 0);
  std::vector<size_t> row_offsets_unused(nthread, 0);
  /* usage of mt19937_64 give 2x speed up for subsampling */
  std::vector<std::mt19937> rnds(nthread);
  /* create engine for each thread */
  for (std::mt19937& r : rnds) {
    r = rnd;
  }
  const size_t discard_size = info.num_row_ / nthread;
  auto upper_border = static_cast<float>(std::numeric_limits<uint32_t>::max());
  uint32_t coin_flip_border = static_cast<uint32_t>(upper_border * param_.subsample);
  dmlc::OMPException exc;
  #pragma omp parallel num_threads(nthread)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (nthread - 1)) ?
                          info.num_row_ : ibegin + discard_size;

      rnds[tid].discard(discard_size * tid);
      for (size_t i = ibegin; i < iend; ++i) {
        if (gpair[i].GetHess() >= 0.0f && rnds[tid]() < coin_flip_border) {
          p_row_indices_used[ibegin + row_offsets_used[tid]++] = i;
        } else {
          p_row_indices_unused[ibegin + row_offsets_unused[tid]++] = i;
        }
      }

      #pragma omp barrier

      if (tid == 0ul) {
        size_t prefix_sum_used = row_offsets_used[0];
        for (size_t i = 1; i < nthread; ++i) {
          const size_t ibegin = i * discard_size;

          for (size_t k = 0; k < row_offsets_used[i]; ++k) {
            p_row_indices_used[prefix_sum_used + k] = p_row_indices_used[ibegin + k];
          }

          prefix_sum_used += row_offsets_used[i];
        }
        /* resize row_indices to reduce memory */
        row_indices->resize(prefix_sum_used);
      }

      if (nthread == 1ul || tid == 1ul) {
        size_t prefix_sum_unused = row_offsets_unused[0];
        for (size_t i = 1; i < nthread; ++i) {
          const size_t ibegin = i * discard_size;

          for (size_t k = 0; k < row_offsets_unused[i]; ++k) {
            p_row_indices_unused[prefix_sum_unused + k] = p_row_indices_unused[ibegin + k];
          }

          prefix_sum_unused += row_offsets_unused[i];
        }
        /* resize row_indices to reduce memory */
        unused_rows_.resize(prefix_sum_unused);
      }
    });
  }
  exc.Rethrow();
  /* discard global engine */
  rnd = rnds[nthread - 1];
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const DMatrix& fmat,
                                          const RegTree& tree) {
  uint64_t t1 = get_time();
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

  {
    // initialize the row set
    row_set_collection_.Clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(nbins);
    hist_local_worker_.Init(nbins);
    hist_buffer_.Init(nbins);
    // initialize histogram builder
    dmlc::OMPException exc;
#pragma omp parallel
    {
      exc.Run([&]() {
        this->nthread_ = omp_get_num_threads();
      });
    }
    exc.Rethrow();
    hist_builder_ = GHistBuilder<GradientSumT>(this->nthread_, nbins);

    std::vector<size_t>& row_indices = *row_set_collection_.Data();
builder_monitor_.Start("InitDataResize");

    if (!is_optimized_branch_ || param_.subsample < 1.0f) {
      row_indices.resize(info.num_row_);
    }
builder_monitor_.Stop("InitDataResize");
    size_t* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, fmat, &row_indices);
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
      CHECK_EQ(row_indices.size() + unused_rows_.size(), info.num_row_);
    } else if (!is_optimized_branch_) {
      MemStackAllocator<bool, 128> buff(this->nthread_);
      bool* p_buff = buff.Get();
      std::fill(p_buff, p_buff + this->nthread_, false);

      const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

      #pragma omp parallel num_threads(this->nthread_)
      {
        exc.Run([&]() {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));

          for (size_t i = ibegin; i < iend; ++i) {
            if (gpair[i].GetHess() < 0.0f) {
              p_buff[tid] = true;
              break;
            }
          }
        });
      }
      exc.Rethrow();

      bool has_neg_hess = false;
      for (int32_t tid = 0; tid < this->nthread_; ++tid) {
        if (p_buff[tid]) {
          has_neg_hess = true;
        }
      }

      if (has_neg_hess) {
        size_t j = 0;
        for (size_t i = 0; i < info.num_row_; ++i) {
          if (gpair[i].GetHess() >= 0.0f) {
            p_row_indices[j++] = i;
          }
        }
        row_indices.resize(j);
      } else {
        #pragma omp parallel num_threads(this->nthread_)
        {
          exc.Run([&]() {
            const size_t tid = omp_get_thread_num();
            const size_t ibegin = tid * block_size;
            const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
                static_cast<size_t>(info.num_row_));
            for (size_t i = ibegin; i < iend; ++i) {
              p_row_indices[i] = i;
            }
          });
        }
        exc.Rethrow();
      }
    }
  }

  row_set_collection_.Init();

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
  time_InitData += get_time() - t1;
  if ((N_CALL + 1) % 100 == 0) {
    std::cout << "[TIMER]:InitData time,s: " <<  (double)(time_InitData)/(double)(1000000000) << std::endl;
  }
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
template <bool default_left, bool any_missing, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionDenseKernel(const common::DenseColumn<BinIdxType>& column,
      common::Span<const size_t> rid_span, const int32_t split_cond,
      common::Span<size_t> left_part, common::Span<size_t> right_part) {
  const int32_t offset = column.GetBaseIdx();
  const BinIdxType* idx = column.GetFeatureBinIdxPtr().data();
  size_t* p_left_part = left_part.data();
  size_t* p_right_part = right_part.data();
  size_t nleft_elems = 0;
  size_t nright_elems = 0;

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
    for (auto rid : rid_span)  {
      if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        p_left_part[nleft_elems++] = rid;
      } else {
        p_right_part[nright_elems++] = rid;
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
template <typename BinIdxType>
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
        child_nodes_sizes = PartitionDenseKernel<true, true>(column, rid_span, split_cond,
                                                             left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<true, false>(column, rid_span, split_cond,
                                                              left, right);
      }
    } else {
      if (column_matrix.AnyMissing()) {
        child_nodes_sizes = PartitionDenseKernel<false, true>(column, rid_span, split_cond,
                                                              left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<false, false>(column, rid_span, split_cond,
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
                                                     std::vector<int32_t>* split_conditions) {
  const size_t n_nodes = nodes.size();
  split_conditions->resize(n_nodes);

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
    (*split_conditions)[i] = split_cond;
  }
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
    saved_split_ind_[nid] = split_cond;
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
                                            RegTree* p_tree) {
  builder_monitor_.Start("ApplySplit");
  // 1. Find split condition for each split
  const size_t n_nodes = nodes.size();
  std::vector<int32_t> split_conditions;
  FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
  // 2.1 Create a blocked space of size SUM(samples in each node)
  common::BlockedSpace2d space(n_nodes, [&](size_t node_in_set) {
    int32_t nid = nodes[node_in_set].nid;
    return row_set_collection_[nid].Size();
  }, kPartitionBlockSize);
  // 2.2 Initialize the partition builder
  // allocate buffers for storage intermediate results by each thread
  partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
    const int32_t nid = nodes[node_in_set].nid;
    const size_t size = row_set_collection_[nid].Size();
    const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
    return n_tasks;
  });
  // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
  // Store results in intermediate buffers from partition_builder_
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    size_t begin = r.begin();
    const int32_t nid = nodes[node_in_set].nid;
    const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
    partition_builder_.AllocateForTask(task_id);
      switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        PartitionKernel<uint8_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      case common::kUint16BinsTypeSize:
        PartitionKernel<uint16_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      case common::kUint32BinsTypeSize:
        PartitionKernel<uint32_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    });
  // 3. Compute offsets to copy blocks of row-indexes
  // from partition_builder_ to row_set_collection_
  partition_builder_.CalculateRowOffsets();

  // 4. Copy elements from partition_builder_ to row_set_collection_ back
  // with updated row-indexes for each tree-node
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    const int32_t nid = nodes[node_in_set].nid;
    partition_builder_.MergeToArray(node_in_set, r.begin(),
        const_cast<size_t*>(row_set_collection_[nid].begin));
  });
  // 5. Add info about splits into row_set_collection_
  AddSplitsToRowSet(nodes, p_tree);
  builder_monitor_.Stop("ApplySplit");
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
  FindSplitConditions(nodes, *p_tree, gmat, split_conditions, compleate_splits);
  //std::cout << "\n FindSplitConditions finished \n";
  (*split_ind)[0] = n_nodes;
  for (size_t i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes[i].nid;
      const bst_uint fid = (*p_tree)[nid].SplitIndex();
      (*split_ind)[(*compleate_splits)[i] + 1] = fid;
  }

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
      if (snode_[parent_id].best.right_sum.GetHess() == snode_[parent_id].best.left_sum.GetHess() && tree[nid].IsLeftChild()) {
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

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitNewNode(int nid,
                                             const GHistIndexMatrix& gmat,
                                             const std::vector<GradientPair>& gpair,
                                             const DMatrix& fmat,
                                             const RegTree& tree) {
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
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint8_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint16_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<float>::PartitionKernel<uint32_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint8_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint16_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree);
template void QuantileHistMaker::Builder<double>::PartitionKernel<uint32_t>(
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
