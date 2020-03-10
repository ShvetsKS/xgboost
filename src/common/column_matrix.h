/*!
 * Copyright 2017 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#include <limits>
#include <vector>
#include "hist_util.h"


namespace xgboost {
namespace common {


/*! \brief column type */
enum ColumnType {
  kDenseColumn,
  kSparseColumn
};

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base. */
template <typename T>
class Column {
 public:
  Column(ColumnType type, const T* index, uint32_t index_base,
         const size_t* row_ind, size_t len, const uint8_t* missing_flags)
      : type_(type),
        index_(index),
        index_base_(index_base),
        row_ind_(row_ind),
        len_(len),
        missing_flags_(missing_flags) {}
  size_t Size() const { return len_; }
  uint32_t GetGlobalBinIdx(size_t idx) const { return index_base_ + (uint32_t)(index_[idx]); }
  T GetFeatureBinIdx(size_t idx) const { return index_[idx]; }
  common::Span<const T> GetFeatureBinIdxPtr() const { return { index_, len_ }; }
  // column.GetFeatureBinIdx(idx) + column.GetBaseIdx(idx) ==
  // column.GetGlobalBinIdx(idx)
  uint32_t GetBaseIdx() const { return index_base_; }
  ColumnType GetType() const { return type_; }
  size_t GetRowIdx(size_t idx) const {
    // clang-tidy worries that row_ind_ might be a nullptr, which is possible,
    // but low level structure is not safe anyway.
    return type_ == ColumnType::kDenseColumn ? idx : row_ind_[idx];  // NOLINT
  }
  bool IsMissing(size_t idx) const {
    return missing_flags_[idx] == 1;//index_[idx] == std::numeric_limits<T>::max();
  }
  const size_t* GetRowData() const { return row_ind_; }

  const uint8_t* missing_flags_;
 private:
  ColumnType type_;
  const T* index_;
  uint32_t index_base_;
  const size_t* row_ind_;
  const size_t len_;
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrix. */
class ColumnMatrix {
 public:
  // get number of features
  inline bst_uint GetNumFeature() const {
    return static_cast<bst_uint>(type_.size());
  }

  // construct column matrix from GHistIndexMatrix
  inline void Init(const GHistIndexMatrix& gmat,
                   double  sparse_threshold) {
    const int32_t nfeature = static_cast<int32_t>(gmat.cut.Ptrs().size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;

    // identify type of each column
    feature_counts_.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);

    uint32_t max_val = std::numeric_limits<uint32_t>::max();
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
    }
    bool all_dense = gmat.IsDense();
    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts_[fid])
                 < sparse_threshold * nrow) {
        type_[fid] = kSparseColumn;
        all_dense = false;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    boundary_.resize(nfeature);
    size_t accum_index_ = 0;
    size_t accum_row_ind_ = 0;
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      boundary_[fid].index_begin = accum_index_;
      boundary_[fid].row_ind_begin = accum_row_ind_;
      if (type_[fid] == kDenseColumn) {
        accum_index_ += static_cast<size_t>(nrow);
        accum_row_ind_ += static_cast<size_t>(nrow);
      } else {
        accum_index_ += feature_counts_[fid];
        accum_row_ind_ += feature_counts_[fid];
      }
      boundary_[fid].index_end = accum_index_;
      boundary_[fid].row_ind_end = accum_row_ind_;
    }

//    type_size_ = 1 << gmat.index.getBinBound();
//    std::cout << "\ngmat.max_num_bins_: " << gmat.max_num_bins_ << "\n";
    if ( (gmat.max_num_bins_ - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max()) ) {
      type_size_ = 1;
//    std::cout << "\ntype_size_: " << type_size_ << "\n";
    } else if ( (gmat.max_num_bins_ - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())){
      type_size_ = 2;
    } else {
      type_size_ = 4;
    }

    index_.resize(boundary_[nfeature - 1].index_end * type_size_);
    if (!all_dense) {
      row_ind_.resize(boundary_[nfeature - 1].row_ind_end);
    }

    // store least bin id for each feature
    index_base_.resize(nfeature);
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      index_base_[fid] = gmat.cut.Ptrs()[fid];
    }

    // pre-fill index_ for dense columns

missing_flags_.resize(boundary_[nfeature - 1].index_end);
    #pragma omp parallel for
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      if (type_[fid] == kDenseColumn) {
        const size_t ibegin = boundary_[fid].index_begin;
        uint8_t* begin = &missing_flags_[ibegin];
        uint8_t* end = begin + nrow;
        std::fill(begin, end, 1);
        // max() indicates missing values
      }
    }

    // loop over all rows and fill column entries
    // num_nonzeros[fid] = how many nonzeros have this feature accumulated so far?
    //std::vector<size_t> num_nonzeros;
    //num_nonzeros.resize(nfeature);
    //std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    if (all_dense) {
      switch (gmat.index.getBinBound()) {
        case POWER_OF_TWO_8:
          SetIndexAllDense(gmat.index.data<uint8_t>(), gmat, nrow);
          break;
        case POWER_OF_TWO_16:
          SetIndexAllDense(gmat.index.data<uint16_t>(), gmat, nrow);
          break;
        case POWER_OF_TWO_32:
          SetIndexAllDense(gmat.index.data<uint32_t>(), gmat, nrow);
          break;
      }
    } else {
      switch (type_size_) {
        case 1:
   //       std::cout << "\nSetIndex\n";
          SetIndex<uint8_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
   //       std::cout << "\nSetIndex end\n";
          break;
        case 2:
          SetIndex<uint16_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
        case 4:
          SetIndex<uint32_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
      }
    }
  }

  /* Fetch an individual column. This code should be used with XGBOOST_TYPE_SWITCH
     to determine type of bin id's */
  template <typename T>
  inline Column<T> GetColumn(unsigned fid) const {
    Column<T> c(type_[fid],
                reinterpret_cast<const T*>(&index_[boundary_[fid].index_begin * type_size_]),
                index_base_[fid], (type_[fid] == ColumnType::kSparseColumn ?
                &row_ind_[boundary_[fid].row_ind_begin] : nullptr),
                boundary_[fid].index_end - boundary_[fid].index_begin, &missing_flags_[boundary_[fid].index_begin]);
    return c;
  }

  template<typename T>
  inline void SetIndexAllDense(T* index, const GHistIndexMatrix& gmat,  const size_t nrow) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);
    for (size_t rid = 0; rid < nrow; ++rid) {
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      size_t fid = 0;
      size_t jp = 0;
      for (size_t i = ibegin; i < iend; ++i, ++jp) {
          const size_t idx = boundary_[jp].index_begin;
          T* begin = &local_index[idx];
          begin[rid] = index[i];
          missing_flags_[idx + rid] = 0;
      }
    }
  }

/*  inline void SetIndexAllDense(uint32_t* index, const GHistIndexMatrix& gmat,  const size_t nrow) {
    uint32_t* local_index = reinterpret_cast<uint32_t*>(&index_[0]);
    for (size_t rid = 0; rid < nrow; ++rid) {
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      size_t fid = 0;
      size_t jp = 0;
      for (size_t i = ibegin; i < iend; ++i, ++jp) {
          uint32_t* begin = &local_index[boundary_[jp].index_begin];
          begin[rid] = index[i] - index_base_[jp];
      }
    }
  }*/

  template<typename T>
  inline void SetIndex(uint32_t* index, const GHistIndexMatrix& gmat,
                       const size_t nrow, const size_t nfeature) {
    const SparsePage& batch = *(gmat.p_fmat_->GetBatches<SparsePage>().begin());

    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(nfeature);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    T* local_index = reinterpret_cast<T*>(&index_[0]);
//    std::cout << "\n++local_index before set: \n";
//    for(size_t i = 0; i < index_.size()/type_size_; ++i)
//      std::cout << local_index[i] << "   ";
//
//    std::cout << "\n++index_base_:\n";
//    for (int32_t fid = 0; fid < index_base_.size(); ++fid) {
//      std::cout << index_base_[fid] << "   ";
//    }

//    std::cout << "\n++bin_id: \n";

    for (size_t rid = 0; rid < nrow; ++rid) {
        const size_t ibegin = gmat.row_ptr[rid];
        const size_t iend = gmat.row_ptr[rid + 1];
        size_t fid = 0;
        SparsePage::Inst inst = batch[rid];

        size_t jp = 0;
        for (size_t i = ibegin; i < iend; ++i, ++jp) {
          const uint32_t bin_id = index[i]/* + disp[jp]*/;
//          std::cout << bin_id << "   ";
/*          auto iter = std::upper_bound(gmat.cut.Ptrs().cbegin() + fid,
                                       gmat.cut.Ptrs().cend(), bin_id);
          fid = std::distance(gmat.cut.Ptrs().cbegin(), iter) - 1;
*/
          fid = inst[jp].index;
          if (type_[fid] == kDenseColumn) {
            T* begin = &local_index[boundary_[fid].index_begin];
            begin[rid] = bin_id - index_base_[fid];
          //  std::cout <<  (uint32_t)begin[rid] << "   ";
            missing_flags_[boundary_[fid].index_begin + rid] = 0;
          } else {
            T* begin = &local_index[boundary_[fid].index_begin];
            begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
            missing_flags_[boundary_[fid].index_begin + num_nonzeros[fid]] = 0;
          //  std::cout <<  (uint32_t)begin[num_nonzeros[fid]]  << "   ";
            row_ind_[boundary_[fid].row_ind_begin + num_nonzeros[fid]] = rid;
            ++num_nonzeros[fid];
          }
//          ++jp;
        }
      }
/*
    for (size_t rid = 0; rid < nrow; ++rid) {
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      size_t fid = 0;
      for (size_t i = ibegin; i < iend; ++i) {
        const uint32_t bin_id = gmat.index[i];
        auto iter = std::upper_bound(gmat.cut.Ptrs().cbegin() + fid,
                                     gmat.cut.Ptrs().cend(), bin_id);
        fid = std::distance(gmat.cut.Ptrs().cbegin(), iter) - 1;
        if (type_[fid] == kDenseColumn) {
          uint32_t* begin = &index_[boundary_[fid].index_begin];
          begin[rid] = bin_id - index_base_[fid];
        } else {
          uint32_t* begin = &index_[boundary_[fid].index_begin];
          begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
          row_ind_[boundary_[fid].row_ind_begin + num_nonzeros[fid]] = rid;
          ++num_nonzeros[fid];
        }
      }
    }*/
  }
  const size_t GetTypeSize() const {
    return type_size_;
  }

 private:
  std::vector<uint8_t> index_;  // index_: may store smaller integers; needs padding
  struct ColumnBoundary {
    // indicate where each column's index and row_ind is stored.
    // index_begin and index_end are logical offsets, so they should be converted to
    // actual offsets by scaling with packing_factor_
    size_t index_begin;
    size_t index_end;
    size_t row_ind_begin;
    size_t row_ind_end;
  };

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  std::vector<ColumnBoundary> boundary_;

  // index_base_[fid]: least bin id for feature fid
  std::vector<uint32_t> index_base_;
  std::vector<uint8_t> missing_flags_;
  uint32_t type_size_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
