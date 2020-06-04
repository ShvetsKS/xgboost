#include "extern_kernels.h"

namespace daal {

namespace primitives {

namespace xgboost {

template<typename FPType, bool do_prefetch, typename BinIdxType>
void ExternalKernelsSeqBuildHist(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const BinIdxType* gradient_index, const uint32_t* offsets,
                             FPType* hist_data) {
  printf("\nExternal DAAL Kernel!!!!\n");
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = rid[i] * n_features;
    const size_t idx_gh = two * rid[i];

    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t j = 0; j < n_features; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

} // xgboost

} // primitives

} // daal

template<typename FPType, bool do_prefetch, typename BinIdxType>
void XGBoostExternalKernels::X_SeqBuildHist(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const BinIdxType* gradient_index, const uint32_t* offsets,
                             FPType* hist_data) {
  daal::primitives::xgboost::ExternalKernelsSeqBuildHist<FPType, do_prefetch, BinIdxType>(size, n_features, rid, pgh, gradient_index, offsets, hist_data);
}

template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, true, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, false, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, true, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, false, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
//////////////////////////
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, true, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, false, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, true, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, false, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
///////////////////////////
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, true, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<double, false, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, true, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void Kernel<XGBoostExternalKernels>::SeqBuildHist<float, false, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
