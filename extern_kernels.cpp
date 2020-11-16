#include "extern_kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
//    const uint32_t offset##IDX = two*(offsets[IDX + 16*J] + (uint32_t)(gr_index_local[IDX + 16*J])); \
//   const uint32_t offset##IDX = two*(((uint32_t*)&voffset)[IDX] + (uint32_t)(gr_index_local[IDX + 16*J])); \
// 3     const uint32_t offset##IDX = two*(voffset[IDX] + (uint32_t)(gr_index_local[IDX + 16*J])); \


 //two*(((uint32_t*)&voffset)[IDX] + (uint32_t)(gr_index_local[IDX + 16*J]));
#define UNR(IDX, J)                                                                                    \
    const uint32_t offset##IDX = two*(offsets[IDX + 20*J] + (uint32_t)(gr_index_local[IDX + 20*J])); \
    asm("vmovapd (%0), %%xmm1;" : : "r" ( hist_data + offset##IDX ) : /*"%xmm1"*/);                 \
    asm("vaddpd %xmm2, %xmm1, %xmm3;");                                                             \
    asm("vmovapd %%xmm3, (%0);" : : "r" ( hist_data + offset##IDX ) : /*"%xmm3"*/);                 \
/*    __m128 hist1##IDX    = _mm_loadu_ps(data_local_hist + idx_bin##IDX); \
    __m128 newHist1##IDX = _mm_add_ps(adds, hist1##IDX);                 \
    _mm_storeu_ps(data_local_hist + idx_bin##IDX, newHist1##IDX);        \*/


namespace daal {

namespace primitives {

namespace xgboost {

struct Prefetch {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize =
      kPrefetchOffset + kCacheLineSize / 64;

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch::kNoPrefetchSize;


template<typename FPType, bool do_prefetch, typename BinIdxType>
void ExternalKernelsSeqBuildHist(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const BinIdxType* gradient_index, const uint32_t* offsets,
                             FPType* hist_data) {
  const size_t nb = n_features / 20;
//  std::cout << "\nnb: " << nb << "\n";
  const size_t tail_size = n_features - nb*20;
//  asm("vmovapd (%0), %%xmm1;" : : "r" ( hist_data ) ); 

//  std::cout << "\ntail_size: " << tail_size << "\n";
//std::cout << "offsets:\n";
//for(size_t i = 0; i < n_features; i++) {
//  if(i%16 == 0)
//    std::cout << "\n";
//  std::cout << offsets[i] << "   ";
//}
//std::cout << "\n";
//  printf("\nExternal DAAL Kernel!!!!\n");
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = rid[i] * n_features;
    const size_t idx_gh = two * rid[i];
    if (do_prefetch) {
      const size_t icol_start_prefetch = rid[i + Prefetch::kPrefetchOffset] * n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
           j += Prefetch::GetPrefetchStep<BinIdxType>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t j = 0; j < n_features; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
//  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
//                           // 2 FP values: gradient and hessian.
//                           // So we need to multiply each row-index/bin-index by 2
//                           // to work with gradient pairs as a singe row FP array
////    //std::cout << "\nsize >= 4\n";
//  for (size_t i = 0; i < size; ++i) {
//    const size_t icol_start = rid[i] * n_features;
//    const size_t idx_gh = two * rid[i];
//
//
//    if (do_prefetch) {
//      const size_t icol_start_prefetch = rid[i + Prefetch::kPrefetchOffset] * n_features;
//
//      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
//      for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
//           j += Prefetch::GetPrefetchStep<BinIdxType>()) {
//        PREFETCH_READ_T0(gradient_index + j);
//      }
//    }
//
///*    const BinIdxType* gr_index_local = gradient_index + icol_start;
//    for (size_t j = 0; j < n_features; ++j) {
//      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
//                                      offsets[j]);
//
//      hist_data[idx_bin]   += pgh[idx_gh];
//      hist_data[idx_bin+1] += pgh[idx_gh+1];
//    }*/
////     std::cout << "\n!!!0starti: " << i << "\n";
////     std::cout << "\n!offsets[0]: " << offsets[0] << "\n";
////     std::cout << "\n!offsets[0]: " << offsets[15] << "\n";
////
////     __m512i offsetx = _mm512_loadu_epi32(offsets);
////     std::cout << "\n!!!1starti: " << i << "\n";
//     const double dpgh[] = {pgh[idx_gh], pgh[idx_gh + 1], };
//     asm("vmovapd (%0), %%xmm2;" : : "r" ( dpgh ) : /*"%xmm2"*/);
////     std::cout << "\n!!!______________starti: " << i << "\n";
//
//     const BinIdxType* gr_index_local = gradient_index + icol_start;
//     for(size_t ib = 0; ib < nb; ++ib) {
////      std::cout << "\n1ib: " << ib << "\n";
////      std::cout << "\nvoffset\n";
//       //__m512i voffset = _mm512_loadu_si512(offsets + 16*ib);
//
///*register uint32_t voffset[16];
//voffset[0] = *(offsets + 16*ib + 0);
//voffset[1 ] = *(offsets + 16*ib + 1 );
//voffset[2 ] = *(offsets + 16*ib + 2 );
//voffset[3 ] = *(offsets + 16*ib + 3 );
//voffset[4 ] = *(offsets + 16*ib + 4 );
//voffset[5 ] = *(offsets + 16*ib + 5 );
//voffset[6 ] = *(offsets + 16*ib + 6 );
//voffset[7 ] = *(offsets + 16*ib + 7 );
//voffset[8 ] = *(offsets + 16*ib + 8 );
//voffset[9 ] = *(offsets + 16*ib + 9 );
//voffset[10] = *(offsets + 16*ib + 10);
//voffset[11] = *(offsets + 16*ib + 11);
//voffset[12] = *(offsets + 16*ib + 12);
//voffset[13] = *(offsets + 16*ib + 13);
//voffset[14] = *(offsets + 16*ib + 14);
//voffset[15] = *(offsets + 16*ib + 15);
//*/
////      std::cout << "\n3ib: " << ib << "\n";
//
//      //std::cout << "\noffset is loaded: " << ((uint32_t*)&voffset)[0] << "\n";
//       UNR(0, ib);
////      std::cout << "\nUNR 0\n";
//       UNR(1, ib);
////      std::cout << "\nUNR 1\n";
//       UNR(2, ib);
////      std::cout << "\nUNR 2\n";
//       UNR(3, ib);
////      std::cout << "\nUNR 3\n";
//       UNR(4, ib);
////      std::cout << "\nUNR 4\n";
//       UNR(5, ib);
////      std::cout << "\nUNR 5\n";
//       UNR(6, ib);
////      std::cout << "\nUNR 6\n";
//       UNR(7, ib);
////      std::cout << "\nUNR 7\n";
//       UNR(8, ib);
////      std::cout << "\nUNR 8\n";
//       UNR(9, ib);
////      std::cout << "\nUNR 9\n";
//       UNR(10, ib);
////      std::cout << "\nUNR 10\n";
//       UNR(11, ib);
////      std::cout << "\nUNR 11\n";
//       UNR(12, ib);
////      std::cout << "\nUNR 12\n";
//       UNR(13, ib);
////      std::cout << "\nUNR 13\n";
//       UNR(14, ib);
////      std::cout << "\nUNR 14\n";
//       UNR(15, ib);
//
//       UNR(16, ib);
////      std::cout << "\nUNR 0\n";
//       UNR(17, ib);
////      std::cout << "\nUNR 1\n";
//       UNR(18, ib);
////      std::cout << "\nUNR 2\n";
//       UNR(19, ib);
////      std::cout << "\nUNR 3\n";
///*       UNR(20, ib);
////      std::cout << "\nUNR 4\n";
//       UNR(21, ib);
////      std::cout << "\nUNR 5\n";
//       UNR(22, ib);
////      std::cout << "\nUNR 6\n";
//       UNR(23, ib);
////      std::cout << "\nUNR 7\n";
//       UNR(24, ib);
////      std::cout << "\nUNR 8\n";
//       UNR(25, ib);
////      std::cout << "\nUNR 9\n";
//       UNR(26, ib);
////      std::cout << "\nUNR 10\n";
//       UNR(27, ib);
////      std::cout << "\nUNR 11\n";
//       UNR(28, ib);
////      std::cout << "\nUNR 12\n";
//       UNR(29, ib);
////      std::cout << "\nUNR 13\n";
//       UNR(30, ib);
////      std::cout << "\nUNR 14\n";
//       UNR(31, ib);
//
//
//       UNR(32, ib);
////      std::cout << "\nUNR 0\n";
//       UNR(33, ib);
////      std::cout << "\nUNR 1\n";
//       UNR(34, ib);
////      std::cout << "\nUNR 2\n";
//       UNR(35, ib);
////      std::cout << "\nUNR 3\n";
//       UNR(36, ib);
////      std::cout << "\nUNR 4\n";
//       UNR(37, ib);
////      std::cout << "\nUNR 5\n";
//       UNR(38, ib);
////      std::cout << "\nUNR 6\n";
//       UNR(39, ib);
////      std::cout << "\nUNR 7\n";
//       UNR(40, ib);
////      std::cout << "\nUNR 8\n";
//       UNR(41, ib);
////      std::cout << "\nUNR 9\n";
//       UNR(42, ib);
////      std::cout << "\nUNR 10\n";
//       UNR(43, ib);
////      std::cout << "\nUNR 11\n";
//       UNR(44, ib);
////      std::cout << "\nUNR 12\n";
//       UNR(45, ib);
////      std::cout << "\nUNR 13\n";
//       UNR(46, ib);
////      std::cout << "\nUNR 14\n";
//       UNR(47, ib);
//
//       UNR(48, ib);
////      std::cout << "\nUNR 0\n";
//       UNR(49, ib);
////      std::cout << "\nUNR 1\n";
//       UNR(50, ib);
////      std::cout << "\nUNR 2\n";
//       UNR(51, ib);
////      std::cout << "\nUNR 3\n";
//       UNR(52, ib);
////      std::cout << "\nUNR 4\n";
//       UNR(53, ib);
////      std::cout << "\nUNR 5\n";
//       UNR(54, ib);
////      std::cout << "\nUNR 6\n";
//       UNR(55, ib);
////      std::cout << "\nUNR 7\n";
//       UNR(56, ib);
////      std::cout << "\nUNR 8\n";
//       UNR(57, ib);
////      std::cout << "\nUNR 9\n";
//       UNR(58, ib);
////      std::cout << "\nUNR 10\n";
//       UNR(59, ib);
////      std::cout << "\nUNR 11\n";
//       UNR(60, ib);
////      std::cout << "\nUNR 12\n";
//       UNR(61, ib);
////      std::cout << "\nUNR 13\n";
//       UNR(62, ib);
////      std::cout << "\nUNR 14\n";
//       UNR(63, ib);
//
//*///      std::cout << "\nUNR 15\n";
//     }
//if(tail_size >= 10)
//{
//       UNR(0, nb);
////      std::cout << "\nUNR 0\n";
//       UNR(1, nb);
////      std::cout << "\nUNR 1\n";
//       UNR(2, nb);
////      std::cout << "\nUNR 2\n";
//       UNR(3, nb);
////      std::cout << "\nUNR 3\n";
//       UNR(4, nb);
////      std::cout << "\nUNR 4\n";
//       UNR(5, nb);
////      std::cout << "\nUNR 5\n";
//       UNR(6, nb);
////      std::cout << "\nUNR 6\n";
//       UNR(7, nb);
////      std::cout << "\nUNR 7\n";
//       UNR(8, nb);
////      std::cout << "\nUNR 8\n";
//       UNR(9, nb);
////      std::cout << "\nUNR 9\n";
///*       UNR(10, nb);
////      std::cout << "\nUNR 10\n";
//       UNR(11, nb);
////      std::cout << "\nUNR 11\n";
//       UNR(12, nb);
////      std::cout << "\nUNR 12\n";
//       UNR(13, nb);
////      std::cout << "\nUNR 13\n";
//       UNR(14, nb);
////      std::cout << "\nUNR 14\n";
//       UNR(15, nb);
//       UNR(16, nb);
////      std::cout << "\nUNR 0\n";
//       UNR(17, nb);
////      std::cout << "\nUNR 1\n";
//       UNR(18, nb);
////      std::cout << "\nUNR 2\n";
//       UNR(19, nb);
////      std::cout << "\nUNR 3\n";
///*       UNR(20, nb);
////      std::cout << "\nUNR 4\n";
//       UNR(21, nb);
////      std::cout << "\nUNR 5\n";
//       UNR(22, nb);
////      std::cout << "\nUNR 6\n";
//       UNR(23, nb);
////      std::cout << "\nUNR 7\n";
//       UNR(24, nb);
////      std::cout << "\nUNR 8\n";
//       UNR(25, nb);
////      std::cout << "\nUNR 9\n";
//       UNR(26, nb);
////      std::cout << "\nUNR 10\n";
//       UNR(27, nb);
////      std::cout << "\nUNR 11\n";
//       UNR(28, nb);
////      std::cout << "\nUNR 12\n";
//       UNR(29, nb);
////      std::cout << "\nUNR 13\n";
//       UNR(30, nb);
////      std::cout << "\nUNR 14\n";
//       UNR(31, nb);
//       UNR(32, nb);
//       UNR(33, nb);
//       UNR(34, nb);
//       UNR(35, nb);
//       UNR(36, nb);
//       UNR(37, nb);
//       UNR(38, nb);
//       UNR(39, nb);
//       UNR(40, nb);
//       UNR(41, nb);
//       UNR(42, nb);
//       UNR(43, nb);
//       UNR(44, nb);
//       UNR(45, nb);
//       UNR(46, nb);
//       UNR(47, nb);
//*/
//     for(size_t j = n_features - tail_size + 10;  j < n_features; ++j) {
//       const size_t offset = (offsets[j]);
//       const size_t gr_index = (size_t)(gr_index_local[j]);
//       asm("vmovapd (%0), %%xmm1;" : : "r" ( hist_data + two*(gr_index + offset) ) : /*"%xmm1"*/);
//       asm("vaddpd %xmm2, %xmm1, %xmm3;");
//       asm("vmovapd %%xmm3, (%0);" : : "r" ( hist_data + two*(gr_index + offset) ) : /*"%xmm3"*/);
//     }
//
//}
//else {
//
//     for(size_t j = n_features - tail_size;  j < n_features; ++j) {
//       const size_t offset = (offsets[j]);
//       const size_t gr_index = (size_t)(gr_index_local[j]);
//       asm("vmovapd (%0), %%xmm1;" : : "r" ( hist_data + two*(gr_index + offset) ) : /*"%xmm1"*/);
//       asm("vaddpd %xmm2, %xmm1, %xmm3;");
//       asm("vmovapd %%xmm3, (%0);" : : "r" ( hist_data + two*(gr_index + offset) ) : /*"%xmm3"*/);
//     }
//}
////     std::cout << "\nendi: " << i << "\n";
//
//}

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

template
void XGBoostExternalKernels::X_SeqBuildHist<double, true, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<double, false, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, true, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, false, uint8_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint8_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
//////////////////////////
template
void XGBoostExternalKernels::X_SeqBuildHist<double, true, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<double, false, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, true, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, false, uint16_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint16_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
///////////////////////////
template
void XGBoostExternalKernels::X_SeqBuildHist<double, true, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<double, false, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             double* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, true, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
template
void XGBoostExternalKernels::X_SeqBuildHist<float, false, uint32_t>(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const uint32_t* gradient_index, const uint32_t* offsets,
                             float* hist_data);
