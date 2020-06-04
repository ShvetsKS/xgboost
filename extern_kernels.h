#include <stdint.h>
#include <stdio.h>
#include "src/common/kernels.h"

class XGBoostExternalKernels: public Kernel<XGBoostExternalKernels> {
public:
    template<typename FPType, bool do_prefetch, typename BinIdxType>
    void X_SeqBuildHist(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                        const BinIdxType* gradient_index, const uint32_t* offsets,
                        FPType* hist_data);
};

/*template<typename FPType, bool do_prefetch, typename BinIdxType>
void ExternalBuildHistDenseKernel(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                             const BinIdxType* gradient_index, const uint32_t* offsets,
                             FPType* hist_data);*/