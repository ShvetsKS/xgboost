#ifndef XGBOOST_KERNEL_H_
#define XGBOOST_KERNEL_H_

template <class T>
class Kernel {
public:
    template<typename FPType, bool do_prefetch, typename BinIdxType>
    void SeqBuildHist(const size_t size, const size_t n_features, const size_t* rid, const float* pgh,
                      const BinIdxType* gradient_index, const uint32_t* offsets,
                      FPType* hist_data) {
        static_cast<T*>(this)->template X_SeqBuildHist<FPType, do_prefetch, BinIdxType>(size, n_features, rid, pgh, gradient_index, offsets, hist_data);
    }
};

#endif  // XGBOOST_KERNEL_H_