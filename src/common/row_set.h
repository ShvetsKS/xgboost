/*!
 * Copyright 2017 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_ROW_SET_H_
#define XGBOOST_COMMON_ROW_SET_H_

#include <xgboost/data.h>
#include <algorithm>
#include <vector>
#include <utility>

namespace xgboost {
namespace common {

/*! \brief collection of rowset */
class RowSetCollection {
 public:
  /*! \brief data structure to store an instance set, a subset of
   *  rows (instances) associated with a particular node in a decision
   *  tree. */
  struct Elem {
    const uint32_t* begin{nullptr};
    const uint32_t* end{nullptr};
    int node_id{-1};
      // id of node associated with this instance set; -1 means uninitialized
    Elem()
         = default;
    Elem(const uint32_t* begin,
         const uint32_t* end,
         int node_id)
        : begin(begin), end(end), node_id(node_id) {}

    inline uint32_t Size() const {
      return end - begin;
    }
  };
  /* \brief specifies how to split a rowset into two */
  struct Split {
    std::vector<uint32_t> left;
    std::vector<uint32_t> right;
  };

  inline std::vector<Elem>::const_iterator begin() const {  // NOLINT
    return elem_of_each_node_.begin();
  }

  inline std::vector<Elem>::const_iterator end() const {  // NOLINT
    return elem_of_each_node_.end();
  }

  /*! \brief return corresponding element set given the node_id */
  inline const Elem& operator[](unsigned node_id) const {
    const Elem& e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr)
        << "access element that is not in the set";
    return e;
  }

  /*! \brief return corresponding element set given the node_id */
  inline Elem& operator[](unsigned node_id) {
    Elem& e = elem_of_each_node_[node_id];
    return e;
  }

  // clear up things
  inline void Clear() {
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  inline void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0U);

    if (row_indices_.empty()) {  // edge case: empty instance set
      // assign arbitrary address here, to bypass nullptr check
      // (nullptr usually indicates a nonexistent rowset, but we want to
      //  indicate a valid rowset that happens to have zero length and occupies
      //  the whole instance set)
      // this is okay, as BuildHist will compute (end-begin) as the set size
      const uint32_t* begin = reinterpret_cast<uint32_t*>(20);
      const uint32_t* end = begin;
      elem_of_each_node_.emplace_back(Elem(begin, end, 0));
      return;
    }

    const uint32_t* begin = dmlc::BeginPtr(row_indices_);
    const uint32_t* end = dmlc::BeginPtr(row_indices_) + row_indices_.size();
    elem_of_each_node_.emplace_back(Elem(begin, end, 0));
  }
  // split rowset into two
  inline void AddSplit(unsigned node_id,
                       unsigned left_node_id,
                       unsigned right_node_id,
                       uint32_t n_left,
                       uint32_t n_right) {
    const Elem e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr);
    uint32_t* all_begin = dmlc::BeginPtr(row_indices_);
    uint32_t* begin = all_begin + (e.begin - all_begin);

    CHECK_EQ(n_left + n_right, e.Size());
    CHECK_LE(begin + n_left, e.end);
    CHECK_EQ(begin + n_left + n_right, e.end);

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(left_node_id + 1, Elem(nullptr, nullptr, -1));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(right_node_id + 1, Elem(nullptr, nullptr, -1));
    }

    elem_of_each_node_[left_node_id] = Elem(begin, begin + n_left, left_node_id);
    elem_of_each_node_[right_node_id] = Elem(begin + n_left, e.end, right_node_id);
    elem_of_each_node_[node_id] = Elem(nullptr, nullptr, -1);
  }

  // stores the row indexes in the set
  std::vector<uint32_t> row_indices_;

 private:
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};


// The builder is required for samples partition to left and rights children for set of nodes
// Responsible for:
// 1) Effective memory allocation for intermediate results for multi-thread work
// 2) Merging partial results produced by threads into original row set (row_set_collection_)
// BlockSize is template to enable memory alignment easily with C++11 'alignas()' feature
template<uint32_t BlockSize>
class PartitionBuilder {
 public:
  template<typename Func>
  void Init(const uint32_t n_tasks, uint32_t n_nodes, Func funcNTaks) {
    left_right_nodes_sizes_.resize(n_nodes);
    blocks_offsets_.resize(n_nodes+1);

    blocks_offsets_[0] = 0;
    for (uint32_t i = 1; i < n_nodes+1; ++i) {
      blocks_offsets_[i] = blocks_offsets_[i-1] + funcNTaks(i-1);
    }

    if (n_tasks > max_n_tasks_) {
      mem_blocks_.resize(n_tasks);
      max_n_tasks_ = n_tasks;
    }
  }

  uint32_t* GetLeftBuffer(int nid, uint32_t begin, uint32_t end) {
    const uint32_t task_idx = GetTaskIdx(nid, begin);
    CHECK_LE(task_idx, mem_blocks_.size());
    return mem_blocks_[task_idx].left();
  }

  uint32_t* GetRightBuffer(int nid, uint32_t begin, uint32_t end) {
    const uint32_t task_idx = GetTaskIdx(nid, begin);
    CHECK_LE(task_idx, mem_blocks_.size());
    return mem_blocks_[task_idx].right();
  }

  void SetNLeftElems(int nid, uint32_t begin, uint32_t end, uint32_t n_left) {
    uint32_t task_idx = GetTaskIdx(nid, begin);
    CHECK_LE(task_idx, mem_blocks_.size());
    mem_blocks_[task_idx].n_left = n_left;
  }

  void SetNRightElems(int nid, uint32_t begin, uint32_t end, uint32_t n_right) {
    uint32_t task_idx = GetTaskIdx(nid, begin);
    CHECK_LE(task_idx, mem_blocks_.size());
    mem_blocks_[task_idx].n_right = n_right;
  }


  uint32_t GetNLeftElems(int nid) const {
    return left_right_nodes_sizes_[nid].first;
  }

  uint32_t GetNRightElems(int nid) const {
    return left_right_nodes_sizes_[nid].second;
  }

  // Each thread has partial results for some set of tree-nodes
  // The function decides order of merging partial results into final row set
  void CalculateRowOffsets() {
    for (uint32_t i = 0; i < blocks_offsets_.size()-1; ++i) {
      uint32_t n_left = 0;
      for (uint32_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
        mem_blocks_[j].n_offset_left = n_left;
        n_left += mem_blocks_[j].n_left;
      }
      uint32_t n_right = 0;
      for (uint32_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
        mem_blocks_[j].n_offset_right = n_left + n_right;
        n_right += mem_blocks_[j].n_right;
      }
      left_right_nodes_sizes_[i] = {n_left, n_right};
    }
  }

  void MergeToArray(int nid, uint32_t begin, uint32_t* rows_indexes) {
    uint32_t task_idx = GetTaskIdx(nid, begin);

    uint32_t* left_result  = rows_indexes + mem_blocks_[task_idx].n_offset_left;
    uint32_t* right_result = rows_indexes + mem_blocks_[task_idx].n_offset_right;

    const uint32_t* left = mem_blocks_[task_idx].left();
    const uint32_t* right = mem_blocks_[task_idx].right();

    std::copy_n(left, mem_blocks_[task_idx].n_left, left_result);
    std::copy_n(right, mem_blocks_[task_idx].n_right, right_result);
  }

 protected:
  uint32_t GetTaskIdx(int nid, uint32_t begin) {
    return blocks_offsets_[nid] + begin / BlockSize;
  }

  struct BlockInfo{
    uint32_t n_left;
    uint32_t n_right;

    uint32_t n_offset_left;
    uint32_t n_offset_right;

    uint32_t* left() {
      return &left_data_[0];
    }

    uint32_t* right() {
      return &right_data_[0];
    }
   private:
    alignas(128) uint32_t left_data_[BlockSize];
    alignas(128) uint32_t right_data_[BlockSize];
  };
  std::vector<std::pair<uint32_t, uint32_t>> left_right_nodes_sizes_;
  std::vector<uint32_t> blocks_offsets_;
  std::vector<BlockInfo> mem_blocks_;
  uint32_t max_n_tasks_ = 0;
};


}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
