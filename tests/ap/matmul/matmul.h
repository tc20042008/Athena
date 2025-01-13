#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CHECK_CUDA(func)                                                            \
  {                                                                                 \
    cudaError_t err = func;                                                         \
    if (err != cudaSuccess) {                                                       \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", "                       \
                << __FUNCTION__ << "] " << "CUDA error(" << err << "), "            \
                << cudaGetErrorString(err) << " when call " << #func << std::endl;  \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

struct GemmEpilogueParams {
  int batch_count;
  int m;
  int n;
  int k;

  const void *input;
  const void *weight;
  const void *bias;
  void *output;

  cudaStream_t stream;
  bool is_C_bias{true};
  void *workspace{nullptr};
};

struct GemmBroadcastEpilogueParams : GemmEpilogueParams {
  bool need_broadcast;
  void* broadcast;
  void* broadcast_out;
};

template <bool Transposed>
struct MatrixLayout {
  using Type = cutlass::layout::RowMajor;
};

template <>
struct MatrixLayout<true> {
  using Type = cutlass::layout::ColumnMajor;
};
