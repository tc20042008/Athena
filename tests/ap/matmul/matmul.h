#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
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

namespace ap {

struct GemmEpilogueParams {
  int batch_count;
  int m;
  int n;
  int k;

  const void *input;
  const void *weight;
  const void *bias;
  void *output;

  bool is_B_weight{true};
  bool is_C_bias{true};
  cudaStream_t stream;
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

template <bool TransposeA, bool TransposeB>
struct GemmShapeArguments {
  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
  int64_t lda;
  int64_t ldb;
  int64_t ldc_bias;
  int64_t ldd;

  GemmShapeArguments(cutlass::gemm::GemmCoord problem_size, bool is_B_weight, bool is_C_bias) {
    batch_stride_A = problem_size.m() * problem_size.k();
    batch_stride_B = is_B_weight ? 0 : problem_size.n() * problem_size.k();
    batch_stride_D = problem_size.m() * problem_size.n();

    /// Only available in RRR format
    batch_stride_C = is_C_bias ? 0 : problem_size.m() * problem_size.n();

    lda = TransposeA ? problem_size.m() : problem_size.k();
    ldb = TransposeB ? problem_size.k() : problem_size.n();
    ldc_bias = is_C_bias ? 0 : problem_size.n();
    ldd = problem_size.n();
  }
};

}
