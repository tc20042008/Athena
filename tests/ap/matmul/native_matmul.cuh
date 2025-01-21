// Modified codes from https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

#pragma once

#include <iostream>
#include "matmul.h"

namespace ap {

template <typename T,
          size_t BlockTileSizeX,
          size_t BlockTileSizeY,
          size_t BlockTileSizeK,
          size_t NumThreads,
          size_t BlockTileSkewSizeX = 0U,
          size_t BlockTileSkewSizeK = 0U>
__device__ void LoadToSharedMemory(T const* A, size_t lda,
                                   T const* B, size_t ldb,
                                   T A_thread_block_tile[BlockTileSizeY][BlockTileSizeK + BlockTileSkewSizeK],
                                   T B_thread_block_tile[BlockTileSizeK][BlockTileSizeX + BlockTileSkewSizeX],
                                   size_t thread_block_tile_idx,
                                   size_t thread_linear_idx,
                                   size_t m, size_t n,
                                   size_t k) {
  // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx = 0U;
       load_idx < (BlockTileSizeY * BlockTileSizeK + NumThreads - 1U) / NumThreads;
       ++load_idx) {
    size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NumThreads) / BlockTileSizeK;
    size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NumThreads) % BlockTileSizeK;
    size_t const A_row_idx = blockIdx.y * BlockTileSizeY + A_thread_block_tile_row_idx;
    size_t const A_col_idx = thread_block_tile_idx * BlockTileSizeK + A_thread_block_tile_col_idx;

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (A_row_idx < m && A_col_idx < k) {
      val = A[A_row_idx * lda + A_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BlockTileSizeK * BlockTileSizeY % NumThreads == 0U);
    // if (A_thread_block_tile_row_idx < BlockTileSizeY &&
    //     A_thread_block_tile_col_idx < BlockTileSizeK) {
    //     A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    // }
    A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
  }

  // Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx = 0U;
     load_idx < (BlockTileSizeK * BlockTileSizeX + NumThreads - 1U) / NumThreads;
     ++load_idx) {
    size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NumThreads) / BlockTileSizeX;
    size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NumThreads) % BlockTileSizeX;
    size_t const B_row_idx = thread_block_tile_idx * BlockTileSizeK + B_thread_block_tile_row_idx;
    size_t const B_col_idx = blockIdx.x * BlockTileSizeX + B_thread_block_tile_col_idx;

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (B_row_idx < k && B_col_idx < n) {
      val = B[B_row_idx * ldb + B_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BlockTileSizeX * BlockTileSizeK % NumThreads == 0U);
    // if (B_thread_block_tile_row_idx < BlockTileSizeK &&
    //     B_thread_block_tile_col_idx < BlockTileSizeX) {
    //     B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    // }
    B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
  }
}


template <typename T,
          size_t BlockTileSizeX,
          size_t BlockTileSizeY,
          size_t BlockTileSizeK,
          size_t ThreadTileSizeX,
          size_t ThreadTileSizeY,
          typename EpilogueFunctor>
__global__ void GemmKernel(size_t m, size_t n, size_t k, T const* A,
                           size_t lda, T const* B, size_t ldb, T* C,
                           size_t ldc, typename EpilogueFunctor::Arguments epilogue_args) {
  // Avoid using blockDim.x * blockDim.y as the number of threads per block.
  // Because it is a runtime constant and the compiler cannot optimize the
  // loop unrolling based on that.
  // Use a compile time constant instead.
  constexpr size_t NumThreads = BlockTileSizeX * BlockTileSizeY / (ThreadTileSizeX * ThreadTileSizeY);
  size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T A_thread_block_tile[BlockTileSizeY][BlockTileSizeK];
  __shared__ T B_thread_block_tile[BlockTileSizeK][BlockTileSizeX];

  size_t const num_thread_block_tiles = (k + BlockTileSizeK - 1) / BlockTileSizeK;

  // Each thread in the block processes BlockTileSizeY output values.
  // Specifically, these values corresponds to
  // C[blockIdx.y * BlockTileSizeY + threadIdx.x / BlockTileSizeX *
  // ThreadTileSizeY : blockIdx.y * BlockTileSizeY + (threadIdx.x /
  // BlockTileSizeX + 1) * ThreadTileSizeY][blockIdx.x *
  // BlockTileSizeX + threadIdx.x % BLOCK_TILE_SIZE_X *
  // ThreadTileSizeX : blockIdx.x * BlockTileSizeX + (threadIdx.x %
  // BlockTileSizeX + 1) * ThreadTileSizeX]
  T C_thread_results[ThreadTileSizeY][ThreadTileSizeX] = {static_cast<T>(0)};
  // A_vals is cached in the register.
  T A_vals[ThreadTileSizeY] = {static_cast<T>(0)};
  // B_vals is cached in the register.
  T B_vals[ThreadTileSizeX] = {static_cast<T>(0)};

  for (size_t thread_block_tile_idx = 0U;
       thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
    LoadToSharedMemory<T, BlockTileSizeX, BlockTileSizeY, BlockTileSizeK, NumThreads>(
        A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
        thread_block_tile_idx, thread_linear_idx, m, n, k);
    __syncthreads();

#pragma unroll
    for (size_t k_i = 0U; k_i < BlockTileSizeK; ++k_i) {
      size_t const A_thread_block_tile_row_idx =
          thread_linear_idx / (BlockTileSizeX / ThreadTileSizeX) * ThreadTileSizeY;
      size_t const A_thread_block_tile_col_idx = k_i;

#pragma unroll
      for (size_t thread_tile_row_idx = 0U;
           thread_tile_row_idx < ThreadTileSizeY; ++thread_tile_row_idx) {
        // There will be shared memory bank conflicts accessing the
        // values from A_thread_block_tile. We can do it better by
        // transposing the A_thread_block_tile when we load the data
        // from DRAM.
        A_vals[thread_tile_row_idx] =
            A_thread_block_tile[A_thread_block_tile_row_idx + thread_tile_row_idx][A_thread_block_tile_col_idx];
      }

      size_t const B_thread_block_tile_row_idx = k_i;
      size_t const B_thread_block_tile_col_idx =
          thread_linear_idx % (BlockTileSizeX / ThreadTileSizeX) * ThreadTileSizeX;
#pragma unroll
      for (size_t thread_tile_col_idx = 0U;
           thread_tile_col_idx < ThreadTileSizeX; ++thread_tile_col_idx) {
        B_vals[thread_tile_col_idx] =
            B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx];
      }

      for (size_t thread_tile_row_idx = 0U;
           thread_tile_row_idx < ThreadTileSizeY; ++thread_tile_row_idx) {
        for (size_t thread_tile_col_idx = 0U;
             thread_tile_col_idx < ThreadTileSizeX; ++thread_tile_col_idx) {
          C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
              A_vals[thread_tile_row_idx] * B_vals[thread_tile_col_idx];
        }
      }
    }
    __syncthreads();
  }

  EpilogueFunctor epilogue;

  // Write the results to DRAM.
  for (size_t thread_tile_row_idx = 0U; thread_tile_row_idx < ThreadTileSizeY; ++thread_tile_row_idx) {
    for (size_t thread_tile_col_idx = 0U; thread_tile_col_idx < ThreadTileSizeX; ++thread_tile_col_idx) {
      size_t const C_row_idx =
          blockIdx.y * BlockTileSizeY +
          threadIdx.x / (BlockTileSizeX / ThreadTileSizeX) * ThreadTileSizeY + thread_tile_row_idx;
      size_t const C_col_idx =
          blockIdx.x * BlockTileSizeX +
          threadIdx.x % (BlockTileSizeX / ThreadTileSizeX) * ThreadTileSizeX + thread_tile_col_idx;
      if (C_row_idx < m && C_col_idx < n) {
        T tmp_C = C_thread_results[thread_tile_row_idx][thread_tile_col_idx];
        ap::GemmCoord3d coord{0, C_row_idx, C_col_idx};
        C[C_row_idx * ldc + C_col_idx] = epilogue(tmp_C, epilogue_args, coord);
      }
    }
  }
}

template <typename T, typename EpilogueFunctor>
void NativeMatmulAdd(const ap::GemmEpilogueParams& params, const typename EpilogueFunctor::Arguments& epilogue_args) {
  size_t m = params.m;
  size_t n = params.n;
  size_t k = params.k;

  size_t lda = params.k;
  size_t ldb = params.n;
  size_t ldc = params.n;

  const T* A = reinterpret_cast<const T*>(params.input);
  const T* B = reinterpret_cast<const T*>(params.weight);
  T* C = reinterpret_cast<T*>(params.output);

  cudaStream_t stream = params.stream;

  constexpr unsigned int kBlockTileSizeX = 128U;
  constexpr unsigned int kBlockTileSizeY = 128U;
  constexpr unsigned int kBlockTileSizeK = 16U;

  // Each thread computes ThreadTileSizeX * ThreadTileSizeY values of C.
  constexpr unsigned int kThreadTileSizeX = 8U;
  constexpr unsigned int kThreadTileSizeY = 8U;

  constexpr unsigned int kNumThreadsPerBlock = kBlockTileSizeX * kBlockTileSizeY / (kThreadTileSizeX * kThreadTileSizeY);

  static_assert(kBlockTileSizeX % kThreadTileSizeX == 0U);
  static_assert(kBlockTileSizeY % kThreadTileSizeY == 0U);
  static_assert(kNumThreadsPerBlock % kBlockTileSizeK == 0U);
  static_assert(kNumThreadsPerBlock % kBlockTileSizeX == 0U);
  static_assert(kBlockTileSizeX * kBlockTileSizeK % kNumThreadsPerBlock == 0U);
  static_assert(kBlockTileSizeK * kBlockTileSizeY % kNumThreadsPerBlock == 0U);

  dim3 const block_dim{kNumThreadsPerBlock, 1U, 1U};
  dim3 const grid_dim{
      (static_cast<unsigned int>(n) + kBlockTileSizeX - 1U) / kBlockTileSizeX,
      (static_cast<unsigned int>(m) + kBlockTileSizeY - 1U) / kBlockTileSizeY,
      1U};
  GemmKernel<T, kBlockTileSizeX, kBlockTileSizeY, kBlockTileSizeK, kThreadTileSizeX, kThreadTileSizeY, EpilogueFunctor>
      <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, epilogue_args);
}

}
