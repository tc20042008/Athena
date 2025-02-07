#pragma once

#include "cutlass/gemm_coord.h"

namespace ap {

template <typename ElementT>
struct GemmTuningConfig {
  using TShape = cutlass::gemm::GemmShape<256, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr int NumStages = 3;
};

template <>
struct GemmTuningConfig<float> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr int NumStages = 4;
};

};
