#pragma once

namespace ap {

template <typename ElementT>
struct GemmTuningConfig {
  using TShape = cutlass::gemm::GemmShape<16,64,64>;
  using WShape = cutlass::gemm::GemmShape<16,32,64>;
  using IShape = cutlass::gemm::GemmShape<16,8,16>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  static constexpr int NumStages = 5;
};

}
