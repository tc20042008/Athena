#pragma once

#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass_epilogue/thread/linear_combination_unary.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "matmul.h"

#ifdef TUNE_TILE_SHAPE
#include "tile_shape.h"
#else
#include "default_tile_shape.h"
#endif

namespace ap {

// Operation performed by GEMM
template <typename ElementT>
struct GemmOperation {
  using Type = cutlass::arch::OpMultiplyAdd;
};

template <>
struct GemmOperation<float> {
  using Type = cutlass::arch::OpMultiplyAddFastF32;
};


cutlass::gemm::GemmUniversalMode GetGemmMode(int batch_count) {
  return batch_count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm;
}

void* GetWorkspace(size_t workspace_size) {
  static cutlass::device_memory::allocation<uint8_t> workspace;
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }
  return workspace.get();
}

template <typename ElementT,
          typename ElementComputeT,
          bool TransposeA = false,
          bool TransposeB = false>
void CutlassMatmulAdd(const GemmEpilogueParams& params) {
  using ElementAccumulator = ElementComputeT;         // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = ElementT;                     // <- data type of elements in input matrix A
  using ElementInputB = ElementT;                     // <- data type of elements in input matrix B
  using ElementOutput = ElementT;                     // <- data type of elements in output matrix D


  // Epilogue operation as LinearCombination:
  //  alpha * accumulator + beta * source
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

  using GemmFunc = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      typename MatrixLayout<TransposeA>::Type,
      ElementInputB,
      typename MatrixLayout<TransposeB>::Type,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfig<ElementT>::TShape,
      typename GemmTuningConfig<ElementT>::WShape,
      typename GemmTuningConfig<ElementT>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfig<ElementT>::SwizzleThreadBlock,             // how threadblocks are scheduled on GPU
      GemmTuningConfig<ElementT>::NumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type            // Operation performed by GEMM
  >;

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  GemmShapeArguments<TransposeA, TransposeB> gemm_shape_args(problem_size, params.is_B_weight, params.is_C_bias);

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutput *bias = reinterpret_cast<const ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1) : static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta},                        // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A
      weight,                               // <- input, ptr_B
      bias,                                 // <- input, ptr_C or bias
      output,                               // <- output, ptr_D
      gemm_shape_args.batch_stride_A,
      gemm_shape_args.batch_stride_B,
      gemm_shape_args.batch_stride_C,
      gemm_shape_args.batch_stride_D,
      gemm_shape_args.lda,
      gemm_shape_args.ldb,
      gemm_shape_args.ldc_bias,
      gemm_shape_args.ldd
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm.run(params.stream));
}


template <typename ElementT,
          typename ElementComputeT,
          template<typename T> class UnaryFunctor,
          bool TransposeA = false,
          bool TransposeB = false>
//           typename TShape = cutlass::gemm::GemmShape<16, 64, 64>,
//           typename WShape = cutlass::gemm::GemmShape<16, 32, 64>,
//           typename IShape = cutlass::gemm::GemmShape<16, 8, 16>,
//           int NumStages = 5>
void CutlassMatmulAddUnary(const GemmEpilogueParams& params, const typename UnaryFunctor<ElementComputeT>::Arguments& unary_args) {
  using ElementAccumulator = ElementComputeT;         // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = ElementT;                     // <- data type of elements in input matrix A
  using ElementInputB = ElementT;                     // <- data type of elements in input matrix B
  using ElementOutput = ElementT;                     // <- data type of elements in output matrix D

  // Epilogue operation as LinearCombinationUnary:
  //  d_ij = unary_op(alpha * sum_k(a_ik * b_kj) + c_ij)
  //
  // - sum_k(a_ik * b_kj), the intermedia result of matrix product, A * B
  // - c_ij, the bias
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationUnary<
      UnaryFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

  using GemmFunc = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      typename MatrixLayout<TransposeA>::Type,
      ElementInputB,
      typename MatrixLayout<TransposeB>::Type,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfig<ElementT>::TShape,
      typename GemmTuningConfig<ElementT>::WShape,
      typename GemmTuningConfig<ElementT>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfig<ElementT>::SwizzleThreadBlock,
      GemmTuningConfig<ElementT>::NumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type  // Operation performed by GEMM
  >;

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  GemmShapeArguments<TransposeA, TransposeB> gemm_shape_args(problem_size, params.is_B_weight, params.is_C_bias);

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutput *bias = reinterpret_cast<const ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1) : static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta, unary_args},            // <- epilogue params, alpha, beta and other arguments
      input,                                // <- input, ptr_A
      weight,                               // <- input, ptr_B
      bias,                                 // <- input, ptr_C or bias
      output,                               // <- output, ptr_D
      gemm_shape_args.batch_stride_A,
      gemm_shape_args.batch_stride_B,
      gemm_shape_args.batch_stride_C,
      gemm_shape_args.batch_stride_D,
      gemm_shape_args.lda,
      gemm_shape_args.ldb,
      gemm_shape_args.ldc_bias,
      gemm_shape_args.ldd
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm.run(params.stream));
}

template <typename ElementT, typename ElementComputeT>
void CutlassMatmulAddBinary(const GemmBroadcastEpilogueParams& params) {
  using ElementAccumulator = ElementComputeT;         // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = ElementT;
  using ElementInputB = ElementT;
  using ElementOutputC = ElementT;
  using ElementOutputZ = ElementT;
  using ElementOutputT = ElementT;

  // Epilogue operation as LinearCombinationBiasElementwise:
  //  Y = GEMM(AB, C)
  //  T[i, j] = BinaryOp(Y[i, j], Broadcast[i])
  //  Z[i, j] = Elementwise(T[i, j])
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    ElementOutputC,
    ElementAccumulator,
    ElementComputeEpilogue,
    ElementOutputZ,
    ElementOutputT,
    128 / cutlass::sizeof_bits<ElementOutputC>::value,
    ap::IdentityFunctor<ElementComputeEpilogue>
  >;

  // Epilogue operation as LinearCombinationResidualBlock:
  //  Y = GEMM(AB, C1)
  //  UnaryOp(BinaryOp2(BinaryOp1(ActivationOp(Y), residual1), residual2))
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
  //   ElementOutput,                        // Element type for output matrix
  //   ElementAccumulator,                   // Element type from internal accumulation
  //   ElementCompute,                       // Element type from internal accumulation
  //   ElementC,                             // Element type for C1/C2/D matrix operands
  //   AlignmentC,                           // Memory access granularity of C and D matrix in units of elements
  //   cutlass::epilogue::thread::Identity,  // Activation
  //   cutlass::plus,                        // Binary operation 1
  //   cutlass::epilogue::thread::Identity,  // Unary operation
  //   cutlass::plus                         // Binary operation 2
  //   >;

  using GemmFunc = cutlass::gemm::device::GemmUniversalWithBroadcast<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::RowMajor,
      ElementOutputC,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfig<ElementT>::TShape,
      typename GemmTuningConfig<ElementT>::WShape,
      typename GemmTuningConfig<ElementT>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfig<ElementT>::SwizzleThreadBlock,
      GemmTuningConfig<ElementT>::NumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type  // Operation performed by GEMM
  >;

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  GemmShapeArguments<false, false> gemm_shape_args(problem_size, params.is_B_weight, params.is_C_bias);

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutputC *bias = reinterpret_cast<const ElementOutputC *>(params.bias);
  ElementOutputZ *output = reinterpret_cast<ElementOutputZ *>(params.output);
  ElementOutputC *broadcast = reinterpret_cast<ElementOutputC *>(params.broadcast);
  ElementOutputT *broadcast_out = reinterpret_cast<ElementOutputT *>(params.broadcast_out);

  const int64_t batch_stride_Broadcast = params.need_broadcast ? problem_size.m() : problem_size.m() * problem_size.n();
  const int64_t ldr_broadcast = params.need_broadcast ? 0 : problem_size.n();

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = static_cast<ElementComputeEpilogue>(1);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta},                        // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A, A, shape={M, K}
      weight,                               // <- input, ptr_B, B, shape={K, N}
      bias,                                 // <- input, ptr_C, shape={M, N} or {1, N}
      output,                               // <- output, ptr_D, Z, shape={M, N}
      broadcast,                            // <- input, ptr_Vector, Broadcast, shape={M, 1}
      broadcast_out,                        // <- output, ptr_Tensor, T
      gemm_shape_args.batch_stride_A,
      gemm_shape_args.batch_stride_B,
      gemm_shape_args.batch_stride_C,
      gemm_shape_args.batch_stride_D,
      batch_stride_Broadcast,               // <- batch_stride_Vector, need broadcast
      problem_size.m() * problem_size.n(),  // <- batch_stride_Tensor
      gemm_shape_args.lda,
      gemm_shape_args.ldb,
      gemm_shape_args.ldc_bias,
      gemm_shape_args.ldd,
      ldr_broadcast,                        // <- ldr, must be zero
      problem_size.n()                      // <- ldt
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm(params.stream));
}

}
