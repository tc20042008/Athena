// #include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass_epilogue/thread/linear_combination_unary.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "matmul.h"
#include "epilogue_op.h"

//template <typename TShape, typename WShape, typename IShape, int NumStages>
cutlass::Status CutlassMatmulAddUnary(const GemmEpilogueParams& params) {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
  using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
  using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

  using TShape = cutlass::gemm::GemmShape<256, 128, 32>;// threadblock tile
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;  // warp tile
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;   // MMA Op tile
  constexpr int NumStages = 3;

  // how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

#if 0
  // Epilogue operation as LinearCombinationRelu:
  //  d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij)
  //
  // - sum_k(a_ik * b_kj), the intermedia result of matrix product, A * B
  // - c_ij, the bias 
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias
#endif
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationUnary<
      ap::ScaleFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

  using GemmFunc = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::RowMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TShape,
      WShape,
      IShape,
      EpilogueOutputOp,
      SwizzleThreadBlock,
      NumStages,
      8,         // AlignA
      8,         // AlignB
      cutlass::arch::OpMultiplyAdd    // Operation performed by GEMM
  >;

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  ElementInputA *input = reinterpret_cast<ElementInputA *>(params.input);
  ElementInputB *weight = reinterpret_cast<ElementInputB *>(params.weight);
  ElementOutput *bias = reinterpret_cast<ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  /// Only available in RRR format
  const int64_t batch_stride_C = params.is_C_bias ? problem_size.n() : problem_size.m() * problem_size.n();
  const int64_t ldc_bias = params.is_C_bias ? 0 : static_cast<int64_t>(params.n);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = static_cast<ElementComputeEpilogue>(1);

  ap::ScaleFunctor<ElementComputeEpilogue>::Arguments unary_args{0.1};

  typename GemmFunc::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,                         // <- problem size of matrix multiplication
      1,                                    // <- batch_count, k-dimension split factor
      {alpha, beta, unary_args},            // <- epilogue params, alpha, beta and other arguments
      input,                                // <- input, ptr_A
      weight,                               // <- input, ptr_B
      bias,                                 // <- input, ptr_C or bias
      output,                               // <- output, ptr_D
      problem_size.m() * problem_size.k(),  // <- batch_stride_A
      problem_size.n() * problem_size.k(),  // <- batch_stride_B
      batch_stride_C,                       // <- batch_stride_C
      problem_size.m() * problem_size.n(),  // <- batch_stride_D
      problem_size.k(),                     // <- lda
      problem_size.n(),                     // <- ldb
      ldc_bias,                             // <- ldc
      problem_size.n()                      // <- ldd
  };

  GemmFunc device_gemm;
  // size_t workspace_size = DeviceKernelName::get_workspace_size(arguments);
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = device_gemm.can_implement(arguments);
  CHECK_CUTLASS(status);

  // status = device_gemm.initialize(arguments, workspace.get());
  status = device_gemm.initialize(arguments, params.workspace);
  CHECK_CUTLASS(status);

  status = device_gemm();
  CHECK_CUTLASS(status);
  return status;
}

cutlass::Status CutlassMatmulAddBinary(const GemmBroadcastEpilogueParams& params) {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutputC = cutlass::half_t;
  using ElementOutputZ = cutlass::half_t;
  using ElementOutputT = cutlass::half_t;

  using TShape = cutlass::gemm::GemmShape<256, 128, 32>;// threadblock tile
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;  // warp tile
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;   // MMA Op tile
  constexpr int NumStages = 3;

  // how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

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
  // using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
  //   ElementOutputC,
  //   ElementAccumulator,
  //   ElementAccumulator,
  //   ElementAccumulator,
  //   128 / cutlass::sizeof_bits<ElementOutput>::value,
  //   cutlass::epilogue::thread::Identity,
  //   cutlass::multiplies,
  //   cutlass::epilogue::thread::Identity>;

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
      TShape,
      WShape,
      IShape,
      EpilogueOutputOp,
      SwizzleThreadBlock,
      NumStages,
      8,         // AlignA
      8,         // AlignB
      cutlass::arch::OpMultiplyAdd    // Operation performed by GEMM
  >;

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  ElementInputA *input = reinterpret_cast<ElementInputA *>(params.input);
  ElementInputB *weight = reinterpret_cast<ElementInputB *>(params.weight);
  ElementOutputC *bias = reinterpret_cast<ElementOutputC *>(params.bias);
  ElementOutputZ *output = reinterpret_cast<ElementOutputZ *>(params.output);
  ElementOutputC *broadcast = reinterpret_cast<ElementOutputC *>(params.broadcast);
  ElementOutputT *broadcast_out = reinterpret_cast<ElementOutputT *>(params.broadcast_out);

  /// Only available in RRR format
  const int64_t batch_stride_C = params.is_C_bias ? problem_size.n() : problem_size.m() * problem_size.n();
  const int64_t ldc_bias = params.is_C_bias ? 0 : static_cast<int64_t>(params.n);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = static_cast<ElementComputeEpilogue>(1);

  typename GemmFunc::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,                         // <- problem size of matrix multiplication
      1,                                    // <- batch_count, k-dimension split factor
      {alpha, beta},                        // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A, A, shape={M, K}
      weight,                               // <- input, ptr_B, B, shape={K, N}
      bias,                                 // <- input, ptr_C, shape={M, N} or {1, N}
      output,                               // <- output, ptr_D, Z, shape={M, N}
      broadcast,                            // <- input, ptr_Vector, Broadcast, shape={M, 1}
      broadcast_out,                        // <- output, ptr_Tensor, T
      problem_size.m() * problem_size.k(),  // <- batch_stride_A
      problem_size.n() * problem_size.k(),  // <- batch_stride_B
      batch_stride_C,                       // <- batch_stride_C
      problem_size.m() * problem_size.n(),  // <- batch_stride_D
      problem_size.m(),                     // <- batch_stride_Vector, need broadcast
      problem_size.m() * problem_size.n(),  // <- batch_stride_Tensor
      problem_size.k(),                     // <- lda
      problem_size.n(),                     // <- ldb
      ldc_bias,                             // <- ldc
      problem_size.n(),                     // <- ldd
      0,                                    // <- ldr, must be zero
      problem_size.n()                      // <- ldt
  }; 


  GemmFunc device_gemm;

  // size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = device_gemm.can_implement(arguments);
  CHECK_CUTLASS(status);

  // status = device_gemm.initialize(arguments, workspace.get());
  status = device_gemm.initialize(arguments, params.workspace);
  CHECK_CUTLASS(status);

  //
  // Run the GEMM
  //
  status = device_gemm();
  CHECK_CUTLASS(status);
  return status;
}
