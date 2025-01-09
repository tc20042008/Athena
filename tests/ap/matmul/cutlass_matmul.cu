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
  ElementInputA *input = (ElementInputA *)(params.input);
  ElementInputB *weight = (ElementInputB *)(params.weight);
  ElementOutput *bias = (ElementOutput *)(params.bias);
  ElementOutput *output = (ElementOutput *)(params.output);

  /// Only available in RRR format
  int64_t batch_stride_C = params.is_vec_bias ? problem_size.n() : batch_stride_C = problem_size.mn().product();

  int64_t lda = static_cast<int64_t>(params.k);
  int64_t ldb = static_cast<int64_t>(params.n);
  int64_t ldc_bias = params.is_vec_bias ? 0 : static_cast<int64_t>(params.n);
  int64_t ldd = static_cast<int64_t>(params.n);

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  ap::ScaleFunctor<ElementComputeEpilogue>::Arguments unary_args{0.1};

  typename GemmFunc::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,                 // <- problem size of matrix multiplication
      1,                            // <- batch_count, k-dimension split factor
      {alpha, beta, unary_args},    // <- epilogue params, alpha, beta and other arguments
      input,                        // <- input, ptr_A
      weight,                       // <- input, ptr_B
      bias,                         // <- input, ptr_C or bias
      output,                       // <- output, ptr_D
      problem_size.mk().product(),  // <- batch_stride_A
      problem_size.nk().product(),  // <- batch_stride_B
      batch_stride_C,               // <- batch_stride_C
      problem_size.mn().product(),  // <- batch_stride_D
      lda,
      ldb,
      ldc_bias,
      ldd
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

// cutlass::Status CutlassMatmulAddBinary(const GemmEpilogueParams& params) {
//   using ElementAccumulator = float;                   // <- data type of accumulator
//   using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
//   using ElementInputA = cutlass::half_t;
//   using ElementInputB = cutlass::half_t;
//   using ElementOutputC = cutlass::half_t;
//   using ElementOutputZ = cutlass::half_t;
//   using ElementOutputT = cutlass::half_t;
// 
//   using TShape = cutlass::gemm::GemmShape<256, 128, 32>;// threadblock tile
//   using WShape = cutlass::gemm::GemmShape<64, 64, 32>;  // warp tile
//   using IShape = cutlass::gemm::GemmShape<16, 8, 16>;   // MMA Op tile
//   constexpr int NumStages = 3;
// 
//   // how threadblocks are scheduled on GPU
//   using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// 
//   // Epilogue operation as LinearCombinationBiasElementwise:
//   //  Y = GEMM(AB, C)
//   //  T[i, j] = BinaryOp(Y[i, j], Broadcast[i])
//   //  Z[i, j] = Elementwise(T[i, j])
//   using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
//     ElementOutputC,
//     ElementAccumulator,
//     ElementComputeEpilogue,
//     ElementOutputZ,
//     ElementOutputT,
//     128 / cutlass::sizeof_bits<ElementOutputC>::value,
//     cutlass::epilogue::thread::GELU_taylor<float>
//   >;  
//   // using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
//   //   ElementOutputC,
//   //   ElementAccumulator,
//   //   ElementAccumulator,
//   //   ElementAccumulator,
//   //   128 / cutlass::sizeof_bits<ElementOutput>::value,
//   //   cutlass::epilogue::thread::Identity,
//   //   cutlass::multiplies,
//   //   cutlass::epilogue::thread::Identity>;
// 
//   using GemmFunc = cutlass::gemm::device::GemmUniversalWithBroadcast<
//       ElementA,
//       cutlass::layout::RowMajor,
//       ElementB,
//       cutlass::layout::RowMajor,
//       ElementOutputC,
//       cutlass::layout::RowMajor,
//       ElementAccumulator,
//       cutlass::arch::OpClassTensorOp,
//       cutlass::arch::Sm80,
//       TShape,
//       WShape,
//       IShape,
//       EpilogueOutputOp,
//       SwizzleThreadBlock,
//       NumStages,
//       8,         // AlignA
//       8,         // AlignB
//       cutlass::arch::OpMultiplyAdd    // Operation performed by GEMM
//   >;
// 
//   /// Arguments
//   cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
//   ElementInputA *input = (ElementInputA *)(params.input);
//   ElementInputB *weight = (ElementInputB *)(params.weight);
//   ElementOutput *bias = (ElementOutput *)(params.bias);
//   ElementOutput *output = (ElementOutput *)(params.output);
// 
//   /// Only available in RRR format
//   int64_t batch_stride_C = params.is_vec_bias ? problem_size.n() : batch_stride_C = problem_size.mn().product();
// 
//   const int64_t lda = static_cast<int64_t>(params.m * params.k);
//   const int64_t ldb = static_cast<int64_t>(params.n * params.k);
//   const int64_t ldc_bias = params.is_vec_bias ? 0 : static_cast<int64_t>(params.m * params.n);
//   const int64_t ldc2 = static_cast<int64_t>(params.m * params.n);
//   const int64_t ldd = static_cast<int64_t>(params.m * params.n);
//   const int64_t ldr = 0;
//   const int64_t ldt = 0;
// 
//   ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
//   ElementComputeEpilogue beta = ElementComputeEpilogue(1);
// 
//   typename GemmSingle::Arguments arguments{
//       cutlass::gemm::GemmUniversalMode::kGemm,
//       problem_size,   // <- problem size of matrix multiplication
//       1,              // <- batch_count, k-dimension split factor
//       {alpha, beta},  // <- epilogue params, alpha, beta
//       input,          // <- input, ptr_A, A, shape={M, K}
//       weight,         // <- input, ptr_B, B, shape={K, N}
//       bias,           // <- input, ptr_C1, C1, shape={M, N} or {1, N}
//       xxx,            // <- input, ptr_C2, C2, shape={M, N}
//       output,         // <- output, ptr_D, Z, shape={M, N}
//       xxx,            // <- output, ptr_Vector, Broadcast, shape={M, N}
//       nullptr,        // <- output, ptr_Tensor, T
//       problem_size.mk().product(),  // <- batch_stride_A
//       problem_size.nk().product(),  // <- batch_stride_B
//       batch_stride_C,               // <- batch_stride_C1
//       problem_size.mn().product(),  // <- batch_stride_C2
//       problem_size.mn().product(),  // <- batch_stride_D
//       problem_size.m(),             // <- batch_stride_Vector
//       problem_size.mn().product(),  // <- batch_stride_Tensor
//       lda,
//       ldb,
//       ldc_bias,
//       ldc2,
//       ldd,
//       ldr,
//       ldt 
//   }; 
// 
// 
//   GemmFunc device_gemm;
// 
// 
//   // size_t workspace_size = GemmFunc::get_workspace_size(arguments);
//   // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
// 
//   cutlass::Status status = device_gemm.can_implement(arguments);
//   CHECK_CUTLASS(status);
// 
//   // status = device_gemm.initialize(arguments, workspace.get());
//   status = device_gemm.initialize(arguments, params.workspace);
//   CHECK_CUTLASS(status);
// 
//   //
//   // Run the GEMM
//   //
//   status = device_gemm();
//   CHECK_CUTLASS(status);
//   return status;
// }
