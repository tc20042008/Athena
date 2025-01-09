// #include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass_epilogue/thread/linear_combination_unary.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal.h"

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

#if 0
  // Epilogue operation as LinearCombinationRelu:
  //  d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij)
  //
  // - sum_k(a_ik * b_kj), the intermedia result of matrix product, A * B
  // - c_ij, the bias 
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias
#endif
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationUnary<
      ap::ScaleFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

  // how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

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
      EpilogueOp,
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
  int64_t batch_stride_C = problem_size.n();
  if(!params.is_vec_bias) {
    batch_stride_C = problem_size.mn().product();
  }

  long lda = (long)params.lda;
  long ldb = (long)params.ldb;
  long ldc_bias = 0;
  if(!params.is_vec_bias) {
    ldc_bias = (long)params.ldd;
  }
  long ldd = (long)params.ldd;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  typename GemmFunc::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,   // <- problem size of matrix multiplication
      1,              // <- batch_count, k-dimension split factor
      {alpha, beta, {0.1}},  // <- epilogue, alpha, beta
      input,          // <- reference to matrix A on device
      weight,         // <- reference to matrix B on device
      bias,           // <- reference to matrix C on device
      output,         // <- reference to matrix D on device
      problem_size.mk().product(),
      problem_size.nk().product(),
      batch_stride_C,
      problem_size.mn().product(),
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

cutlass::Status CutlassMatmulAddBinary(const GemmEpilogueParams& params) {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
  using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
  using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

  using TShape = cutlass::gemm::GemmShape<256, 128, 32>;// threadblock tile
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;  // warp tile
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;   // MMA Op tile
  constexpr int NumStages = 3;

#if 0
  // Epilogue operation as LinearCombinationRelu:
  //  d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij)
  //
  // - sum_k(a_ik * b_kj), the intermedia result of matrix product, A * B
  // - c_ij, the bias 
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias
#endif
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationUnary<
      ap::ScaleFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

  // how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

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
      EpilogueOp,
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
  int64_t batch_stride_C = problem_size.n();
  if(!params.is_vec_bias) {
    batch_stride_C = problem_size.mn().product();
  }

  long lda = (long)params.lda;
  long ldb = (long)params.ldb;
  long ldc_bias = 0;
  if(!params.is_vec_bias) {
    ldc_bias = (long)params.ldd;
  }
  long ldd = (long)params.ldd;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  typename GemmFunc::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,   // <- problem size of matrix multiplication
      1,              // <- batch_count, k-dimension split factor
      {alpha, beta, {0.1}},  // <- epilogue, alpha, beta
      input,          // <- reference to matrix A on device
      weight,         // <- reference to matrix B on device
      bias,           // <- reference to matrix C on device
      output,         // <- reference to matrix D on device
      problem_size.mk().product(),
      problem_size.nk().product(),
      batch_stride_C,
      problem_size.mn().product(),
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
