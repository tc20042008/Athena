import ap_tpl_codegen


class MatrixAddUnaryTemplate:
  def __init__(self, trivial_op_code_gen_class):
    self.class_factory = ap_tpl_codegen.GeneratorClassFactory()
    self.lir_code_gen_ctx_class = ap_tpl_codegen.CudaLikeIrCodeGenCtx
    self.trivial_op_code_gen_class = trivial_op_code_gen_class

  def compile(self, dim0_karg, input0_karg, input1_karg, output_karg):
    dim0_value = dim0_karg.value
    loop_index_tuple_expr = IndexTupleExpr.Domain([dim0_value])
    fusion_code_gen = self.trivial_op_code_gen_class(
      class_factory=self.class_factory,
      loop_index_tuple_expr=loop_index_tuple_expr,
      loop_var_names=["threadIdx.x"]
    )
    lir_code_gen_ctx = self.lir_code_gen_ctx_class()
    values = fusion_code_gen.load_from_register(lir_code_gen_ctx, "x", 0)
    values = fusion_code_gen.compute(lir_code_gen_ctx, values)
    values = fusion_code_gen.store_to_register(lir_code_gen_ctx, values, "y", 0)
    print("===================================")
    print("fusion_code_gen.compute(): ", values)
    print("low level ir of fusion_op:\n", lir_code_gen_ctx.get_stmts_joined_str())
    print("===================================")
    trivial_code_str = lir_code_gen_ctx.get_stmts_joined_str()
    project_module = self.make_project(trivial_code_str)
    return CodeGenResult(
      module=project_module,
      kernel_dispatch_func=KernelDispatch,
      kernel_dispatch_const_data=BuiltinSerializableAttrMap(
        kernel_args_getters=[
          dim0_karg.runtime_getter,
          input0_karg.runtime_getter,
          input1_karg.runtime_getter,
          output_karg.runtime_getter
        ]
      )
    )

  def make_project(self, trivial_code_str):
    code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_fp16.h>

#define CINN_WITH_CUDA
#include "epilogue_op.h"

template <typename T>
struct UnaryEpilogueFunctor {
  using Arguments = typename ap::ScaleFunctor<T>::Arguments;

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    T y;
    TRIVAL_UNARY_OPERATOR_STRING;
    return y;
  }
};

#include "cutlass_matmul.cuh"

extern "C" {

void MatmulAddUnaryKernel(const int64_t num, const half* input, const half* weight, half* output) {
  ap::GemmEpilogueParams params;

  params.batch_count = 1;
  params.m = 256;
  params.n = 256;
  params.k = 256;

  params.input = input;
  params.weight = weight;
  params.bias = nullptr;
  params.output = output;

  std::cout << "-- [MatmulAddUnaryKernel] input=" << input << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] weight=" << weight << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] output=" << output << std::endl;

  UnaryEpilogueFunctor<float>::Arguments unary_args{1.0};
  ap::CutlassMatmulAddUnary<cutlass::half_t, float, UnaryEpilogueFunctor, false, false>(params, unary_args);
}
}

  """
    code = code_template.replace("TRIVAL_UNARY_OPERATOR_STRING", trivial_code_str)
    print("---------- generated code ----------")
    print(code)
    print("")
    source_dir = "/work/abstract_pass/Athena/tests/ap/matmul"
    cutlass_dir = "/work/abstract_pass/Athena/tests/ap/matmul/cutlass"
    compile_cmd = "nvcc -std=c++17 -O3 --ptxas-options=-v -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
    compile_cmd = compile_cmd + " -I " + cutlass_dir + "/include"
    compile_cmd = compile_cmd + " -I " + cutlass_dir + "/tools/util/include"
    compile_cmd = compile_cmd + " -I " + source_dir
    compile_cmd = compile_cmd + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=1 "
    compile_cmd = compile_cmd + " --shared matmul_add_unary_kernel.cu -o libmatmul_add_unary_kernel.so"
    print("-- CodeModule")
    return CodeModule(
      FuncDeclare(DataType.void, "MatmulAddUnaryKernel", [
        DataType.const_int64,
        PointerType.const_float16_ptr,
        PointerType.const_float16_ptr,
        PointerType.float16_ptr,
      ]),
      Project(
        nested_files=Project.Directory(
          ["matmul_add_unary_kernel.cu", Project.FileContent(code)],
          ["make.sh", Project.FileContent(compile_cmd)]
        ),
        compile_cmd="sh make.sh",
        so_relative_path="libmatmul_add_unary_kernel.so"
      )
    )

def KernelDispatch(ctx):
  print("-- KernelDispatch")
  so_func = ctx.get_so_function("MatmulAddUnaryKernel")
  getters = ctx.kernel_dispatch_const_data.kernel_args_getters
  print(f"-- [KernelDispatch] getters[0](ctx): {getters[0](ctx)}")
  print(f"-- [KernelDispatch] getters[1](ctx): {getters[1](ctx)}")
  print(f"-- [KernelDispatch] getters[2](ctx): {getters[2](ctx)}")
  print(f"-- [KernelDispatch] getters[3](ctx): {getters[3](ctx)}")
  so_func(getters[0](ctx), getters[1](ctx), getters[2](ctx), getters[3](ctx))
