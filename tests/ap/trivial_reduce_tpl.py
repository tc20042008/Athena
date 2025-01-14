import ap_tpl_codegen

class TrivialReduceTemplate:
  def __init__(self, trivial_op_code_gen_class):
    self.class_factory = ap_tpl_codegen.GeneratorClassFactory()
    self.lir_code_gen_ctx_class = ap_tpl_codegen.CudaLikeIrCodeGenCtx
    self.trivial_op_code_gen_class = trivial_op_code_gen_class

  def compile(self, dim0_karg, input_karg, output_karg):
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
          input_karg.runtime_getter,
          output_karg.runtime_getter
        ]
      )
    )

  def make_project(self, trivial_code_str):
    code_template = """
#include <cstdint>
#include <iostream>
#define CINN_WITH_CUDA

extern "C" __global__
void trivial_reduce(const int64_t num, const float* input, float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float buf[1024];
  float y;
  if (idx < num) {
    float x = input[idx];
    TRIVIAL_CODE_STRING;
    buf[idx] = y;
  }
  __syncthreads();
  if (idx==0) {
    for (int i = 1; i < num; ++i) {
      buf[0] += buf[i];
    }
    output[0] = buf[0];
  }
}

extern "C"
void api_trivial_reduce(void* stream_ptr, const int64_t num, const float* input, float* output) {
  std::cout << "stream_ptr: " << stream_ptr << std::endl;
  cudaStream_t stream = *(cudaStream_t*)(stream_ptr);
  std::cout << "yiqun nice" << std::endl;
  std::cout << "num: " << num << std::endl;
  std::cout << "input: " << input << std::endl;
  std::cout << "output: " << output << std::endl;
  return trivial_reduce<<<1, num, 0, stream>>>(num, input, output);
}

  """
    code = code_template.replace("TRIVIAL_CODE_STRING", trivial_code_str)
    compile_cmd = "nvcc --ptxas-options=-v --compiler-options '-fPIC' -gencode arch=compute_80,code=sm_80 --shared trivial_reduce.cu -o libtrivial_reduce.so"
    return CodeModule(
      FuncDeclare(DataType.void, "api_trivial_reduce", [
        PointerType.void_ptr,
        DataType.const_int64,
        PointerType.const_float_ptr,
        PointerType.float_ptr,
      ]),
      Project(
        nested_files=Project.Directory(
          ["trivial_reduce.cu", Project.FileContent(code)],
          ["make.sh", Project.FileContent(compile_cmd)]
        ),
        compile_cmd="sh make.sh",
        so_relative_path="libtrivial_reduce.so"
      )
    )

def KernelDispatch(ctx):
  so_func = ctx.get_so_function("api_trivial_reduce")
  getters = ctx.kernel_dispatch_const_data.kernel_args_getters
  stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
  print("stream_ptr:", stream_ptr)
  print("getters[0](ctx):", getters[0](ctx))
  print("getters[1](ctx):", getters[1](ctx))
  print("getters[2](ctx):", getters[2](ctx))
  so_func(stream_ptr, getters[0](ctx), getters[1](ctx), getters[2](ctx))
