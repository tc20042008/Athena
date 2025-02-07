import ap_tpl_codegen
import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util

def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(
        param_struct_name="param"
    )

def get_loop_iter_var_names():
    return ["i", "j"]

class MatmulBinaryTemplate:
    def __init__(
        self,
        program_translator,
        mut_kernel_arg_id_lazy_ctx,
    ):
        self.program_translator = program_translator
        self.mut_kernel_arg_id_lazy_ctx = mut_kernel_arg_id_lazy_ctx

    def compile(
        self,
        dtype_of_ptr_args,
        input0_karg,
        input1_karg,
        input2_karg,
        output_karg,
        m_karg,
        n_karg,
        k_karg,
    ):
        mut_lir_code_gen_ctx = low_level_ir_code_gen_ctx_util.CudaLikeIrCodeGenCtx()
        self.program_translator.translate(
            mut_kernel_arg_id_lazy_ctx=self.mut_kernel_arg_id_lazy_ctx,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx
        )
        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str()
        print("matmul_binary_epilogue_code: ", trivial_code_str)

        project_module = self.make_project(trivial_code_str, dtype_of_ptr_args)
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=[
                    input0_karg.runtime_getter,
                    input1_karg.runtime_getter,
                    input2_karg.runtime_getter,
                    output_karg.runtime_getter,
                    m_karg.runtime_getter,
                    n_karg.runtime_getter,
                    k_karg.runtime_getter,
                ]
            ),
        )

    def make_project(self, trivial_code_str, dtype_of_ptr_args):
        code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_fp16.h>

#include "native_matmul.cuh"

namespace ap {

struct BroadcastConfig {
};

struct InputExtra {
  const void* ptr{nullptr};
  bool need_broadcast{false};
  BroadcastConfig config;
};

template <typename T, int NumIns = 1, int NumOuts = 1>
struct AddFunctor {
  struct Arguments {
    int out_shape_len{0};
    int64_t out_shape[10];
    InputExtra ins[NumIns];
    // void* outs[NumOuts];
  };

  __forceinline__ __host__ __device__
  T Load(const Arguments& args, const MatrixCoord& coord, int idx) const {
    // Specially for the case of out_shape_len = 2
    size_t offset = coord.j * args.out_shape[1] + coord.k;
    return reinterpret_cast<const T*>(args.ins[idx].ptr)[offset];
  }

  __forceinline__ __host__ __device__
  T LoadWithBroadcast(const Arguments& args, const MatrixCoord& coord, int idx) const {
    // Specially for the bias case
    size_t offset = coord.k;
    return reinterpret_cast<const T*>(args.ins[idx].ptr)[offset];
  }

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x, const Arguments& args, const MatrixCoord& coord) const {
    T y = args.ins[0].need_broadcast ? LoadWithBroadcast(args, coord, 0) : Load(args, coord, 0);
    T out;
    AP_GENERATED_BINARY_EPILOGUE_STRING;
    return out;
    // return x + y;
  }
};

}

extern "C" {

void MatmulBinaryKernel(void* stream_ptr, const AP_GENERATED_INPUT0_DTYPE* input, const AP_GENERATED_INPUT1_DTYPE* weight, const AP_GENERATED_INPUT2_DTYPE* bias, AP_GENERATED_OUTPUT_DTYPE* output, const int64_t m, const int64_t n, const int64_t k) {
  ap::GemmEpilogueParams params;

  params.batch_count = 1;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = nullptr;
  params.output = output;

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  params.stream = *cuda_stream_ptr;

  // std::cout << "-- [MatmulBinaryKernel] m=" << m << ", n=" << n << ", k=" << k << std::endl;
  // std::cout << "-- [MatmulBinaryKernel] input=" << input << std::endl;
  // std::cout << "-- [MatmulBinaryKernel] weight=" << weight << std::endl;
  // std::cout << "-- [MatmulBinaryKernel] bias=" << bias << std::endl;
  // std::cout << "-- [MatmulBinaryKernel] output=" << output << std::endl;
  // std::cout << "-- [MatmulBinaryKernel] stream=" << cuda_stream_ptr << std::endl;

  typename ap::AddFunctor<AP_GENERATED_ELEMENT_DTYPE, 1, 1>::Arguments epilogue_args;
  epilogue_args.out_shape_len = 2;
  epilogue_args.out_shape[0] = m;
  epilogue_args.out_shape[1] = n;

  epilogue_args.ins[0].ptr = bias;
  epilogue_args.ins[0].need_broadcast = false;

  ap::NativeMatmulAdd<AP_GENERATED_ELEMENT_DTYPE, ap::AddFunctor<AP_GENERATED_ELEMENT_DTYPE, 1, 1>>(params, epilogue_args);
}
}

  """

        dtype2type_name = OrderedDict(
            [
                [DataType.float, "float"],
                [DataType.float16, "half"],
            ]
        )
        input0_dtype = dtype2type_name[dtype_of_ptr_args[0]]
        input1_dtype = dtype2type_name[dtype_of_ptr_args[1]]
        input2_dtype = dtype2type_name[dtype_of_ptr_args[2]]
        output_dtype = dtype2type_name[dtype_of_ptr_args[3]]

        code = (
            code_template.replace(
                "AP_GENERATED_BINARY_EPILOGUE_STRING", trivial_code_str
            )
            .replace("AP_GENERATED_INPUT0_DTYPE", input0_dtype)
            .replace("AP_GENERATED_INPUT1_DTYPE", input1_dtype)
            .replace("AP_GENERATED_INPUT2_DTYPE", input1_dtype)
            .replace("AP_GENERATED_OUTPUT_DTYPE", output_dtype)
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
        )

        source_dir = "/workspace/Athena/tests/ap/matmul"
        cutlass_dir = "/workspace/Athena/tests/ap/matmul/cutlass"
        compile_cmd = (
            "nvcc -std=c++17 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        )
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/include"
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/tools/util/include"
        compile_cmd = compile_cmd + " -I " + source_dir
        compile_cmd = (
            compile_cmd
            + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 "
        )
        compile_cmd = (
            compile_cmd
            + " --shared matmul_binary_kernel.cu -o libmatmul_binary_kernel.so"
        )

        dtype2const_pointer_type = OrderedDict(
            [
                [DataType.float, PointerType.const_float_ptr],
                [DataType.float16, PointerType.const_float16_ptr],
            ]
        )
        dtype2pointer_type = OrderedDict(
            [
                [DataType.float, PointerType.float_ptr],
                [DataType.float16, PointerType.float16_ptr],
            ]
        )
        return CodeModule(
            FuncDeclare(
                DataType.void,
                "MatmulBinaryKernel",
                [
                    PointerType.void_ptr,
                    dtype2const_pointer_type[dtype_of_ptr_args[0]],
                    dtype2const_pointer_type[dtype_of_ptr_args[1]],
                    dtype2const_pointer_type[dtype_of_ptr_args[2]],
                    dtype2pointer_type[dtype_of_ptr_args[3]],
                    DataType.const_int64,
                    DataType.const_int64,
                    DataType.const_int64,
                ],
            ),
            Project(
                nested_files=Project.Directory(
                    ["matmul_binary_kernel.cu", Project.FileContent(code)],
                    ["make.sh", Project.FileContent(compile_cmd)],
                ),
                compile_cmd="sh make.sh",
                so_relative_path="libmatmul_binary_kernel.so",
            ),
        )


def KernelDispatch(ctx):
    so_func = ctx.get_so_function("MatmulBinaryKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()

    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    input = getters[0](ctx)
    weight = getters[1](ctx)
    bias = getters[2](ctx)
    output = getters[3](ctx)
    m = getters[4](ctx)
    n = getters[5](ctx)
    k = getters[6](ctx)
    so_func(stream_ptr, input, weight, bias, output, m, n, k)
