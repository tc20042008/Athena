import ap_tpl_codegen
import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util

def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(
        param_struct_name="args"
    )

def get_anchor_iter_var_names():
    return ["coord.j", "coord.k"]

class MatmulBinaryTemplate:
    def __init__(
        self,
        program_translator,
        mut_kernel_arg_id_lazy_ctx,
    ):
        self.program_translator = program_translator
        self.mut_kernel_arg_id_lazy_ctx = mut_kernel_arg_id_lazy_ctx
        self.kernel_arg_translator = make_kernel_arg_translator()
        self.dtype2type_name = OrderedDict(
            [
                [PointerType.const_float_ptr, "const float*"],
                [PointerType.const_float16_ptr, "const half*"],
                [PointerType.float_ptr, "float*"],
                [PointerType.float16_ptr, "half*"],
                [DataType.float, "float"],
                [DataType.float16, "half"],
                [DataType.int64_t, "int64_t"],
            ]
        )

    def _register_name(self, pair):
        registry = self.mut_kernel_arg_id_lazy_ctx
        registry.get_or_create_kernel_arg_id_manul_var_name(
            kernel_arg_id=pair[0],
            cpp_var_name=pair[1]
        )

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
        kernel_arg_id_and_names = [
            [input0_karg, "input"],
            [input1_karg, "weight"],
            [input2_karg, "bias"],
            [output_karg, "output"],
            [m_karg,      "m"],
            [n_karg,      "n"],
            [k_karg,      "k"],
        ]
        map(self._register_name, kernel_arg_id_and_names)
        mut_lir_code_gen_ctx = low_level_ir_code_gen_ctx_util.CudaLikeIrCodeGenCtx()
        self.program_translator.translate(
            mut_kernel_arg_id_lazy_ctx=self.mut_kernel_arg_id_lazy_ctx,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx
        )
        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str()
        print("matmul_binary_epilogue_code: ", trivial_code_str)
        arg_type_list = map(lambda pair: pair[0].type, kernel_arg_id_and_names)
        def get_kernel_arg_list_str():
            def declare_epilogue_arguments_field(pair):
                kernel_arg_id = pair[0]
                var_name = pair[1]
                field_name = self.kernel_arg_translator.get_param_struct_field_name(var_name)
                dtype = kernel_arg_id.type
                type_name = self.dtype2type_name[dtype]
                return f"{type_name} {field_name}"
            return ", ".join(
                map(declare_epilogue_arguments_field, kernel_arg_id_and_names)
            )

        def get_epilogue_arguments_fields_str():
            def declare_epilogue_arguments_field(pair):
                kernel_arg_id = pair[0]
                var_name = pair[1]
                field_name = self.kernel_arg_translator.get_param_struct_field_name(var_name)
                dtype = kernel_arg_id.type
                type_name = self.dtype2type_name[dtype]
                return f"    {type_name} {field_name};"
            generated_kernel_arg_id_and_names = (
                self.mut_kernel_arg_id_lazy_ctx.generated_kernel_arg_id2unique_name.items()
            )
            return "\n".join(
                map(declare_epilogue_arguments_field, generated_kernel_arg_id_and_names)
            )

        def get_epilogue_arguments_init_str(param_obj_name):
            def declare_epilogue_arguments_assign(pair):
                kernel_arg_id = pair[0]
                var_name = pair[1]
                field_name = self.kernel_arg_translator.get_param_struct_field_name(var_name)
                return f"  {param_obj_name}.{field_name} = {var_name};"
            generated_kernel_arg_id_and_names = (
                self.mut_kernel_arg_id_lazy_ctx.generated_kernel_arg_id2unique_name.items()
            )
            return "\n".join(
                map(declare_epilogue_arguments_assign, generated_kernel_arg_id_and_names)
            )
        project_module = self.make_project(
            trivial_code_str,
            dtype_of_ptr_args,
            arg_type_list,
            get_kernel_arg_list_str,
            get_epilogue_arguments_fields_str,
            get_epilogue_arguments_init_str
        )
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=map(
                    lambda pair: pair[0].runtime_getter,
                    kernel_arg_id_and_names
                )
            ),
        )

    def make_project(
        self,
        trivial_code_str,
        dtype_of_ptr_args,
        arg_type_list,
        get_kernel_arg_list_str,
        get_epilogue_arguments_fields_str,
        get_epilogue_arguments_init_str
    ):
        code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_fp16.h>

#include "native_matmul.cuh"

namespace ap {

template <typename T, int NumIns = 1, int NumOuts = 1>
struct AddFunctor {
  struct Arguments {
EPILOGUE_ARGUMENTS_FIELDS
  };

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x, const Arguments& args, const MatrixCoord& coord) const {
    T out;
    AP_GENERATED_BINARY_EPILOGUE_STRING;
    return out;
  }
};

}

extern "C" {

void MatmulBinaryKernel(void* stream_ptr, AP_KERNEL_ARGS_DECLARE) {
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

EPILOGUE_ARGUMENTS_INIT

  ap::NativeMatmulAdd<AP_GENERATED_ELEMENT_DTYPE, ap::AddFunctor<AP_GENERATED_ELEMENT_DTYPE, 1, 1>>(params, epilogue_args);
}
}

  """
        output_dtype = self.dtype2type_name[dtype_of_ptr_args[3]]

        code = (
            code_template.replace(
                "AP_GENERATED_BINARY_EPILOGUE_STRING", trivial_code_str
            )
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
            .replace("AP_KERNEL_ARGS_DECLARE", get_kernel_arg_list_str())
            .replace("EPILOGUE_ARGUMENTS_FIELDS", get_epilogue_arguments_fields_str())
            .replace("EPILOGUE_ARGUMENTS_INIT", get_epilogue_arguments_init_str("epilogue_args"))
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

        return CodeModule(
            FuncDeclare(
                DataType.void,
                "MatmulBinaryKernel",
                [
                    PointerType.void_ptr,
                    *arg_type_list
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
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
