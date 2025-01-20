import ap_tpl_codegen


class MatrixAddUnaryTemplate:
    def __init__(self, trivial_op_code_gen_class):
        self.class_factory = ap_tpl_codegen.GeneratorClassFactory()
        self.lir_code_gen_ctx_class = ap_tpl_codegen.CudaLikeIrCodeGenCtx
        self.trivial_op_code_gen_class = trivial_op_code_gen_class

    def compile(
        self,
        dtype_of_ptr_args,
        input0_karg,
        input1_karg,
        output_karg,
        batch_count_karg,
        m_karg,
        n_karg,
        k_karg,
    ):
        m_value = m_karg.value
        loop_index_tuple_expr = IndexTupleExpr.Domain([m_value])
        fusion_code_gen = self.trivial_op_code_gen_class(
            class_factory=self.class_factory,
            loop_index_tuple_expr=loop_index_tuple_expr,
            loop_var_names=["threadIdx.x"],
        )
        lir_code_gen_ctx = self.lir_code_gen_ctx_class()
        values = fusion_code_gen.load_from_register(lir_code_gen_ctx, "x", 0)
        values = fusion_code_gen.compute(lir_code_gen_ctx, values)
        values = fusion_code_gen.store_to_register(lir_code_gen_ctx, values, "y", 0)
        trivial_code_str = lir_code_gen_ctx.get_stmts_joined_str()
        print("===================================")
        print("fusion_code_gen.compute(): ", values)
        print("low level ir of fusion_op:\n", trivial_code_str)
        print("===================================")

        project_module = self.make_project(trivial_code_str, dtype_of_ptr_args)
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=[
                    input0_karg.runtime_getter,
                    input1_karg.runtime_getter,
                    output_karg.runtime_getter,
                    batch_count_karg.runtime_getter,
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

#define CINN_WITH_CUDA
#include "epilogue_op.h"

template <typename T>
struct UnaryEpilogueFunctor {
  using Arguments = typename ap::ScaleFunctor<T>::Arguments;

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    T y;
    AP_GENERATED_UNARY_EPILOGUE_STRING;
    return y;
  }
};

#include "cutlass_matmul.cuh"

extern "C" {

void MatmulAddUnaryKernel(void* stream_ptr, const AP_GENERATED_INPUT0_DTYPE* input, const AP_GENERATED_INPUT1_DTYPE* weight, AP_GENERATED_OUTPUT_DTYPE* output, const int64_t batch_count, const int64_t m, const int64_t n, const int64_t k) {
  ap::GemmEpilogueParams params;

  params.batch_count = batch_count;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = nullptr;
  params.output = output;

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  params.stream = *cuda_stream_ptr;

  // std::cout << "-- [MatmulAddUnaryKernel] batch_count=" << batch_count <<", m=" << m << ", n=" << n << ", k=" << k << std::endl;
  // std::cout << "-- [MatmulAddUnaryKernel] input=" << input << std::endl;
  // std::cout << "-- [MatmulAddUnaryKernel] weight=" << weight << std::endl;
  // std::cout << "-- [MatmulAddUnaryKernel] output=" << output << std::endl;
  // std::cout << "-- [MatmulAddUnaryKernel] stream=" << cuda_stream_ptr << std::endl;

  UnaryEpilogueFunctor<float>::Arguments unary_args{0.1};
  ap::CutlassMatmulAddUnary<AP_GENERATED_ELEMENT_DTYPE, float, UnaryEpilogueFunctor, false, false>(params, unary_args);
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
        output_dtype = dtype2type_name[dtype_of_ptr_args[2]]

        code = (
            code_template.replace(
                "AP_GENERATED_UNARY_EPILOGUE_STRING", trivial_code_str
            )
            .replace("AP_GENERATED_INPUT0_DTYPE", input0_dtype)
            .replace("AP_GENERATED_INPUT1_DTYPE", input1_dtype)
            .replace("AP_GENERATED_OUTPUT_DTYPE", output_dtype)
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
        )

        source_dir = "/work/abstract_pass/Athena/tests/ap/matmul"
        cutlass_dir = "/work/abstract_pass/Athena/tests/ap/matmul/cutlass"
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
            + " --shared matmul_add_unary_kernel.cu -o libmatmul_add_unary_kernel.so"
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
                "MatmulAddUnaryKernel",
                [
                    PointerType.void_ptr,
                    dtype2const_pointer_type[dtype_of_ptr_args[0]],
                    dtype2const_pointer_type[dtype_of_ptr_args[1]],
                    dtype2pointer_type[dtype_of_ptr_args[2]],
                    DataType.const_int64,
                    DataType.const_int64,
                    DataType.const_int64,
                    DataType.const_int64,
                ],
            ),
            Project(
                nested_files=Project.Directory(
                    ["matmul_add_unary_kernel.cu", Project.FileContent(code)],
                    ["make.sh", Project.FileContent(compile_cmd)],
                ),
                compile_cmd="sh make.sh",
                so_relative_path="libmatmul_add_unary_kernel.so",
            ),
        )


def KernelDispatch(ctx):
    so_func = ctx.get_so_function("MatmulAddUnaryKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    # print(f"-- [KernelDispatch] stream_ptr: {stream_ptr}")

    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    # print(f"-- [KernelDispatch] getters[0](ctx): {getters[0](ctx)}")
    # print(f"-- [KernelDispatch] getters[1](ctx): {getters[1](ctx)}")
    # print(f"-- [KernelDispatch] getters[2](ctx): {getters[2](ctx)}")
    # print(f"-- [KernelDispatch] getters[3](ctx): {getters[3](ctx)}")
    # print(f"-- [KernelDispatch] getters[4](ctx): {getters[4](ctx)}")
    # print(f"-- [KernelDispatch] getters[5](ctx): {getters[5](ctx)}")
    so_func(
        stream_ptr,
        getters[0](ctx),
        getters[1](ctx),
        getters[2](ctx),
        getters[3](ctx),
        getters[4](ctx),
        getters[5](ctx),
        getters[6](ctx),
    )
