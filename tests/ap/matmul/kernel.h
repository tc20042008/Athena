// auto generated

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {

void MatmulAddKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b);

void MatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b);

void MatmulAddBinaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* broadcast, void* broadcast_out, void* output, int m, int n, int k, bool need_broadcast);

}
