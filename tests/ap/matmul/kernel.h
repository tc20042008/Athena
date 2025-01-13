// auto generated


extern "C" {

void MatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int m, int n, int k);

void MatmulAddBinaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* broadcast, void* broadcast_out, void* output, int m, int n, int k);

}
