#include <cuda_profiler_api.h>
#include "matmul.h"


class GpuTimer {
 public:
  explicit GpuTimer(bool profile) : profile_(profile) {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    CHECK_CUDA(cudaEventDestroy(start_));
    CHECK_CUDA(cudaEventDestroy(stop_));
  }

  void Start(cudaStream_t stream) {
    CHECK_CUDA(cudaEventRecord(start_, stream));
    if (profile_) {
      CHECK_CUDA(cudaProfilerStart());
    }
  }

  void Stop(cudaStream_t stream) {
    CHECK_CUDA(cudaEventRecord(stop_, stream));
    if (profile_) {
      CHECK_CUDA(cudaProfilerStop());
    }
  }

  float ElapsedTime() {
    float milliseconds = 0;
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

 private:
  bool profile_{false};
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
};


#if ENABLE_PROFILE
#define KERNEL_PROFILE(func)                    \
  {                                             \
    for (int i = 0; i < 10; ++i) {              \
      func;                                     \
    }                                           \
    CHECK_CUDA(cudaStreamSynchronize(stream));  \
    GpuTimer gpu_timer(true);                   \
    gpu_timer.Start(stream);                    \
    for (int i = 0; i < 1000; ++i) {            \
      func;                                     \
    }                                           \
    gpu_timer.Stop(stream);                     \
  }
#else
#define KERNEL_PROFILE(func) func
#endif


void TuneMatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k);
