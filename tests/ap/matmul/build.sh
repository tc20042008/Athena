#!/bin/bash

CUDA_ROOT=/usr/local/cuda
CUTLASS_DIR=/work/abstract_pass/Athena/tests/ap/matmul/cutlass
SOURCE_DIR=/work/abstract_pass/Athena/tests/ap/matmul

nvcc -std=c++17 -O3 \
    -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr \
    -I ${CUDA_ROOT}/include \
    -I ${CUTLASS_DIR}/include \
    -I ${CUTLASS_DIR}/tools/util/include \
    -I ${SOURCE_DIR} \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -DCUTLASS_DEBUG_TRACE_LEVEL=1 \
    --shared cutlass_matmul.cu util.cu -o libmatmul.so

nvcc -std=c++17 -O3 \
    -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr \
    -I ${CUTLASS_DIR}/include \
    -I ${CUTLASS_DIR}/tools/util/include \
    -L./ -lmatmul -lcuda -lcudart \
    test_main.cc -o test_main
