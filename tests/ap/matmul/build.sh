#!/bin/bash

CUDA_ROOT=/usr/local/cuda
CUTLASS_DIR=/work/abstract_pass/Athena/tests/ap/matmul/cutlass
SOURCE_DIR=/work/abstract_pass/Athena/tests/ap/matmul
AP_LIB_DIR=/work/abstract_pass/Athena/tests/ap/ap_workspace/2984375663431405214/main

#SO_NAME=matmul_add_unary_kernel
SO_NAME=matmul_kernel
TEST_NAME=test_main_matmul_unary

rm -rf lib${SO_NAME}.so
rm -rf test_main

nvcc -std=c++17 -O3 \
    -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr \
    -I ${CUDA_ROOT}/include \
    -I ${CUTLASS_DIR}/include \
    -I ${CUTLASS_DIR}/tools/util/include \
    -I ${SOURCE_DIR} \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
    -DTUNE_TILE_SHAPE=0 \
    --shared kernel.cu -o lib${SO_NAME}.so

#    --shared ${AP_LIB_DIR}/matmul_add_unary_kernel.cu -o libmatmul_add_unary_kernel.so

nvcc -std=c++17 -O3 \
    -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr \
    -I ${CUTLASS_DIR}/include \
    -I ${CUTLASS_DIR}/tools/util/include \
    -I ${SOURCE_DIR} \
    -I ${AP_LIB_DIR} \
    -L./ -l${SO_NAME} -lcuda -lcudart \
    -DENABLE_PROFILE=1  \
    test_util.cu ${TEST_NAME}.cc -o test_main
