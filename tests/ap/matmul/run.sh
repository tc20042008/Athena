#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

#export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/abstract_pass/Athena/tests/ap/matmul:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH

#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o cutlass_matmul"


#AP_LIB_DIR=/work/abstract_pass/Athena/tests/ap/ap_workspace/15859713568798564682/main
#export LD_LIBRARY_PATH=${AP_LIB_DIR}:$LD_LIBRARY_PATH
ldd ./test_main
${nsys_args} ./test_main
