#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

FILENAME=${1:-"test_trivial_reduce"}
#FILENAME=${1:-"test_matmul_unary"}

sh make_axpr.sh ${FILENAME}

export FLAGS_enable_ap=1
export AP_WORKSPACE_DIR=$(pwd)/ap_workspace
export AP_PATH=$(pwd)/
#export AP_ENTRY=$(pwd)/${FILENAME}.py.json

export FLAGS_check_infer_symbolic=1
export FLAGS_enable_pir_api=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_new_group_scheduler=1

#export GLOG_v=4

export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_trivial_reduce"
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas -x true --force-overwrite true -o test_trivial_reduce"

${nsys_args} python $(pwd)/paddle-tests/${FILENAME}.py 2>&1 | tee log_${FILENAME}.txt
