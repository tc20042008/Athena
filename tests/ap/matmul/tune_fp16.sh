#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

#export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/abstract_pass/Athena/tests/ap/matmul:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH

nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o cutlass_matmul"

INST_SHAPE_SM80="16,8,16"
TILE_SHAPES=(
    "256,128,32 3 64,64,32"
    "128,256,32 3 64,64,32"
    "256,64,32 3 64,64,32"
    "256,64,32 4 64,64,32"
    "64,256,32 4 64,64,32"
    "128,128,32 3 64,64,32"
    "128,128,32 4 64,64,32"
    "128,128,32 5 64,64,32"
    "128,64,32 6 64,32,32"
    "64,128,32 6 32,64,32"
    "64,64,32 10 32,32,32"
    "256,128,64 3 64,64,64"
    "128,256,64 3 64,64,64"
    "256,64,64 4 64,64,64"
    "64,256,64 4 64,64,64"
    "128,128,64 4 64,64,64"
    "256,64,64 3 64,64,64"
    "64,256,64 3 64,64,64"
    "128,128,64 3 64,64,64"
    "128,64,64 3 64,32,64"
    "64,128,64 3 32,64,64"
    "64,64,64 5 32,32,64"
    "32,64,64 5 16,32,64"
    "16,64,64 5 16,32,64"
)

SWIZZLE_VALUES=("1" "2" "4")

for config in "${TILE_SHAPES[@]}"; do
    IFS=' ' read -r tshape stages wshape <<< "$config"
    for value in "${SWIZZLE_VALUES[@]}"; do
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "tshape={$tshape}, wshape={$wshape}, ishape={$INST_SHAPE_SM80}, stages={$stages}"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        cat <<EOF > tile_shape.h
#pragma once

namespace ap {

template <typename ElementT>
struct GemmTuningConfig {
  using TShape = cutlass::gemm::GemmShape<$tshape>;
  using WShape = cutlass::gemm::GemmShape<$wshape>;
  using IShape = cutlass::gemm::GemmShape<$INST_SHAPE_SM80>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<$value>;
  static constexpr int NumStages = $stages;
};

}
EOF
        cat tile_shape.h
        ./build.sh
        ${nsys_args} ./test_main
        echo ""
    done
done
