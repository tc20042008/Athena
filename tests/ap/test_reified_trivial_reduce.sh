sh make_axpr.sh
mkdir -p ap_workspace
mkdir -p $(pwd)/reified_drr
rm -rf $(pwd)/reified_drr/*
FLAGS_enable_ap=1 AP_PACKAGE_DUMP_DIR=$(pwd)/reified_drr AP_WORKSPACE_DIR=$(pwd)/ap_workspace AP_PATH=$(pwd)/ AP_ENTRY=$(pwd)/test_trivial_reduce.py.json FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_new_group_scheduler=1 python3.9 $(pwd)/paddle-tests/test_trivial_reduce.py
FLAGS_enable_ap=1 AP_PACKAGE_DIR=$(pwd)/reified_drr/ AP_PATH=$(pwd)/reified_drr/ AP_ENTRY=$(ls $(pwd)/reified_drr/*/reified_drr.json | awk '{print $1}') FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_new_group_scheduler=1 python3.9 $(pwd)/paddle-tests/test_trivial_reduce.py
