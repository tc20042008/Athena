sh make_axpr.sh
FLAGS_enable_ap=1 AP_WORKSPACE_DIR=$(pwd)/ap_workspace AP_PATH=$(pwd)/ FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_new_group_scheduler=1 python3.9 $(pwd)/paddle-tests/test_binary_trivial_reduce.py
