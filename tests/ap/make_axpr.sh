#!/bin/bash

TEST_FILENAME=${1:-"test_trivial_reduce"}
#TEST_FILENAME=${1:-"test_matmul_unary"}

TEST_TPL_FILENAME=`echo ${TEST_FILENAME/test_/}`

echo "-- Write 'import ${TEST_FILENAME}' to __main__.py"
echo "import ${TEST_FILENAME}" > __main__.py


FILENAMES_ARRAY=(
    "index_code_gen_value_util"
    "index_drr_pass_util"
    "kernel_arg_translator_util"
    "index_program_translator_util"
    "low_level_ir_code_gen_ctx_util"
    "kernel_arg_id_util"
    "code_gen_value_util"
    "op_index_translator_util"
    "op_compute_translator_util"
    "program_translator_util"
    "__main__"
    "topo_drr_pass"
    "op_convertion_drr_pass"
    "access_topo_drr"
    "abstract_drr"
    "ap_tpl_codegen"
    "${TEST_FILENAME}"
    "${TEST_TPL_FILENAME}_tpl"
)
for filename in "${FILENAMES_ARRAY[@]}"
do
    echo "-- Convert ${filename}.py -> ${filename}.py.json"
    python ../../athena/advanced_pass/py_to_json.py ${filename}.py ${filename}.py.json
done
