#!/bin/bash

FILENAME=${1:-"test_trivial_reduce"}
#FILENAME=${1:-"test_matmul_unary"}

TPL_FILENAME=`echo ${FILENAME/test_/}`

python3 ../../athena/advanced_pass/py_to_json.py __main__.py __main__.py.json

echo "-- Convert access_topo_drr.py -> access_topo_drr.py.json"
python ../../athena/advanced_pass/py_to_json.py access_topo_drr.py access_topo_drr.py.json

echo "-- Convert abstract_drr.py -> abstract_drr.py.json"
python ../../athena/advanced_pass/py_to_json.py abstract_drr.py abstract_drr.py.json

echo "-- Convert ap_tpl_codegen.py -> ap_tpl_codegen.py.json"
python ../../athena/advanced_pass/py_to_json.py ap_tpl_codegen.py ap_tpl_codegen.py.json

echo "-- Convert ${FILENAME}.py -> ${FILENAME}.py.json"
python ../../athena/advanced_pass/py_to_json.py ${FILENAME}.py ${FILENAME}.py.json

echo "-- Convert ${TPL_FILENAME}_tpl.py -> ${TPL_FILENAME}_tpl.py.json"
python ../../athena/advanced_pass/py_to_json.py ${TPL_FILENAME}_tpl.py ${TPL_FILENAME}_tpl.py.json
