#!/bin/bash
cd build_out/
echo "install .run-----------------------------"
./custom_opp_euleros_aarch64.run --install-path=/home/ma-user/Ascend/ascend-toolkit/latest/
cd ../../case_910b/PdistGrad
echo "compile whl-----------------------------"

python3 setup.py build bdist_wheel
echo "install whl-----------------------------"

pip3 install dist/*.whl --force-reinstall
echo "source env-----------------------------"

source /home/ma-user/Ascend/ascend-toolkit/latest/vendors/customize/bin/set_env.bash

 