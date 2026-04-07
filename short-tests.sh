#!/bin/bash
test_name="test_$(date +"%Y-%m-%d_%H-%M-%S")"
echo "$test_name"
set -e

if [ ! -n "$CUDA_INSTALL_PATH" ]; then
	echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
	exit;
fi

#Make the simulator
export PATH=$CUDA_INSTALL_PATH/bin:$PATH;
source ./gpu-simulator/setup_environment.sh
make clean -C ./gpu-simulator
make -j -C ./gpu-simulator

#Get the pre-run trace files
# rm -rf ./hw_run/rodinia_2.0-ft
# wget https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz
# mkdir -p ./hw_run
# tar -xzvf rodinia_2.0-ft.tgz -C ./hw_run
#rm rodinia_2.0-ft.tgz

#Run the tests on the trace
./util/job_launching/run_simulations.py -C QV100-SASS -B rodinia_2.0-ft -T ./hw_run/rodinia_2.0-ft/9.1 -N $test_name

# Wait for them to finish
./util/job_launching/monitor_func_test.py -v -N $test_name

./util/job_launching/get_stats.py -N $test_name | tee stats_$test_name.csv

