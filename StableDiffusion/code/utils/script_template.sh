#!/bin/bash
INTERNAL_SENSOR_DIR=
# ILO_SENSOR_SCRIPT=
RESULT_DIR=
PERIOD=
benchmark_execution=
sleep_before=
sleep_after=
# bash ${ILO_SENSOR_SCRIPT} ${RESULT_DIR} &
# ilo_pid=$!
# echo "Ilo sensor running with pid $ilo_pid"
echo ${INTERNAL_SENSOR_DIR}/target/release/nvml_sensor
sudo ${INTERNAL_SENSOR_DIR}/target/release/nvml_sensor --result-dir ${RESULT_DIR} --period-seconds ${PERIOD} &
sensor_pid=$!
echo "Intenal sensors running with pid $sensor_pid"
sleep $sleep_before
echo "BENCHMARK_TAG start_benchmark DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')"
$benchmark_execution
echo "BENCHMARK_TAG stop_benchmark DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')"
sleep $sleep_after
# kill $ilo_pid
# kill $sensor_pid
sudo pkill nvml_sensor