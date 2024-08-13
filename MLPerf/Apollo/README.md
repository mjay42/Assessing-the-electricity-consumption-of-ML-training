# Assessing the electricity consumption of the MLPerf benchmark on the HPE champollion cluster

This repository contains the code for launching the experiments, processing the logs, and analyzing the results.
It is based on the code used by HPE to conduct MLPerf benchmark on Champollion in April 2022.
It was adapted to measure the electricity consumption during the experiments with iLO and Alumet/nvml_sensor.

To summarize, below are the main modifications that were made to each model repository, in the "run_and_time.sh" script:
```
# monitoring RAPL and NVIDI-SMI
srun --overlap -l -n${SLURM_JOB_NUM_NODES} bash -c 'echo -n "Creating directory at " && hostname && pwd && mkdir -p ${LOGDIR}/$(hostname)/'
srun --overlap -l -n${SLURM_JOB_NUM_NODES} bash -c 'echo -n "Starting energy monitoring of ilo in " && hostname && bash ./mxnet/get_power.sh ${LOGDIR}/$(hostname)/ilo_power.csv' &
srun --overlap -l -n${SLURM_JOB_NUM_NODES} bash -c 'echo -n "Starting energy monitoring of RAPL and nvidia-smi in " && hostname && ./mxnet/nvml_sensor --result-dir ${LOGDIR}/$(hostname)/ --period-seconds 0.5' &
```

Timestamps and relevant hyperparameters were added too. MLPerf only cares about performances like throughput and total run time while studying electricity consumption requires additional data collection.

To start training, you need access to the Champollion cluster and execute the following command, as an example for the BERT model:
```
./ennrot.bert.pfss.sh NUMBER_NODES
```

The analysis can be done on a different machine, but requires parsing:
```
python parse_logs.py 
    --model bert 
    --log_dir "bert/implementations/logs" 
    --save_dir "bert/implementations/processed_logs"
    --TimeSeries
```
