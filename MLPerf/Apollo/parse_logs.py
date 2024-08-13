"""Extract hyperparameters and energy consumption from logs.

This script extracts hyperparameters and energy consumption 
from logs of experiments done at HPE on Champollion.
Extracted data from log_dir is saved in csv files in save_dir.

Expect an execution time from 1 to 10 minutes depending on the number of jobs.

Example:
    $ cd ML_benchmark
    $ python parse_logs.py 
        --model bert 
        --log_dir "/nfs/matjay/github/training2.1/HPE/benchmarks/bert/implementations/logs" 
        --save_dir "/nfs/matjay/github/training2.1/HPE/benchmarks/bert/implementations/processed_logs"
        --TimeSeries
        
    $ python parse_logs.py \
        --model maskrcnn  \
        --log_dir "/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov/maskrcnn/"  \
        --save_dir "/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov_processed/maskrcnn/" \
        --TimeSeries
"""
import sys
sys.path.append("./logging/mlperf_logging/")

from compliance_checker.mlp_parser import parse_file
import dask.dataframe as dd
import glob
import pandas as pd
import datetime
import time
import re
import argparse
import logging
from tqdm import tqdm

CONSTANTS = {
    "srun_start_time_ms": "Launching srun at (.*)M",
    "start_time_ms": "STARTING TIMING RUN AT (.*)M",
    "end_time_ms": "ENDING TIMING RUN AT (.*)M",
}

MASKRCNN_CONSTANTS = {
    "training_start_time_ms": "STARTING train_mlperf.py AT (.*)M",
    "end_time_ms": "ENDING train_mlperf.py AT (.*)M",
}

UNET_CONSTANTS = {}

DLRM_CONSTANTS = {}

RESNET_TIME_CONSTANTS = {
    "training_start_time_ms": "STARTING train_imagenet.py AT (.*)M",
    "train_end_time_ms": "ENDING train_imagenet.py AT (.*)M",
}

RNNT_TIME_CONSTANTS = {
    "training_start_time_ms": "STARTING train.py AT (.*)M",
    "train_end_time_ms": "ENDING train.py AT (.*)M",
}

BERT_TIME_CONSTANTS = {
    "training_start_time_ms": "STARTING run_pretraining.py AT (.*)M",
    "train_end_time_ms": "ENDING run_pretraining.py AT (.*)M",
}

TIME_CONSTANTS = {
    "bert": BERT_TIME_CONSTANTS,
    "resnet": RESNET_TIME_CONSTANTS,
    "rnnt": RNNT_TIME_CONSTANTS,
    "unet": UNET_CONSTANTS,
    "dlrm": DLRM_CONSTANTS,
    "maskrcnn": MASKRCNN_CONSTANTS,
}

ML_TIME_CONSTANTS = {
    "run": {"start": "run_start", "stop": "run_stop"},
    "init": {"start": "init_start", "stop": "init_stop"},
    "time": {"start": "start_time_ms", "stop": "end_time_ms"}
}


def get_msec(date_time, pattern="%Y-%m-%d %I:%M:%S %p"):
    """Convert date time to seconds."""
    return time.mktime(
        datetime.datetime.strptime(
            date_time,
            pattern
        ).timetuple())


def search_cmd_params(line, job, cmd_found, job_hyper_arg):
    """Apply regex for explicit command in log line."""
    if ("BERT_CMD=" in line or "train.py --" in line) and not cmd_found:
        cmd_found = True
        for arg in line.split(" "):
            if "--" in arg and "=" in arg:
                key, value = arg.split("--")[1].split("=")
                job_hyper_arg["cmd_" + key] = value
        logging.info("found cmd params in job %s", job)
    return job_hyper_arg, cmd_found


def search_hyper_params(line, job, param_found, job_hyper_arg):
    """Apply regex for explicit hyperparameters in log line."""
    param_str = "(?<=Hyper-parameters: ).*"
    if re.search(param_str, line) and not param_found:
        param_found = True
        regex = re.findall(param_str, line)[0]
        # regex = line[32:]
        for arg in regex.split(" "):
            key, value = arg.split("=")
            if len(value) > 0 and value[-1] == "'":
                value = value[:-1]
            job_hyper_arg[f"sub_{key}"] = value
        logging.info("found run.hpe.sub params in job %s", job)
    return job_hyper_arg, param_found


def get_value(row):
    """Get key value pairs from mllog rows."""
    for key, value in row['metadata'].items():
        row["meta_"+key] = value
    return row


def parse_mllogs(file, job_hyper_arg):
    """Parse mllog from log file."""
    loglines, _ = parse_file(file, ruleset="2.1.0")
    mllog = pd.DataFrame(loglines)
    mllog['metadata'] = mllog.apply(
        lambda row: row['value']['metadata'], axis=1)
    mllog['value'] = mllog.apply(lambda row: row['value']['value'], axis=1)
    mllog = mllog.apply(get_value, axis=1)

    job_hyper_arg["epoch_nb"] = len(mllog[mllog["key"] == "epoch_start"])
    job_hyper_arg["block_nb"] = len(mllog[mllog["key"] == "block_start"])
    job_hyper_arg["eval_nb"] = len(mllog[mllog["key"] == "eval_accuracy"])

    mllog_params = mllog[(mllog.value.notna())][["key", "value"]].set_index(
        'key').T.to_dict('list')
    mllog_params_update = {}
    for key, value in mllog_params.items():
        mllog_params_update[f"mllog_{key}"] = value[0]
    job_hyper_arg.update(mllog_params_update)

    mllog_timeserie = mllog[(mllog.value.isna()) & (
        mllog.key != "weights_initialization")][["key", "timestamp"]]

    return job_hyper_arg, mllog_timeserie


def parse_logs(lines, hyper_arg, job, file, constants):
    """Parse logs from a job.

    Parses logs and look for hyperparameters from commands and mllogs.
    checks if logs are ending with RESULT.
    """
    job_hyper_arg = {}
    job_hyper_arg["file"] = file

    # start_time = lines[-8].split(",")[-1][:-1]
    # start_time_sec = get_msec(start_time)
    # job_hyper_arg["start_time_ms"]=start_time_sec*(10**3)
    # logging.info(f"found timing in job {job}: {start_time_sec}")

    # get hyperparameters
    param_found = False
    cmd_found = False
    for line in lines:
        job_hyper_arg, cmd_found = search_cmd_params(
            line, job, cmd_found, job_hyper_arg)

        job_hyper_arg, param_found = search_hyper_params(
            line, job, param_found, job_hyper_arg)

        for col in constants:
            if col in job_hyper_arg:
                continue
            job_hyper_arg = parse_time(
                constants[col], line, job_hyper_arg, col)

    job_hyper_arg, mllog_timeserie = parse_mllogs(file, job_hyper_arg)

    # if went through every line without break, then save results
    hyper_arg[job] = job_hyper_arg
    mllog_timeserie["job"] = job
    return hyper_arg, mllog_timeserie


def parse_time(pattern, line, df, key):
    """Parse time from logs."""
    if re.search(pattern, line):
        regex = re.findall(pattern, line)
        end_time = [x+"M" for x in regex][0]
        # this is in seconds
        end_time_sec = get_msec(end_time)
        df[key] = end_time_sec*(10**3)
    return df


def get_error(f, errors, job):
    """Look for error message in logs. 
    Returns error message at first error found.
    """
    for line in f:
        if "CANCELLED" in line or "error" in line or "Error" in line:
            logging.info("error in job %s", str(job))
            errors[job] = line
            return errors
    return errors


def get_metadata(model, log_dir, constants, start_log=0):
    """Extract hyperparameters from logs.

    Parses logs from log_dir. Check if logs are ending with RESULT.
    if no error found, extracts hyperparameter.

    Returns 
    - a dataframe with hyperparameters
    - a dataframe with mllog timeseries
    - a dataframe with erros.
    """
    jobs = [path.split("/")[-1] for path in glob.glob(log_dir+"/*")]
    hyper_arg = {}
    mllog_df = pd.DataFrame()
    errors = {}

    for job in tqdm(jobs):
        if int(job) < start_log:
            continue
        logging.info("Processing job %s", job)
        files = glob.glob(log_dir+"/"+job+"/*.log")

        several_logs = (model == "unet" or model == "dlrm")
        if several_logs:
            for file in files:
                if (model == "unet" and "unet3d" in file) or (model == "dlrm" and "raw" not in file):
                    files.remove(file)

        if len(files) < 1:  # len(files) > 1 or
            errors[job] = f"Error: Number of log files incorrect: {len(files)}"
            continue
        file = files[0]
        job = int(job)

        with open(file, 'r') as f:  # encoding='latin-1'
            # process result line
            lines = f.readlines()
            result = False
            for line in lines[-8:]:
                if "RESULT" in line:
                    result = True
                    logging.info("RESULT found in job %s", str(job))
                    break

            # check if execution ending well, and look for error message if it's the case
            if not result:
                errors[job] = "Not ending with RESULT"
                errors = get_error(f, errors, job)
                continue

            # if everything went well, parse remaining logs
            hyper_arg, mllog = parse_logs(
                lines, hyper_arg, job, file, constants)

        mllog_df = pd.concat([mllog_df, mllog])

    hyper_arg_df = pd.DataFrame(hyper_arg).T.reset_index().rename(
        columns={"index": "job_id"})

    return hyper_arg_df, mllog_df, errors


def preprocess(df):
    """Convert energy data to joule and kWh."""
    df["energy_joule"] = df["energy_consumption_since_previous_measurement_milliJ"] * \
        10**(-3)
    df["energy_kWh"] = df["energy_joule"]*10**(-3)/3600
    return df


def parse_monitor_files(log_dir, jobs_df, csv_pattern="-nvml.csv", update_jobs=True, verbose=1):
    """Add energy consumption to hyperparameter df.

    Parse csv files of energy consumption for each node 
    of each job and concatenate them with hyperparameters from jobs_df.
    Removes first 8 lines of nvml csv files.
    Rectify number of nodes in jobs_df if update_jobs=True.
    """
    gpu_df = pd.DataFrame({})
    # for all jobs in log dir
    for job in tqdm(jobs_df["job_id"]):
        node_count = 0
        if verbose > 1:
            logging.info("## JOB ID: %s", job)
        # go through all nodes
        for node_log_path in glob.glob(log_dir+"/"+str(job)+"/o*"):
            node_count += 1
            node = node_log_path.split("/")[-1]
            # look for demanded csv file
            files = glob.glob(log_dir+"/"+str(job)+"/"+node+"/*"+csv_pattern)
            if len(files) > 0:
                file = files[0]
            else:
                logging.error(
                    f"no matching to {csv_pattern} in "+log_dir+"/"+str(job)+"/"+node)
                continue
            # load data
            new_df = pd.read_csv(file, sep=";")
            if csv_pattern == "-nvml.csv":
                new_df = new_df[8:]
            # add hyperparametres
            new_df["node"] = node
            new_df["job"] = job
            gpu_df = pd.concat([gpu_df, new_df])
        if verbose > 1:
            logging.info("Number of nodes: %s", node_count)
        if update_jobs:
            mask = jobs_df["job_id"] == job
            jobs_df.loc[mask, "node_nb"] = node_count
    return gpu_df, jobs_df


def compute_sec_since_start(row, jobs):
    """Compute seconds since start of job."""
    timestamp = row["timestamp"]
    job = row["job"]
    origin = jobs[jobs["job_id"] == job]["start_time_ms"].values[0]
    res = timestamp - origin
    return res


def compute_global_summary(
        logs_dir="/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov",
        processed_logs_dir="/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov_processed",
        ml_keys_time=['run_start', 'run_stop', 'init_start', 'init_stop'],
        keeped_cols=['model', 'job', 'energy_kWh', 'energy_joule', 'energy_consumption_since_previous_measurement_milliJ',
                     'sub_seed', 'start_time_ms', 'end_time_ms', 'mllog_train_samples', 'mllog_eval_samples', 'node_nb']
):
    """Compute global summary statistics of all models. Save it at processed_logs_dir/summary.csv.

    The processing being specific to my measurements, this function is not generalizable and 
    would require modifications (specifically lines 11-13).

    Args:
        logs_dir (str, optional): Directory path where the log files are stored. 
            Defaults to "/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov".
        processed_logs_dir (str, optional): Directory path where the processed log files will be saved. 
            Defaults to "/Users/mathildepro/Documents/code_projects/hpe_logs/resultats_nov_processed".
        ml_keys_time (list, optional): List of keys in mllog to be used for the analysis. 
            Defaults to ['run_start', 'run_stop', 'init_start', 'init_stop'].
        keeped_cols (list, optional): List of columns to keep in the summary. Defaults to [

    Returns:
        DataFrame: Summary statistics of all models.
    """
    models = ["bert", "dlrm", "maskrcnn", "resnet", "rnnt", "unet"]

    summaries = []
    for model in models:
        save_dir = f"{processed_logs_dir}/{model}/"
        job_file = save_dir+"/summary.csv"
        summary = pd.read_csv(job_file).drop(columns=["Unnamed: 0"])
        summary["model"] = model
        summaries.append(summary)
    summarydf = pd.concat(summaries)

    summarydf = summarydf[~summarydf["job"].isin(
        [126534, 126536, 126537, 126538, 126990])]
    summarydf = summarydf[~summarydf["job"].isin(
        [126279, 126280, 126281, 126282, 126283])]
    summarydf = summarydf[(summarydf["node_nb"] == 1) & (
        summarydf["mllog_gradient_accumulation_steps"] == 1)]

    mldf = []

    # Get information from mllog
    for model in models:
        path = f"{logs_dir}/{model}/"
        jobs = summarydf[summarydf["model"] == model]
        for job in jobs["job"].unique():
            file = jobs[jobs["job"] == job]["file"].values[0]
            log_file = path + "/".join(file.split("/")[-2:])

            loglines, _ = parse_file(log_file, ruleset="2.1.0")
            mllog = pd.json_normalize(pd.DataFrame(loglines)["value"])
            mllog = mllog.merge(pd.DataFrame(loglines)[
                                ["timestamp", "key"]], left_index=True, right_index=True)
            mllog["job"] = job
            mldf.append(mllog)
    mldf = pd.concat(mldf)
    mldf = mldf.merge(summarydf, on="job", how="left")

    # Only keep important columns
    summarydf = summarydf[keeped_cols]

    # add to summarydf chosen timestamps of the mllog interesting to the analysis
    for model in models:
        for job in mldf[mldf["model"] == model]["job"].unique():
            mllog = mldf[(mldf["model"] == model) & (mldf["job"] == job)]
            index = summarydf[(summarydf["model"] == model) & (
                summarydf["job"] == job)].index.values[0]
            for key in ml_keys_time:
                key_mllog = mllog[mllog["key"] == key]["timestamp"]
                summarydf.at[index, key] = key_mllog.values[0]
    summarydf = summarydf.reset_index(drop=True)

    # compute energy within the chosen timestamps
    for model in models:
        save_dir = f"{processed_logs_dir}/{model}/"
        energy = dd.read_parquet(save_dir+'energy.parquet', engine='pyarrow')
        for job in summarydf[summarydf["model"] == model]["job"].unique():
            index = summarydf[(summarydf["model"] == model) & (
                summarydf["job"] == job)].index.values[0]
            energy_job = energy[energy["job"] == job]
            ilo_path = f"{logs_dir}/{model}/{job}/o186i225/ilo_power.csv" 
            ilo = pd.read_csv(ilo_path)
            ilo_energy = (ilo["power_watt"] * ilo["timestamp"].diff() * 1e-3/3600).sum()
            summarydf.loc[index, "ILO_energy_kWh"] = ilo_energy
            for time_key, time_value in ML_TIME_CONSTANTS.items():
                # RAPL & NVML energy
                energy_time_df = energy_job[
                    (
                        energy_job["timestamp"] >= summarydf.loc[index,
                                                                 time_value["start"]]
                    ) & (
                        energy_job["timestamp"] <= summarydf.loc[index,
                                                                 time_value["stop"]]
                    )]
                energy_joule = energy_time_df["energy_consumption_since_previous_measurement_milliJ"].sum(
                ).compute()*1e-3
                energy_kWh = energy_joule*1e-3/(60*60)
                summarydf.loc[index, f"energy_kWh_{time_key}"] = energy_kWh

                # CPU & GPU utilization
                cpu_avg_util = energy_time_df["utilization_percent"].mean(
                ).compute()  # .groupby(["domain", "socket"])
                summarydf.loc[index,
                              f"cpu_utilization_{time_key}"] = cpu_avg_util
                gpu_avg_util = energy_time_df[~energy_time_df["device_index"].isna(
                )]["global_utilization_percent"].mean().compute()  # .groupby(["device_index"])
                summarydf.loc[index,
                              f"gpu_utilization_{time_key}"] = gpu_avg_util
                
                # ILO energy
                ilo_time_df = ilo[
                    (
                        ilo["timestamp"]*1e3 >= summarydf.loc[index, time_value["start"]]
                    ) & (
                        ilo["timestamp"]*1e3 <= summarydf.loc[index, time_value["stop"]]
                    )]
                ilo_energy = (ilo_time_df["power_watt"] * ilo_time_df["timestamp"].diff() * 1e-3/3600).sum()
                summarydf.loc[index, f"ILO_energy_kWh_{time_key}"] = ilo_energy
    
    for time_key, time_value in ML_TIME_CONSTANTS.items():
        summarydf[f"duration_{time_key}(min)"] = (summarydf[time_value["stop"]] - summarydf[time_value["start"]])/1000/60
    
    summarydf.to_csv(f"{processed_logs_dir}/summary.csv")
    return summarydf


def process_logs_main():
    args = parsing_arguments()
    constants = TIME_CONSTANTS[args.model] | CONSTANTS
    log_dir = args.log_dir
    save_dir = args.save_dir

    logging.info("Start processing logs for every jobs at %s", log_dir)
    jobs, mllog_df, errors = get_metadata(args.model, log_dir, constants)
    logging.info("Processed %d logs", len(jobs))
    mllog_df.to_csv(f"{save_dir}/mllogs.csv")
    logging.info("MLLOGS saved at %s/energy.csv", save_dir)

    logging.info(
        "Start processing energy and monitoring files for every jobs.")
    jobs["node_nb"] = 1
    gpu_df, jobs = parse_monitor_files(log_dir, jobs, verbose=1)
    print(gpu_df)
    rapl_df, _ = parse_monitor_files(
        log_dir, jobs, "-rapl.csv", update_jobs=False)
    sysinfo_df, _ = parse_monitor_files(
        log_dir, jobs, "-sysinfo.csv", update_jobs=False)
    energy_df = pd.concat([gpu_df, rapl_df, sysinfo_df])
    energy_df = preprocess(energy_df)
    energy_df.to_parquet(save_dir+'energy.parquet', engine='pyarrow')
    logging.info(
        "Concat of energy csv done and saved at %s/energy.parquet", save_dir)

    logging.info("Start merging hyper parameters and energy data for job %s.",
                 energy_df["job"].values[1])
    energy_df = dd.read_csv(
        save_dir+'energy.parquet',
        dtype={'cpu': 'object', 'socket': 'object', 'domain': 'object'}).drop(columns=["Unnamed: 0"])
    energy_job = energy_df.groupby("job").sum()[
        ["energy_kWh", "energy_joule",
            "energy_consumption_since_previous_measurement_milliJ"]
    ].compute().reset_index()
    jobs_merged = energy_job.merge(
        jobs, right_on="job_id", left_on="job", how="outer")
    jobs_merged.to_csv(save_dir+"/summary.csv")
    logging.info(
        "Merged jobs and energy data and saved it at %s/summary.csv", save_dir)


def parsing_arguments():
    parser = argparse.ArgumentParser(description='Process results')
    parser.add_argument('--model',
                        help='Model.',
                        type=str)
    parser.add_argument('--log_dir',
                        type=str)
    parser.add_argument('--save_dir',
                        type=str)
    parser.add_argument('--TimeSeries',
                        help='Whether to process timeseries or not.',
                        action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    process_logs_main()
