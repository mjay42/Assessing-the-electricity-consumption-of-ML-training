"""Functions to process sensor data, retrieve wattmeter data for all experiments and integrate them into the dataset.

The functions can be used to merge the various result file and process them. 
The output of the file can be easily processed.
The results need to be synchronised with the experiment table (throught the experiment table).
    - add_benchmark_id_to_merged_timeseries().
The timestamps need to be synchronised: add_benchmark_id_to_merged_timeseries().
The energy time data need to be converted to same unit:
    - convert_energy_scope_timeseries_to_sec(), 
    - convert_bmc_to_sec(), 
    - convert_watt_to_sec().
For Energy Scope and the physical wattmeters, the total energy needs to be computed:
    - retrieve_wattmeter_data(), 
    - compute_energy_rm_sleep().
The results need to be cleaned in general:
    - set_repetition_to_ten(),
    - cleaning_table().

Typical use (on grid5000):
```
module load conda
conda activate gpu_benchmark
python utils/process_results.py --analysis_git_dir "/home/mjay/laion/pokemon/results" --expe_dir "/home/mjay/laion/pokemon/results" --result_folder "/home/mjay/GPU_benchmark_energy/results/night_exp_08_11/"
```
To merge Energy Scope and Wattmeter data of various experiments:

```
analysis_git_dir = "/home/mjay/GPU_benchmark_energy/" 
prefix = [analysis_git_dir + "results/night_exp_20_04/", analysis_git_dir + "results/night_exp_19_04/"]

sensor_df = pd.concat([pd.read_csv(file + 'es_ts.csv') for file in prefix])
watt_df = pd.concat([pd.read_csv(file + 'g5k_metrics.csv') for file in prefix])
exp_table = pd.concat([pd.read_csv(file + 'processed_table.csv') for file in prefix])

grouped_watt_df = watt_df.groupby(by=['timestamp_sec']).mean().reset_index()
merged_df = pd.merge(sensor_df,grouped_watt_df, on='timestamp_sec', how='outer').sort_values(by=['timestamp_sec'])
merged_df['wattmetre_es_diff'] = abs(merged_df['wattmetre_power_watt'] - merged_df['data.data.etotal(W)'])
b_df = add_benchmark_id_to_merged_timeseries(exp_table, merged_df)

b_df.plot(x='timestamp_sec', figsize=(20,15), linestyle=' ', marker='.')
plt.legend(bbox_to_anchor=(1.1, 1))
```

"""
import argparse
import datetime
import pandas as pd
import time
import numpy as np
import getpass
import os
import logging
from requests import auth
import sys
import tqdm

START_NAME = "start"
END_NAME = "end"

TIME_CONSTANTS = {
    "experiment" : {
        START_NAME:["STARTING EXPERIMENT AT ", "%Y-%m-%d %I:%M:%S %p"],
        END_NAME:["ENDING EXPERIMENT AT ", "%Y-%m-%d %I:%M:%S %p"]
        },
    "bench": {
        START_NAME:["BENCHMARK_TAG start_benchmark DATE ", "%Y/%m/%dT%H:%M:%S.%f"],
        END_NAME:["BENCHMARK_TAG stop_benchmark DATE ", "%Y/%m/%dT%H:%M:%S.%f"]
        },
    "inference":{
        START_NAME:["Starting inference at ", None],
        END_NAME:["Ending inference at ", None]
        }
}

ENERGY_COLUMNS = {
    "exp_without_sleep": 'exp_rm_sleep_energy_consumption(kWh)',
    "experiment" : 'exp_energy_consumption(kWh)',
    "bench": 'bench_energy_consumption(kWh)',
    "inference": 'inference_energy_consumption(kWh)'
}

def get_g5k_auth():
    user = getpass.getpass(prompt='Grid5000 login:')
    password = getpass.getpass(prompt='Grid5000 password:')
    return auth.HTTPBasicAuth(user, password)

def convert_paths(cell, analysis_dir, expe_dir):
    """Converting the table paths from expe_dir to analysis_dir."""
    if type(cell) is str and expe_dir in cell:
        return analysis_dir + '/' +  "/".join(cell.split("/")[3:])
    return cell

def convert_timestamp_to_sec(timestamp, pattern):
    """Return timestamp in seconds.
    
    examples of pattern:
    "%Y/%m/%d %H:%M:%S.%f"
    "%Y-%m-%dT%H:%M:%S+01:00"
    "%Y-%m-%dT%H:%M:%S.%f+02:00"
    "%Y-%m-%d %I:%M:%S %p"
    """
    return time.mktime(
        datetime.datetime.strptime(
            timestamp, 
            pattern
            ).timetuple())
    
def convert_wattmeter_timestamps_to_sec(row):
    try:
        return convert_timestamp_to_sec(row, "%Y-%m-%dT%H:%M:%S+02:00")
    except:
        return convert_timestamp_to_sec(row, "%Y-%m-%dT%H:%M:%S.%f+02:00")

def retrieve_wattmeter_data(Wattmeter, table): #, auth):
    """Get wattmetre data with http request for every experiment in table.
    
    args:
        table: DataFrames. Table describing experiments.
        auth: HTTP Authentification for http requests to the wattmeter database.
    """
    # retrieving wattmeter data afterwards
    full_df = pd.DataFrame()
    for table_id in table.index:
        start, stop = table.loc[table_id][['experiment_start', 'experiment_end']]
        wattmeter_data = Wattmeter(
            table.loc[table_id]['node'], 
            table.loc[table_id]['site'], 
            start - table.loc[table_id]['execution_script_args.sleep_before'], 
            stop + table.loc[table_id]['execution_script_args.sleep_after'],
            # auth,
            metrics=["wattmetre_power_watt"],
        )
        for metric in wattmeter_data.metrics:
            df = wattmeter_data.results[metric]['data']
            full_df = pd.concat([full_df, df], ignore_index=True)
    full_df['timestamp(sec)']=full_df['timestamp'].apply(convert_wattmeter_timestamps_to_sec)
    full_df['wattmetre_energy_consumption(kWh)'] = full_df["wattmetre_power_watt"]*10**(-3)/3600
    return full_df

def convert_sensor_timeserie_to_sec(serie):
    """Convert sensor timeseries to dataframes from table describing experiments.
    
    Les timestamps sont en miliseconds dans les csv.

    args:
        serie: DataFrames. serie describing experiments.

    returns:
        sensor_df: DataFrames. Table containing the timeseries concatenated.
    """
    gpu_df = pd.read_csv(serie["tool_csv_file_nvml"], sep=";")[8:]
    rapl_df = pd.read_csv(serie["tool_csv_file_rapl"], sep=";")
    sysinfo_df = pd.read_csv(serie["tool_csv_file_sysinfo"], sep=";")

    origin_timestamp = min(
        gpu_df["timestamp"].min(), 
        rapl_df["timestamp"].min(),
        sysinfo_df["timestamp"].min()
        )

    for df in [gpu_df, rapl_df, sysinfo_df]:
        df["timestamp(sec)"]=df["timestamp"]*10**(-3)
        df["timestamp_origin(sec)"]=(df["timestamp"]-origin_timestamp)*10**(-3)
        
    return gpu_df, rapl_df, sysinfo_df

def process_timeseries(table):
    """Creates energy scope timeseries from table describing experiments.
    
    Goes throught all experiments involving Energy Scope and builts
    its timeserie dataframe.
    
    args:
        table: DataFrames. Table describing experiments.

    returns:
        table: DataFrames. Table containing the timeseries concatenated.
    """
    sensor_df = pd.DataFrame()

    for index in tqdm.tqdm(table.index):
        serie = table.loc[index]
        if os.path.exists(serie.result_dir):
            gpu_df, rapl_df, sysinfo_df = convert_sensor_timeserie_to_sec(serie)
            energy_df = pd.concat([gpu_df, rapl_df, sysinfo_df])
            energy_df["exp_table_index"]=index
            energy_df["power(W)"]=energy_df["energy_consumption_since_previous_measurement_milliJ"]*10**(-3)/serie["period"] #J=Ws /s=W
            sensor_df = pd.concat([sensor_df, energy_df], ignore_index=True)
        else:
            logging.info("No result_dir found for index {index}", index)
            
    sensor_df["energy_consumption_since_previous_measurement(kWh)"]=sensor_df["energy_consumption_since_previous_measurement_milliJ"]*10**(-6)/3600
    return sensor_df

def process_stdout(result_dir):
    """Get times from stdouts.
    
    STARTING EXPERIMENT AT 2023-08-03 11:03:10 PM
    BENCHMARK_TAG start_benchmark DATE 2023/08/03T23:03:21.902820
    Starting inference at 1691096611.2322817
    Ending inference at 1691096619.1224568 
    BENCHMARK_TAG stop_benchmark DATE 2023/08/03T23:03:40.025295
    ENDING EXPERIMENT AT 2023-08-03 11:03:50 PM
    """
    file_path = result_dir + "stdout.txt"
    timestamps_sec = {}
    for col in TIME_CONSTANTS:
        timestamps_sec[col]={}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            for col in TIME_CONSTANTS:
                for start_or_stop in [START_NAME, END_NAME]:
                    line_pattern, date_pattern = TIME_CONSTANTS[col][start_or_stop]
                    if line_pattern in line:
                        date = line[len(line_pattern):-1]
                        timestamps_sec[col][start_or_stop] = convert_timestamp_to_sec(date, date_pattern) if date_pattern is not None else float(date)
    return timestamps_sec

def add_timestamps_from_stdout(table):
    """Modifies table by adding information from stdout.
    
    Returns the corrected description table.
    
    args:
        table: DataFrame. Table describing experiments.
        timeserie_df: DataFrame. Timeseries to compute energy from.
    
    returns:
        corrected description table: DataFrame
    """
    for index in tqdm.tqdm(table.index):
        # get exp, benchmark(, inference) start and stop time
        timestamps = process_stdout(table.loc[index]["result_dir"])
        for col in timestamps:
            if START_NAME in timestamps[col]:
                start_sec, end_sec = timestamps[col][START_NAME], timestamps[col][END_NAME]
                table.at[index, col+"_"+START_NAME+"(sec)"] = start_sec
                table.at[index, col+"_"+END_NAME+"(sec)"] = end_sec
    return table


def compute_energy(table, timeserie_df, timeserie_df_energy_kWh_column, timeserie_name):
    """Modifies table by computing energy from power timeseries.
    
    Requirements: 
    - execute add_timestamps_from_stdout() before.
    - timeserie_df_energy_kWh_column in kWh
    - timeserie_df has a column 'timestamp' in milli seconds and a column 'timestamp_sec' in sec
    
    Since experiments sometimes include several benchmarks, we need to compute the energy 
    for each benchmark and for each energy column.
    Returns the corrected description table.
    
    args:
        table: DataFrame. Table describing experiments.
        timeserie_df: DataFrame. Timeseries to compute energy from.
    
    returns:
        corrected description table: DataFrame
    """
    for index in tqdm.tqdm(table.index):
        # get exp, benchmark(, inference) start and stop time
        timestamps = process_stdout(table.loc[index]["result_dir"])
        for col in timestamps:
            if START_NAME in timestamps[col]:
                start_msec = table.loc[index, col+"_"+START_NAME+"(sec)"]
                end_msec = table.loc[index, col+"_"+END_NAME+"(sec)"]
                bench_df = timeserie_df[
                    (timeserie_df['timestamp(sec)']>=start_msec)&(timeserie_df['timestamp(sec)']<=end_msec)
                    ].reset_index(
                        drop=True
                    ).sort_values(
                        by=['timestamp(sec)']
                    )[['timestamp(sec)', timeserie_df_energy_kWh_column]].dropna()
                total_energy_kWh = bench_df[timeserie_df_energy_kWh_column].sum()
                # update the table
                table.at[index, timeserie_name + "_" + ENERGY_COLUMNS[col]] = total_energy_kWh
    return table

def compute_energy_rm_sleep(table, timeserie_df, timeserie_df_energy_kWh_column):
    """Modifies table by computing energy from power timeseries.
    
    Since experiments sometimes include several benchmarks, we need to compute the energy 
    for each benchmark and for each energy column.
    Returns the corrected description table.
    
    args:
        table: DataFrame. Table describing experiments.
        timeserie_df: DataFrame. Timeseries to compute energy from.
        table_energy_column: string. Name of the energy column to correct in table.
        timeserie_df_power_column: string. Name of the power column
            to compute the energy from.
        bench_ids: List of strings. Subset of benchmarks ids to process.
            None by default.
    
    returns:
        corrected description table: DataFrame
    """
    for index in tqdm.tqdm(table.index):
        # get benchmark start and stop time
        col_names = ['experiment_start', 'experiment_end', 'execution_script_args.sleep_before', 'execution_script_args.sleep_after']
        start, stop, sleep_before, sleep_after = table.loc[index][col_names].values
        # Retrieves power timeserie between the start and stop time
        start_msec = (start+sleep_before)*10**(3)
        end_msec = (stop-sleep_after)*10**(3)
        bench_df = timeserie_df[
            (timeserie_df['timestamp']>=start_msec)&(timeserie_df['timestamp']<=end_msec)
            ].reset_index(
                drop=True
            ).sort_values(
                by=['timestamp_sec']
            )[['timestamp_sec', timeserie_df_energy_kWh_column]].dropna()
        # Compute energy = power * duration between datapoints
        # bench_df['energy(Ws)'] = period*bench_df[timeserie_df_energy_column]
        total_energy_kWh = bench_df[timeserie_df_energy_kWh_column].sum()
        # update the table
        table.at[index, ENERGY_COLUMNS["exp_without_sleep"]] = total_energy_kWh
    return table

def get_hyperparameters(row):
    ls = row["execution_script_args.benchmark_execution"].split(' ')
    ls = [l for l in ls if l != '']
    for arg in ls:
        if '=' in arg:
            key, value = arg.split('--')[-1].split('=')
            row[key]=value
    return row

def process_table_add_energy_from_timeseries(exp_table, sensor_df, watt_df):
    """Process hyperparamters and data from energy df.
    
    timeseries must have a "timestamp(sec)" column.
    
    args:
        exp_table: DataFrame. Experiment table.
        watt_df: DataFrame. Wattmeter df.
        sensor_df: DataFrame. Energy Scope df.

    returns:
        exp_table: DataFrame. Updated experiment table.
    """
    logging.info("Correcting energy per benchmark")
    exp_table = exp_table.apply(get_hyperparameters, axis=1)
    for col in [
        "train_batch_size", 
        "gradient_accumulation_steps", 
        "max_train_samples", 
        "num_train_epochs"
        ]:
        if col in exp_table.columns:
            exp_table[col] = exp_table[col].astype(float, copy=True)
    exp_table["model_version"]=exp_table["pretrained_model_name_or_path"].apply(lambda x: x[-4:])
    
    exp_table = compute_energy(
        exp_table, sensor_df, 
        'energy_consumption_since_previous_measurement(kWh)',
        "sensor"
        )
    
    sensor_drop = sensor_df.drop(sensor_df[sensor_df["device_index"].isin(range(1,8))].index)
    exp_table = compute_energy(
        exp_table, sensor_drop, 
        'energy_consumption_since_previous_measurement(kWh)',
        "sensor_gpu0"
        )
    
    exp_table = compute_energy(
        exp_table, watt_df, 
        'wattmetre_energy_consumption(kWh)',
        "wattmetre"
        )
    return exp_table
    
def parsing_arguments():
    parser = argparse.ArgumentParser(description='Process results')
    parser.add_argument('--analysis_git_dir', 
                        help='Path to the local git repository.', 
                        type=str)
    parser.add_argument('--expe_dir', 
                        help='Path to the directory in gemini.', 
                        type=str, default='/root/energy-consumption-of-gpu-benchmarks/')
    parser.add_argument('--result_folder', 
                        help='Path to the result folder.', 
                        type=str)
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parsing_arguments()

    analysis_git_dir = args.analysis_git_dir # "/Users/mathildepro/Documents/code_projects/GPU_benchmarks/" #
    expe_dir = args.expe_dir #"/root/energy-consumption-of-gpu-benchmarks/"
    #analysis_git_dir = "/home/mjay/GPU_benchmark_energy/" 
    prefix = args.result_folder # analysis_git_dir + "results/night_exp_08_11/"
    experiments_table = prefix + "/experiment_table.csv"
    merged_table_csv_file = prefix + "/processed_table.csv"
    es_timeseries_csv_file = prefix + '/timeseries.csv'
    wattmeter_csv_file = prefix + '/g5k_metrics.csv'
    stdout_file = prefix + '/stdout.txt'
    
    sys.path.append(analysis_git_dir+"/utils/")

    from tools import Wattmeter
    
    # Experiment metadata
    logging.info("Retrieving data from " + experiments_table)
    exp_table = pd.read_csv(experiments_table)
    logging.info(exp_table['result_dir'].unique()[:5])
    
    if "error" in exp_table.columns:
        exp_table = exp_table[exp_table["error"].isna()].reset_index(drop=True)

    # Modify the table if the location of the directory has changed
    # Typically when the experiments were done in grid5000 and the analysis locally 
    # however the fct convert_paths needs to be modify depending on your case
    # logging.info("File directory "+ os.path.dirname(os.path.abspath(__file__)))
    # exp_table = exp_table.applymap(
    #     lambda cell: convert_paths(cell, analysis_git_dir, expe_dir))
    # logging.info(exp_table['result_dir'].unique()[:5])
    
    exp_table = add_timestamps_from_stdout(exp_table)
    # exp_table.to_csv(merged_table_csv_file, index=False)
    
    # Getting timeseries from tool result files
    logging.info("Retrieving sensor data")
    sensor_df = process_timeseries(exp_table)
    logging.info(sensor_df.head())
    sensor_df.to_csv(es_timeseries_csv_file, index=False)
    
    # sensor_df = pd.read_csv(es_timeseries_csv_file)

    # This takes too much time so it is discarded
    
    # exp_table = pd.read_csv(merged_table_csv_file)
    
    # # Retrieving wattmeter data
    # logging.info("Retrieving wattmeter data")
    # # auth = get_g5k_auth()
    watt_df = retrieve_wattmeter_data(Wattmeter, exp_table)
    logging.info(watt_df.head())
    watt_df.to_csv(wattmeter_csv_file, index=False)
    # 
    watt_df = pd.read_csv(wattmeter_csv_file)
    
    # Correct total energy per benchmark
    exp_table = process_table_add_energy_from_timeseries(exp_table, sensor_df, watt_df)
    
    exp_table.to_csv(merged_table_csv_file, index=False)
    
    
    
if __name__ == '__main__':
    main()