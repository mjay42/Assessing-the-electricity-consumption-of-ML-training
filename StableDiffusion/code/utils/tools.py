"""How to use the software wattmeters or as called here, the power tools.

The Wattmeter class allows to retrieve wattmeter data.
The PowerTool class is a skeleton for every Tool class.
    It contains a method to start the tools with the benchmarks in various ways: 
        - with the benchmark in parallel,
        - the tool monitoring one benchmark or all of them.
    The results are processed depending on the tool outputs.

Typical use:

tool = EnergyScope("./")
bench_timestamps, output = tool.launch_experiments(
    benchmarks, # Benchmark object
    False,
)
print(tool.process_power_results())

# ...
wattmeter_data = Wattmeter(
    "gemini", 
    "lyon", 
    2022-04-20T17:49:26+02:00, 
    2022-04-20T18:49:26+02:00,
    _,
)
print(wattmeter_data.results["wattmetre_power_watt"]['energy_kWh'])
"""
import datetime
import json
import logging
from multiprocessing import Pool
import os
import time
from typing import Tuple, List
import glob
import subprocess

import requests
# you may need to install requests for python3 with sudo-g5k apt install python3-requests
import pandas as pd
import numpy as np

def merge_dict_with_identical_keys(dicts):
    new_dict = {}
    for key in dicts[0].keys():
        new_dict[key]=[]
        for dico in dicts:
            if type(dico[key])==list:
                new_dict[key] = new_dict[key] + dico[key]
            else:
                new_dict[key].append(dico[key])
    return new_dict  
    
def convert_energy_joule_in_kWh(energy_joule: float) -> float:
    """Converts joule in kWh"""
    return energy_joule/3600*10**(-3)

def convert_energy_kWh_in_joules(energy_kWh: float) -> float:
    """Converts kWh in Joules"""
    return energy_kWh*3600*10**(3)

def compute_time_serie_energy_joule(serie, interval) -> float:
    """Returns total energy in Joule. 
    
    args:
        serie: List. Time serie of power values in watt
        interval: float. Time interval in seconds
    """
    return serie.sum() * interval # in Joule
    
class Wattmeter:

    def __init__(
        self, 
        node: str, 
        site: str, 
        start: float, 
        stop: float, 
        g5k_auth = None,
        metrics: List[str] = ["wattmetre_power_watt", "bmc_node_power_watt", "pdu_outlet_power_watt"], 
        margin=30,
        ) -> None:

        self.node = node
        self.site = site
        self.start = start
        self.stop = stop
        self.metrics = metrics
        self.g5k_auth = g5k_auth
        self.plot_start, self.plot_stop = self.process_timestamps(margin=margin)
        self.energy_start, self.energy_stop = self.process_timestamps()
        self.results = {}
        raw_data = self.retrieve_power()
        for metric in metrics:
            self.results[metric]={}
            self.results[metric]['plot_data'] = self.process_power(metric, raw_data)
            self.results[metric]['data'] = self.results[metric]['plot_data'][
                (
                    self.results[metric]['plot_data']['timestamp'] < self.energy_stop
                )&(
                    self.results[metric]['plot_data']['timestamp'] > self.energy_start
                )
            ]
            self.results[metric]['energy_joule'] = compute_time_serie_energy_joule(
                self.results[metric]['data'][metric].values, 1)
            self.results[metric]['energy_kWh'] = convert_energy_joule_in_kWh(self.results[metric]['energy_joule'])
    
    def process_timestamps(self, margin=0) -> Tuple[str, str]:
        request_start = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.start - margin))
        request_stop = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.stop + margin))
        return request_start, request_stop

    def retrieve_power(self) -> List[dict]:
        """
        one dictionnary for every power data point. One data point every seconds.
        returns list of dictionnaries.
        """
        url = "https://api.grid5000.fr/stable/sites/%s/metrics?metrics=%s&nodes=%s&start_time=%s&end_time=%s" \
                % (self.site, ','.join(self.metrics), self.node, self.plot_start, self.plot_stop)
        logging.info(url)
        if self.g5k_auth is not None:
            return requests.get(url, auth=self.g5k_auth, verify=False).json()  # 
        else:
            return requests.get(url, verify=False).json() 


    def process_power(self, metric: str, raw_data: dict) -> pd.DataFrame:
        """
        Converts list of dict to dataframe 
        
        Format of raw data: List of dict:
            {
            'timestamp': '2022-02-25T09:02:15.40325+01:00', 
            'device_id': 'chifflet-7', 
            'metric_id': 'bmc_node_power_watt', 
            'value': 196, 
            'labels': {}
            }
        
        returns:
            df with columns "timestamp", "value"
        """
        timestamps = np.array([d['timestamp'] for d in raw_data if d['metric_id']==metric])
        values = np.array([d['value'] for d in raw_data if d['metric_id']==metric])
        dict_for_df = {'timestamp': timestamps, metric:values}
        return pd.DataFrame(dict_for_df)
    
    def save_power(self, result_dir: str) -> None:
        for metric in self.metrics:
            path = result_dir+metric+".csv"
            self.results[metric]["csv_file"]=path
            self.results[metric]['data'].to_csv(path, index=False)


class PowerTool:
    """Super class for power tool objects.

    Because all should have in commun:
    - a directory to save the results in
    - a method to process results that return a csv file or a data frame
    - a method to be launched (alongside a list of benchmarks)

    Attributes:
        result_dir: A string of the absolute path of the folder to save the 
            results in.
    """
    def __init__(
        self, 
        result_dir: str,
        tool_name: str,
        ) -> None:
        self.result_dir = result_dir
        self.tool_name = tool_name
    
    def process_stdout(self, output):
        res = {}
        for line in output.split('\n'):
            if 'BENCHMARK_TAG' in line:
                # read line
                info = line.split(' ')
                start_or_stop = info[1]
                date = info[3]
                # add to dict
                key = "{}".format(start_or_stop)
                if key not in res.keys():
                    res[key]=[]
                res[key].append(
                    datetime.datetime.strptime(
                        date, '%Y/%m/%dT%H:%M:%S.%f'
                    ).timestamp()
                )
        return res
    
    def process_power_results(self):
        """Returns whatever power information given by tool"""
        return {}

class Sensor(PowerTool):
    """Implementation of PowerTool for Sensor."""
    def __init__(
        self,
        result_dir: str,
        source_dir: str,
        period: float,
    ) -> None: 
        super().__init__(result_dir, "sensor")
        self.source_dir = source_dir
        self.period = period
        self.result_dir = result_dir
        self.results={}
        
    def launch_experiments(self, benchmark) -> str:   
        os.popen("chmod +x "+benchmark.execution_script_path)
        _ = subprocess.run(benchmark.execution_script_path, shell=True) # capture_output=True)
        # output = stream.read()
        # return self.process_stdout(output), output
        return {}, None
    
    def process_power_results(self):
        """Returns whatever power information given by tool"""
        nvml_csv_file = glob.glob(self.result_dir+"*-nvml.csv")[0]
        rapl_csv_file = glob.glob(self.result_dir+"*-rapl.csv")[0]
        sysinfo_csv_file = glob.glob(self.result_dir+"*-sysinfo.csv")[0]
        nvml_df = pd.read_csv(nvml_csv_file, sep=";")
        rapl_df = pd.read_csv(rapl_csv_file, sep=";")
        sysinfo_df = pd.read_csv(sysinfo_csv_file, sep=";")
        # NVML
        serie = nvml_df[8:].groupby(
            "device_index"
            )[
                "energy_consumption_since_previous_measurement_milliJ"
                ]
        nvml_energy = serie.sum()
        nvml_energy_kWh = convert_energy_joule_in_kWh(nvml_energy)*10**(-3)
        nvml_total_energy_kWh = nvml_energy_kWh.sum()
        # RAPL
        rapl_total_energy_df = rapl_df.groupby(
            "domain")[
                "energy_consumption_since_previous_measurement_milliJ"
                ]
        rapl_total_energy_df = rapl_total_energy_df.sum()
        rapl_total_energy = rapl_total_energy_df.sum()
        rapl_total_energy_kWh = convert_energy_joule_in_kWh(rapl_total_energy)*10**(-3)
        # AMD RAPL doesn't monitor the Dram
        if "Dram" in rapl_total_energy_df.keys():
            dram=convert_energy_joule_in_kWh(rapl_total_energy_df["Dram"])*10**(-3)
        else:
            dram=None
        results = {
            "tool_csv_file_nvml":nvml_csv_file, 
            "tool_csv_file_rapl":rapl_csv_file, 
            "tool_csv_file_sysinfo":sysinfo_csv_file, 
            "tool_energy_consumption(kWh)":rapl_total_energy_kWh+nvml_total_energy_kWh,
            "tool_GPU_energy_consumption(kWh)":nvml_total_energy_kWh,
            "tool_CPU_energy_consumption(kWh)":
                convert_energy_joule_in_kWh(rapl_total_energy_df["Package"])*10**(-3),
            "tool_RAM_energy_consumption(kWh)":dram,
            "tool_GPU_utilization(percent)":nvml_df["global_utilization_percent"].mean(),
            "tool_CPU_utilization(percent)":sysinfo_df["utilization_percent"].mean(),
            "tool_GPU_memory_utilization(percent)":nvml_df["global_memory_percent"].mean(),
            }
        self.results.update(results)
        return results

    
class EnergyScope(PowerTool):
    """Implementation of PowerTool for Energy Scope."""
    def __init__(
        self, 
        result_dir: str, 
        source_dir: str,
        job_id: str,
        ) -> None: 
        super().__init__(result_dir, "energy scope")
        self.source_dir = source_dir
        self.job_id = job_id
        self.result_dir = result_dir
        self.es_execution_command = self.source_dir+"/energy_scope_mypc.sh"
        self.es_analysis_command = self.source_dir+"/energy_scope_run_analysis_allgo.sh"

    def process_inputs(self) -> Tuple[str, str]:
        prefix_src = "ENERGY_SCOPE_SRC_DIR=" + self.source_dir
        prefix_traces = "ENERGY_SCOPE_TRACES_PATH=" + self.result_dir
        return prefix_src, prefix_traces
        
    def launch_experiments(self, benchmark) -> str:
        prefix_src, prefix_traces = self.process_inputs()
        es_command = "{} {} {} {}".format(
            prefix_src, 
            prefix_traces, 
            self.es_execution_command, 
            benchmark.execution_script_path
        )
        logging.info("Starting the execution of energy scope: "+es_command)
        stream = os.popen(es_command)
        output = stream.read()
        logging.info("Execution of energy scope DONE")
        return self.process_stdout(output), output
    
    def analyse_trace(self) -> None:
        logging.info("trace analysis")
        logging.info(os.listdir(self.result_dir))
        self.trace_tar_file = glob.glob(self.result_dir + "energy_scope_*.tar.gz")[-1]
        self.es_trace_id = self.trace_tar_file.split('.')[0].split('_')[-1]
        logging.info("Energy scope trace id")
        logging.info(self.es_trace_id)
        command = "{} {}".format(self.es_analysis_command, self.trace_tar_file)
        logging.info("Starting the analysis of "+self.trace_tar_file)
        stream = os.popen(command)
        output = stream.read()
        logging.info("Energy scope analysis output: \n")
        logging.info(output)
        logging.info("Analysis DONE")
        
    def process_eprofile(self) -> pd.DataFrame():
        logging.info("eprofile analysis")
        logging.info(os.listdir(self.result_dir + "tmp_energy_scope_{}/".format(self.es_trace_id)))
        self.trace_eprofile = self.result_dir + "tmp_energy_scope_{}/energy_scope_eprofile_{}.txt".format(self.es_trace_id, self.es_trace_id)
        with open(self.trace_eprofile) as f:
            norm_df = pd.json_normalize(json.load(f))
        return norm_df

    def process_power_results(self):
        """Returns whatever power information given by tool"""
        self.analyse_trace()
        energy_df = self.process_eprofile()
        watt_cols = [col for col in energy_df.columns if '(W)' in col]
        watt_df = pd.DataFrame()
        for col in energy_df[watt_cols]:
            watt_df[col] = energy_df[col][0]
        path_csv_param = self.result_dir+'energy_scope_parameters.csv'
        energy_df.to_csv(path_csv_param, index=False) 
        path_csv = self.result_dir+'energy_scope.csv'
        watt_df.to_csv(path_csv, index=False) 
        energy = convert_energy_joule_in_kWh(energy_df["data.data.tags.es_appli_total.joule(J)"].values[0])
        results = {
            "tool_csv_file":path_csv, 
            "tool_parameters":path_csv_param, 
            "tool_energy_consumption(kWh)":energy
            }
        return results