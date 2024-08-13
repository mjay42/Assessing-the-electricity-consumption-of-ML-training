"""To start the experiments and save its metadata, including system description and wattmeter data.

The System class gathers system information. It also contains a method to create a log directory
    automatically unique to each experiment.
The Benchmark class can be used to automatically generate scripts executing bencharks.
The Experiment class collects necessary metadata, start the experiment and saves its results.

Typical use:
- Example in start_exp.py
- Simpler example, to start the application EP A on a single GPU and monitoring it with Energy Scope:

```
from experiments import System, Benchmark, Experiment
from tools import EnergyScope

# Variables
binary_dir = "NAS_benchmark_binaries/"
execution_script_template = "code/templates/script_template.sh"
experiments_dir="results/"

# Initialisation
current_system = System()
log_dir, exp_id = current_system.create_log_dir(experiments_dir)
script_dir = log_dir + "scripts/"

# Create benchmark object
benchmark = Benchmark(
    benchmark_id = exp_id,
    sleep_before = 30,
    sleep_after = 30,
    execution_script_template = execution_script_template,
    execution_script_path = script_dir,
)

# Create tool object
tool_instance = EnergyScope(log_dir)

# Create the experiment object
current_exp = Experiment(
    experiments_table=experiments_dir+"experiment_table.csv",
    tool=tool_instance, 
    benchmarks=[[benchmark]],
    system=current_system,
)

# Start the experiment
current_exp.start()

# Process results
current_exp.retrieve_wattmeter_values() #to do afterwards
current_exp.process_power_tool_results()
current_exp.save_experiment_results()
```
"""
import cpuinfo
import getpass
import logging
import os
import psutil
import random
import re
from requests import auth
import time
from typing import Tuple

import pandas as pd

from utils.tools import PowerTool, Wattmeter
from collections.abc import MutableMapping

class System:
    """Retrieves system information.

    Description of the system is necessary to conduct reproducible experiments.
    This class contains methods to automatically retrieve needed information.
    It also contains the create_log_dir() method which creates a folder for
    each new experiment.

    TODO: add space (total and available)

    Attributes:
        job_id: A string corresponding the g5k job id 
        exp_id: A string corresponding the experiment id
        host_name: A string corresponding the g5k host name 
        site: A string corresponding the g5k site
        node: A string corresponding the g5k node
        gpu_name: A string corresponding the model of the GPU used
        gpu_count: An integer correspodning the number of GPU used
        cpu_name: A string corresponding the model of the CPU used
        cpu_count: An integer corresponding the total number of virtual cores 
        cpu_phyical_core_count: An integer corresponding the total 
            number of physical cores 
    """
    def __init__(self) -> None:
        self.job_id = self.retrieve_g5k_job_id()
        self.exp_id = self.job_id
        self.host_name = self.retrieve_host_name()
        self.site, self.node = self.retrieve_site_node()
        if "\n" in self.node:
            self.node = self.node[-1]
        self.gpu_name = self.retrieve_gpu_name()
        self.gpu_count = self.retrieve_gpu_count()
        self.cpu_name = self.retrieve_cpu_name()
        self.cpu_count = self.retrieve_cpu_count()
        self.cpu_phyical_core_count = self.retrieve_cpu_count(virtual = False)
        #self.g5k_auth = self.get_g5k_auth()

    def get_g5k_auth(self):
        """The credentials are asked at every execution to avoid saving them."""
        user = getpass.getpass(prompt='Grid5000 login:')
        password = getpass.getpass(prompt='Grid5000 password:')
        return auth.HTTPBasicAuth(user, password)

    def retrieve_g5k_job_id(self) -> str:
        name = "OAR_JOB_ID"
        if name in os.environ:
            return os.environ.get(name)
        else:
            return str(random.randint(0,1000))

    def retrieve_gpu_name(self) -> str:
        nvsmi_res = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv").read()
        match = re.search(r'\n(.*?)\n', nvsmi_res)
        if match:
            gpu_name = match.group(1)
            logging.info(gpu_name)
            return gpu_name
        
    def retrieve_gpu_count(self) -> str:
        nvsmi_res = os.popen("nvidia-smi --query-gpu=count --format=csv").read()
        match = re.search(r'\n(.*?)\n', nvsmi_res)
        if match:
            gpu_name = match.group(1)
            logging.info(gpu_name)
            return gpu_name
        
    def retrieve_cpu_name(self) -> str:
        return cpuinfo.get_cpu_info()['brand_raw']
        
    def retrieve_cpu_count(self, virtual=True) -> str:
        return psutil.cpu_count(logical=virtual)

    def retrieve_host_name(self) -> str:
        return os.popen("hostname").read()

    def retrieve_site_node(self) -> Tuple[str, str]:
        """
        hostname is of the form "chifflet-6.lille.grid5000.fr"
        """
        if "grid5000" in self.host_name:
            node, site, _, _ = self.host_name.split(".")
            return site, node
        elif "chifflet" in self.host_name:
            return "lille", self.host_name
        elif "gemini" in self.host_name:
            return "lyon", self.host_name
        else:
            return "no_site", "no_host"



class Benchmark:
    """Describes a benchmark to be launched within one experiment.

    Benchmarks are defined by an application (EP, LU or MG) and a class (A to E).
    The generate_script method creates a shell script in execution_script_path 
        which needs to be provided to the tool object. 
    A script template (execution_script_path) must be provided so that the only action to 
        do is add the application, the class, the gpu on which the benchmark will be 
        executed and the sleep durations.
    Sleeps are added before and after the benchmark is launched to make
        sure the machine is idle.
    Timestamps are echoed when the benchmark starts and ends to be able to cut
        the power time series.

    Attributes:
        benchmark_id: str. The benchmark id.
        benchmark_execution_command: path of the file to be executed. 
        sleep_before: int (seconds). Time to wait before starting the benchmark.
        sleep_after: int (seconds). Time to wait after the benchmark ended.
        execution_script_template: A string of the path to the template to use.
        execution_script_path: A string of the path to save the execution script at.
    """
    def __init__(
        self,
        benchmark_id: str,
        execution_script_args: dict,
        execution_script_template: str,
        execution_script_path: str,
        tool_args: dict
    ) -> None:
        self.benchmark_id = benchmark_id
        self.execution_script_args = execution_script_args
        self.execution_script_template = execution_script_template
        self.execution_script_path = execution_script_path
        self.tool_args = tool_args
        
    def generate_script(self) -> None:
        script_args={}
        script_args.update(self.tool_args)
        script_args.update(self.execution_script_args)
        logging.debug("Script arguments: {}".format(script_args))
        logging.debug("Parsing script to add arguments.")
        with open(self.execution_script_template, "r") as template:
            template_lines = template.readlines()
        for i in range(len(template_lines)):
            for key, value in script_args.items():
                if key==template_lines[i][:-2]:
                    if type(value)==str:
                        template_lines[i]='{}="{}"\n'.format(key, value)
                    else:
                        template_lines[i]='{}={}\n'.format(key, value)
        execution_script="".join(template_lines)
        with open(self.execution_script_path, "w") as script:
            script.write(execution_script)
        os.popen("chmod 777 '{}'".format(self.execution_script_path))
                

class Experiment:
    """Describes one experiment.

    One experiment correspond to the benchmarking of one software wattmeter. 
    It usually includes several benchmark applications with sleeps inbetween.
    The class contains the methods to start an experiment, process and save the tool
    results and retrieve the corresponding wattmeter values.

    The variable results doesn't include time series or dataframe, only the path 
        where to store/find them.

    Attributes:
        experiments_table: str. Path to table (csv file) where all experiments
            are described.
        tool: A PowerTool object.
        system: System object containing system information 
            like the number of GPU. 
        benchmark: 
            Benchmark object.
        results: dict of results.
    """
    def __init__(
        self,
        experiments_table: str,
        tool: PowerTool,
        benchmark,
        system: System,
    ) -> None:
        self.experiments_table = experiments_table
        self.tool = tool
        self.result_dir = tool.result_dir
        self.benchmark = benchmark
        self.system = system
        self.results = {}
         
    def start(self):
        logging.info("Starting execution of experiment {}".format(self.system.exp_id))
        exp_start = time.time()
        bench_timestamps, output = self.tool.launch_experiments(
            self.benchmark,
        )
        exp_end = time.time()
        logging.info("Ending execution of experiment {}".format(self.system.exp_id))
        exp_timestamps = {"experiment_start": exp_start, "experiment_end": exp_end}
        self.results.update(exp_timestamps)
        self.results.update(bench_timestamps)

    def process_power_tool_results(self):
        """Processes the results and add them to the results dict.

        The exception is needed to be able to save the error and
            make sure the following benchmark can be conducted
            as planned.
        """
        try:
            results = self.tool.process_power_results()
            logging.debug(results)
            self.results.update(results)
        except Exception as err:
            logging.error(err)
            self.results['error']=err
            logging.error("Continuing so the experience can be saved.")

    def retrieve_wattmeter_values(self):
        """Retrieves the wattmeter data and saves them to the results dict.
        """
        wattmeter_data = Wattmeter(
            self.system.node, 
            self.system.site, 
            self.results["experiment_start"], 
            self.results["experiment_end"],
            self.system.g5k_auth,
        )
        wattmeter_data.save_power(self.tool.result_dir)
        for metric in wattmeter_data.metrics:
            self.results[metric+'_energy_consumption(kWh)'] = wattmeter_data.results[metric]['energy_kWh']
            self.results[metric+'_csv_file'] = wattmeter_data.results[metric]['csv_file']

    def save_experiment_results(self):
        """Saves results of the experiment."""
        logging.info("Starting processing results.")
        
        self.results.update(self.tool.__dict__)
        self.results.update(flatten_dict(self.benchmark.__dict__))
        self.results.update(self.system.__dict__)

        logging.info("Experiment info: \n")
        
        for key, val in self.results.items():
            logging.info("{} : {}".format(key, len(val) if type(val) is list else val))
        
        # add experiment to existing results
        if os.path.exists(self.experiments_table):
            logging.info('Result file already created, appending results.')
            exp_df = pd.read_csv(self.experiments_table)
            exp_df = pd.concat([exp_df, pd.DataFrame(self.results, index=[0])], ignore_index=True)
        else:
            logging.info('Creating result file at '+self.experiments_table)
            exp_df = pd.DataFrame(self.results, index=[0])
        
        print(len(exp_df))
        exp_df.to_csv(self.experiments_table, index=False)
        logging.info("Result df saved.")
        
        return exp_df
    
    
def flatten_dict(d):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kc, vc in v.items():
                items["{}.{}".format(k, kc)] = vc
        else:
            items[k]=v
    return items