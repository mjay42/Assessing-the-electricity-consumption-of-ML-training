"""Start experiments.

It creates experiments from given arguments, starts them and processes and saves outputs.

typical use: run.sh
"""
import argparse
import logging
import os

from utils.experiments import System, Benchmark, Experiment
from utils.tools import EnergyScope, Sensor

def parsing_arguments():
    parser = argparse.ArgumentParser(description='GPU benchmark & energy tools')
    parser.add_argument('--template_script', 
                        help='Path to this git repo.', 
                        type=str, default='/root/energy-consumption-of-gpu-benchmarks/')
    parser.add_argument('--result_folder', 
                        help='Path to the folder to save the experiment summary in.', 
                        type=str, default='/root/energy-consumption-of-gpu-benchmarks/results/')
    parser.add_argument('--log_dir', 
                        help='Path to the folder to save experiment details in.', 
                        type=str)
    parser.add_argument('--tool_folder', 
                        help='Path to energy scope folder.', 
                        type=str, default='/root/energy_scope/')
    parser.add_argument('--bmc_monitoring_script', 
                        help='Path to the script fetching data from bmc.', 
                        type=str,
                        default="~/ai-energy-consumption-framework/utils/get_power.sh",
                        )
    parser.add_argument('--sampling_period', 
                        help='Period between each measurements in seconds.', 
                        type=float, default=1)
    parser.add_argument('--sleep_before', 
                        type=int, default=30)
    parser.add_argument('--sleep_after', 
                        type=int, default=30)
    parser.add_argument('--benchmark_execution_command', 
                        help='Benchmark bash execution command', 
                        type=str)
    return parser.parse_args()


def main():
    '''
    ajouter si on est sur g5k ou pas
    pour champollion, un truc pour récupérer ilo
    
    comment propager l'emplacement du result dir ?
    c'est dans le docker run que ça se passe

    installation
    '''
    logging.info('start main.py')
    
    # Processing arguments
    args = parsing_arguments()

    result_folder = args.result_folder
    
    current_system = System()
    logging.info(current_system.__dict__)

    log_dir=args.log_dir
    
    execution_script_args = {
        "sleep_before":args.sleep_before,
        "sleep_after":args.sleep_after,
        "benchmark_execution":args.benchmark_execution_command,
    }
    tool_args = {
        "RESULT_DIR":log_dir,
        "INTERNAL_SENSOR_DIR":args.tool_folder,
        "PERIOD":args.sampling_period,
        # "ILO_SENSOR_SCRIPT":args.bmc_monitoring_script,
    }
    
    benchmark = Benchmark(
        benchmark_id = log_dir,
        execution_script_args = execution_script_args,
        execution_script_template = args.template_script,
        execution_script_path = log_dir+"run.sh",
        tool_args=tool_args,
    )
    benchmark.generate_script() # creates a shell file including the sleeps and the benchmark command to be used by ES

    # Configure the tool object
    # tool_instance = EnergyScope(log_dir, args.tool_folder, current_system.job_id)
    tool_instance = Sensor(log_dir, args.tool_folder, args.sampling_period)
    
    # Create the experiment object
    current_exp = Experiment(
        experiments_table=result_folder+"experiment_table.csv",
        tool=tool_instance, 
        benchmark=benchmark,
        system=current_system,
    )

    # Start the experiment
    current_exp.start()

    # Process results
    #current_exp.retrieve_wattmeter_values() #to do afterwards
    current_exp.process_power_tool_results()
    current_exp.save_experiment_results()
    logging.info("Experiment DONE.")


if __name__ == '__main__':
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs.log'), 
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    try:
        main()
    except Exception as err:
        logging.error(err)
