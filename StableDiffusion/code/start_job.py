"""Reserve a grid'5000 node and start experiment script.

Example of use: 

    python start_job.py --site_id lyon \
        --host sirius \
        --script_path "/home/mjay/ai-energy-consumption-framework/stable-diffusion/run_training.sh"  \
        --result_path "/home/mjay/laion/pokemon/training_6_09_sirius/" \
        --reservation_date "2023-09-06 1:30:00"   \
        --walltime 50000  \
        --gpu_nb 8 
        
    python start_job.py --site_id lyon \
        --host sirius \
        --script_path "/home/mjay/ai-energy-consumption-framework/stable-diffusion/run_training.sh"  \
        --result_path "/home/mjay/laion/pokemon/training_27_09_sirius_adamW/" \
        --walltime 7200  \
        --gpu_nb 8 
    
    python start_job.py --site_id lyon \
        --host sirius \
        --script_path "/home/mjay/ai-energy-consumption-framework/stable-diffusion/run_inference.sh"  \
        --result_path "/home/mjay/laion/pokemon/inference_29_09_sirius" \
        --walltime 3000  \
        --gpu_nb 8
"""
import argparse
from datetime import datetime

from execo_g5k import oarsub, OarSubmission, wait_oar_job_start

def parsing_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site_id', 
                        help='Grid5000 site.', 
                        default="lyon",
                        type=str)
    parser.add_argument('--host', 
                        help='The name of the cluster or the node.', 
                        default="gemini",
                        type=str)
    parser.add_argument('--script_path', 
                        help='Path to the script to execute.', 
                        default="/home/mjay/ai-energy-consumption-framework/run_sd.sh",
                        type=str)
    parser.add_argument('--result_path', 
                        help='Path of the directory where the results should be saved.', 
                        default="/home/mjay/laion/pokemon/results_31_08/",
                        type=str)
    parser.add_argument('--reservation_date', 
                        help='Date for the job reservation. Please follow the format: "2023-09-01 9:03:00". "" if no reservation ', 
                        default="",
                        type=str)
    parser.add_argument('--walltime', 
                        help='Walltime of the job in seconds. Will be killed when time reached. 2 hours by default', 
                        default="7200",
                        type=str)
    parser.add_argument('--gpu_nb', 
                        help='Number of GPUs to use', 
                        default="8",
                        type=str)

    return parser.parse_args()

def main():
    args = parsing_arguments()
     
    command = f'bash {args.script_path} {args.result_path}'

    dt_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print("Creating job at", dt_string)
    if args.reservation_date=="":
        [(jobid, site)] = oarsub([
            (OarSubmission(
                resources = f"gpu={args.gpu_nb}", 
                sql_properties = args.host,
                walltime = args.walltime,
                command = command,
                job_type = "exotic",
                project = "datamove",
                ), 
            f"{args.site_id}")
            ])
    else:
        [(jobid, site)] = oarsub([
            (OarSubmission(
                resources = f"gpu={args.gpu_nb}", 
                sql_properties = args.host,
                walltime = args.walltime,
                command = command,
                job_type = "exotic",
                reservation_date = args.reservation_date,
                ), 
            f"{args.site_id}")
            ])
    if jobid:
        print(f"job submitted, id is {jobid}")
        try:
            success = wait_oar_job_start(jobid, site)
            if success:
                start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print(f"job started at {start_time}.")
        except:
            print("job NOT started.")
        
if __name__ == "__main__":
    main()
