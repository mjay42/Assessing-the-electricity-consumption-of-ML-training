#
import os
import sys
from utils.experiment import execute_command_on_server_and_clients
from execo_g5k import oarsub, oardel, OarSubmission, get_current_oar_jobs, get_oar_job_nodes, get_oar_job_info, Deployment, deploy
from execo import SshProcess
import logging
import datetime
import time
import subprocess, json
import pandas as pd
import random

jobname="resnet"
nodecount=1
walltime="18:00:00"
node="estats-1"
resources_selection=f"-t exotic -p {node}"
site="toulouse"
reservation=None # "2024-04-23 19:00:00"
force_redeploy=False
environment_dsc_file='./images/jetson_resnet.yaml'
storage_group="energyfl"
date_exp="20240723_132409"

TRAIN_REPO="/home/mjay/jetson-inference"
TRAINING_DIR=f"{TRAIN_REPO}/python/training/classification"
REPO="/home/mjay/ai-energy-consumption/Jetson"
TEGRA_SCRIPT=f"{REPO}/utils/jetson_monitoring.py"
JETSON_DATA="/root/energyfl/imagenet-1k" 
BASH_LOG="run.log"
EXP_CSV=f"{REPO}/logs/experiment_summary.csv"

# Reserve a job and deploy the chosen environment
jobs = get_current_oar_jobs()
jobid = None
waiting_jobs = []
ifdeploy=True
while jobs:
    j, site = jobs.pop()
    info = get_oar_job_info(j, site)
    if info['name'] == jobname:
        if info['state'] == 'Running':
            jobid = j
            logging.info("A {} job is already running, using it. jobid is {}".format(jobname, jobid))
            ifdeploy=False
            break
        else:
            waiting_jobs.append(j)
if not jobid and not waiting_jobs:
    jobspec = OarSubmission(resources="/nodes={}".format(nodecount), 
                            walltime=walltime,
                            reservation_date=reservation,
                            additional_options=resources_selection, 
                            name=jobname, 
                            job_type="deploy",
                            )
    jobid, _ = oarsub([(jobspec, site)]).pop()
    logging.info("New job submitted, jobid is {}".format(jobid))
elif not jobid:
    logging.info("One or more {} jobs exist ({}) but are not running.\n"
          " Connect to the frontend to see what is happening, and/or run the cell again.".format(
          jobname, ", ".join([str(j) for j in waiting_jobs])))
    if len(waiting_jobs) > 1:
        logging.info("Waiting for more than one job to run is not supported, please cancel all but one job.")
    else:
        jobid = waiting_jobs[0]
        logging.info("A {} job is already waiting, using it. jobid is {}".format(jobname, jobid))

#
logging.info("Waiting for job to start")
nodes = get_oar_job_nodes(jobid, site, timeout=None)
nodes.sort(key=lambda n: n.address)
logging.info("Node is ready: {}".format(nodes[0].address))
node = nodes[0]

# Deploy
logging.info("Deploying environment")
deployment = Deployment(hosts=nodes, env_file=os.path.abspath(environment_dsc_file))
deploy_ok, deploy_failed = deploy(deployment, check_deployed_command=not force_redeploy
                            )
logging.info("Deployement status:\n* ok: {}\n* failed: {}".format(deploy_ok, deploy_failed))

# [markdown]
# # Allow access for the node to the nfs storage group and mount the storage to a folder

logging.info("Allowing access to the storage")
site="toulouse"
subprocess.run([
  'curl',
  '-X', 'POST',
  '-H', 'Content-Type: application/json',
  '-d', json.dumps({"termination" : {"job": jobid, "site": site}}),
  'https://api.grid5000.fr/stable/sites/toulouse/storage/storage1/energyfl/access',
])

#
cmd0 = f"mkdir /root/{storage_group} ; mount storage1.toulouse.grid5000.fr:/export/group/{storage_group} /root/{storage_group}/"
_ = execute_command_on_server_and_clients(nodes[0], cmd0, background=False) 
logging.info("Storage mounted")

logging.info("Restart jtop")
cmd1 = f"sudo systemctl restart jtop.service"
process = SshProcess(cmd1, host=nodes[0], connection_params={'user': 'root'},
                    )
process.run()
logging.info("Jtop restarted")
time.sleep(10)

# Training parameters
SLEEP_TIME=60

BATCH_SIZE=192
EPOCHS=20000000000
ARCH="resnet50"
PRINTFREQ=100
EVALFREQ=500
EARLYSTOP=20000
RESOLUTION=224
LR=0.001
POWER_MODE='MODE_30W_ALL' #'MAXN', 'MODE_10W', 'MODE_15W', 'MODE_30W_ALL', 'MODE_30W_6CORE', 'MODE_30W_4CORE', 'MODE_30W_2CORE', 'MODE_15W_DESKTOP'

SEED=random.randint(0, 10000)
LOG_DIR=f"{REPO}/logs/{date_exp}/"
logging.basicConfig(
    force=True,
    filename=LOG_DIR+BASH_LOG, 
    level=logging.DEBUG,
    format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
    )
# Train
logging.info(f"Experiment started at {date_exp}.")
logging.info(f"Batch size: {BATCH_SIZE}")

USER_COMMAND = f"cd {TRAIN_REPO}; python3 {REPO}/train.py \
    {JETSON_DATA} \
    --arch={ARCH} \
    --seed={SEED} \
    --batch-size={BATCH_SIZE} \
    --epochs={EPOCHS} \
    --print-freq={PRINTFREQ} \
    --eval-freq={EVALFREQ} \
    --early-stop={EARLYSTOP} \
    --model-dir={LOG_DIR} \
    --log-dir={LOG_DIR} \
    --resolution={RESOLUTION} \
    --learning-rate={LR} \
    --resume={LOG_DIR}/checkpoint.pth.tar \
    --workers=2 \
        "

logging.info("Freeing cache")
cmd2 = "sync && echo 3 | tee /proc/sys/vm/drop_caches"
process = SshProcess(cmd2, host=nodes[0], connection_params={'user': 'root'},
                    stdout_handlers=[sys.stdout, LOG_DIR+BASH_LOG],
                    stderr_handlers=[sys.stderr, LOG_DIR+BASH_LOG],
                    )
process.run()

#
cmd3 = f"cd {REPO}; python3 {TEGRA_SCRIPT} --log-dir={LOG_DIR} --power-mode={POWER_MODE}"
monitoring_process = SshProcess(cmd3, host=nodes[0], connection_params={'user': 'root'},
                    stdout_handlers=[sys.stdout, LOG_DIR+BASH_LOG],
                    stderr_handlers=[sys.stderr, LOG_DIR+BASH_LOG],
                    )
monitoring_process.start()
logging.info("Tegra-stats monitoring script running")

#
logging.info(f"Sleeping for {SLEEP_TIME} seconds")
time.sleep(SLEEP_TIME)

logging.info("Running training command")

process = SshProcess(USER_COMMAND, host=nodes[0], connection_params={'user': 'root'},
                    stdout_handlers=[sys.stdout, LOG_DIR+BASH_LOG],
                    stderr_handlers=[sys.stderr, LOG_DIR+BASH_LOG])
process.run()

logging.info("Training ended")

logging.info(f"Sleeping for {SLEEP_TIME} seconds")
time.sleep(SLEEP_TIME)
#
monitoring_process.kill()
logging.info("Killed tegra-stats script")

# And don't forget to kill the job when you're done with the experiments
oardel([(jobid,site)])