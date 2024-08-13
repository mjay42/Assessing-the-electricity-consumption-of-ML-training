"""
This module contains utility functions and classes for running experiments on a server and multiple clients.
"""
from types import SimpleNamespace
import warnings
from execo import SshProcess, Remote
from execo.host import Host
from execo_g5k import get_oar_job_nodes
from execo_engine import Engine, logger, ParamSweeper, sweep
from datetime import datetime
from typing import Any, Dict, Tuple, List
from pathlib import Path
from box import Box
import pandas as pd
import yaml
import os
import logging
import time
import shutil


logger = logging.getLogger("MyEXP")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def execute_command_on_server_and_clients(host, command, background=False):
    """
    Executes a command on a host via ssh.

    Args:
        host (list): hostname or IP addresses of the servers and clients.
        command (str): The command to be executed.
        background (bool, optional): If True, the command will be executed in the background. 
            Defaults to False.

    Returns:
        list: List of SshProcess objects representing the executed processes.
    """
    
    processes = []
    process = SshProcess(
        command, 
        host=host, 
        connection_params={'user':'root'},           
        )
    processes.append(process)
    if background:
        process.start()
    else:
        process.run()
        if process.ok:
            print(f"Successfully executed on {host} command '{command}'")
        else:
            process.kill()
            print(f"Failed to execute on {host} command '{command}'")
    return processes
    