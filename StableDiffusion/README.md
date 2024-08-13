# Measuring the electricity consumption of Stable Diffusion
This project was developed on the Lyon site of the Grid'5000 test bed. You will need to ssh there to execute the code.

The librairies required were installed using Miniconda3 that can be loaded as a module in Grid'5000 (`module load conda`).
The requirements are listed in the file with the same name. 

You will have to modify the parameters in the bash files.

The prompts folder contains a list of prompts that can be used to run inferences.

## Inference
```
bash code/run_inference.sh <result folder path>
```
## Training
```
bash code/run_inference.sh <result folder path>
```
## Automate reservation of Grid'5000 jobs
Example of use: 
```
    python start_job.py --site_id lyon \
        --host sirius \
        --script_path "/home/mjay/ai-energy-consumption-framework/stable-diffusion/run_training.sh"  \
        --result_path "/home/mjay/laion/pokemon/training_6_09_sirius/" \
        --reservation_date "2023-09-06 1:30:00"   \
        --walltime 50000  \
        --gpu_nb 8 
````
