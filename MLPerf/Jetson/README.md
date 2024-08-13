# Train ResNet-50 using the ImageNet dataset on an Nvidia Jetson AGX Xavier
This project was developed on the Toulouse site of the Grid'5000 test bed. You will need to ssh there to execute the code.

The librairies required were installed using venv.
The requirements are listed in the file with the same name. 

## Download ImageNet
You need to modify dataset.py to change the paths, and create an huggingface account to access the dataset. 
We relied on a storage group to save the dataset.

```
pip install datasets[vision]
huggingface-cli login 
python dataset.py 
```
## Reserve a node and execute the training:
You will have to modify the parameters in the `reserve_execute.py` file before executing it:

```
python reserve_execute.py
```