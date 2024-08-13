# Assessing the environmental impact of ML training
##### Author: Mathilde Jay    
##### Last updated: 15 Aug. 2024.

This repository corresponds to the projects I included in my PhD manuscript. It contains the code I used to conduct the experiments, and how I analysed the resulting logs and energy database.

This repository is organized as follow:
- MLPerf   
    - Apollo: Training MLPerf on a node of the HPE Champollion cluster, and analyze the electricity consumption.   
    - Jetson: Training ResNet-50 on ImageNet on a node of the Estats cluster of Grid'5000, and analyze the electricity consumption.   
- StableDiffusion: Training and infering from Stable Diffusion on the Sirius and Gemini clusters of Grid'5000, and analyze the electricity consumption.   

Please refer to each folder for more details on each section.

Logs and results from each projects will be made available in the coming months.

## Sensors
To monitor the electricity of computing components, we used a precursor of [Alumet](https://alumet.dev/), that we called NVML sensor (CPU + GPU).
- Github repository: [https://github.com/TheElectronWill/nvml-sensor](https://github.com/TheElectronWill/nvml-sensor)
- HAL identifier: [hal-04664358](https://hal.science/hal-04664358)

The sensor is developed in Rust thus it needs to be compiled before it can be used. One solution is to execute the commands below and follow the steps describes in nvml-sensor.

``` 
mkdir sensors
cd sensors/
git clone https://github.com/TheElectronWill/nvml-sensor.git
git clone https://github.com/TheElectronWill/cpu-energy-consumption-comparative-analysis.git
```

## Acknowlodgment
Experiments presented in this thesis were carried out on several clusters. The Champollion cluster was designed by HPE and made available to us thanks to Bruno Monnet through collaboration with MIAI. The Grid’5000 / Slices testbed is supported by a scientific interest group hosted by Inria that includes CNRS, RENATER, and several Universities as well as other organizations. This work was funded by MIAI (ANR19-P3IA-0003) and the BATE project (BATE-UGAREG21A87) of the Auvergne Rhône-Alpes French region.

## License
This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
You should have received a copy of the license along with this work. If not, see http://creativecommons.org/licenses/by-sa/4.0/.
