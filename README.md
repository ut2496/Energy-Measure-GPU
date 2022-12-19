# Analysis of Energy Efficiency for GPU's in Deep Learning Tasks
This is the submission for the Project for HPML (ECE-GY 9143) at NYU. <br>
<br>

**Developers: George Gao, Utkarsh Shekhar**    

## Overview
The objective of this project is to analyze the efficiency of various GPU's in Deep Learning Tasks using the `nvidia-smi` and `pynvml` bindings. <br>
The Project is structured in 2 parts : 
* The Models directory contains all the models being used for the experiment (Here its the ResNets)
* The `measure_energy.py` file contains the training loop as well as the functions and setup to measure the Power consumption in Watts and Temperature in Celsius.

## Dataset
CIFAR-10

## Models
ResNet-18 , 
ResNet-50 , 
ResNet-152 

## Requirements/Prerequisites
* Currently runs only on Nvidia GPUs
* pynvml, Pytorch, Matplotlib

## Recreating the results
Run the command `python measure_energy.py`. You can chose the custom optimizers from the list of available ones by passing the `-opt 'sgd'` argument.
The Training loop ends when the Validation accuracy of 80% is reached and returns back the Average and Peak Temperature of the GPU along with the Average and Peak Power Consumption for the training job. Additonally plots charting the power and temperature measurements are also created.

<br>
To replicate the results of throttling the data, use the command `python measure_energy.py` and replace `POWER_LIMIT` with the desired value in Watts. After this command is run successfully run the main python file with `python measure_energy.py`.

## Results 
Sample results of v100 on Resnet-50

<img width="535" alt="resnet50_v100_80" src="https://user-images.githubusercontent.com/14841193/208510703-f4fd7c00-e285-404a-8453-b4f59bed3742.png">

![power_resnet50_v100](https://user-images.githubusercontent.com/14841193/208510868-2e2482e2-1bb7-4384-8f0c-7cccfa904c18.png)
