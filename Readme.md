This codes includes our works on Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting.



The code consists of two parts. The first part is dynamic causal graph generation module, and the second part is multi-step prediction module. We supply the demo of transportation and FMRI dataset.

##For the dynamic causal graph generation module,
The setting are shown in configuration document：./graph_generation/configurations/Tdrive.conf
## Requirements:
* tensorflow 
* scipy 
* numpy 
* matplotlib 
* pandas 
* math 
* seaborn 
* sklearn
* argparse
* configparser
* time

## Run the demo:
./graph_generation/Transportation or FMRI\main.py

Then we will get the parameter file after training, for example, Tdrive_normalization_parameter.npz. Run dynamic_graph_trans_.py to generate the dynamic transition matrix, such as dynamic_Tdrive_adj.npy file. 




##For the second part,
put the generative file from the first step to ./prediction/Transportation or FMRI/generated_adj (file directory)

## Requirements:
* torch 
* shutil  
* numpy
* matplotlib 
* pandas 
* math 
* tensorflow 
* sklearn 
* argparse 
* csv 
* time

## Run the demo:
./prediction/Transportation or FMRI\main.py



If you have any questions, please feel free to email me!

