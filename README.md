# HRNet 

## Installation 

+ Create a conda environment: ```conda create -n hrnet python=3.7```
+ Download cudatoolkit=11.0 from [here](https://developer.nvidia.com/cuda-11.0-download-archive) and install 
+ ```pip3 install torch==1.7.1+cu115 torchvision==0.8.2+cu115 -f https://download.pytorch.org/whl/torch_stable.html```
+ ```pip3 install -r requirements.txt```


https://hooni-playground.com/903/

python
import torch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
