# RFENet

The code for the paper "Surface Mesh Reconstruction from Medical Images via Enrichment Feature Learning and Mesh Contour Loss", which is accepted to IEEE Transactions on Medical Robotics and Bionics, 2025.12.

Our code is based on [Vox2Cortex](https://github.com/ai-med/Vox2Cortex?tab=readme-ov-file) and [voxel2mesh](https://github.com/cvlab-epfl/voxel2mesh).

## How to run
### 1. Environment
Please prepare an virtual environment with Python 3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. Dataset
[OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/) form [OASIS dataset](https://sites.wustl.edu/oasisbrains/).

[ADNI dataset](https://adni.loni.usc.edu/)

[WORD dataset](https://github.com/HiLab-git/WORD)

Download the dataset, and change the dataset_path in utils/params.py, data/supported_datasets.py and scripts/eval_meshes.py

### 3. Training
The default implementation details are in the utils/params.py. Please update the paths within it before training.

python3 main.py --train

### 4. Testing
For testing, N_TEMPLATE_VERTICES = 168058

1. generate meshes:

   python3 main.py --test
   
3. evaluate meshes:

   cd scripts

   python3 eval_meshes.py
   
![image](https://github.com/VCL-HNU/RFENet/blob/main/graph_abstract.jpg)
