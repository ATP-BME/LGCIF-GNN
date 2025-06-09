# LGCIF-GNN
This repository provides the official PyTorch implementation of LGCIF-GNN: a Local-Global Clinical and Imaging Feature Fusion Graph Neural Network for predicting acute-phase treatment response to selective serotonin reuptake inhibitors (SSRIs). 
Please refer to our paper for detailed methodology and evaluation.

# Requirements
```
pytorch >= 1.13.0
torch-geometric == 2.3.1
torch-sparse == 0.6.17
torch-cluster == 1.6.1
torch-scatter == 2.1.1
```

# Data Preparation
To train and evaluate the model using your own dataset, you need to prepare both functional imaging data and clinical data.

For functional imaging data, you should provide the following for each subject:  
-an N by N adjacency matrix (N is the number of nodes),  
-ROI-wise BOLD time series, which can be extracted using the DPABI/DPARSF toolbox.
Data preprocessing scripts are available in ./data/preprocess.

For clinical data:
-Clinical features should be saved in a .csv file following the format of the provided demo file: clinical_info_demo.csv.

## Training
To quickly start training with the default configuration:
```
sh train.sh
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader_local_global.py` accordingly.  

## Infering
To perform inference using a pretrained model checkpoint, run:
```
python main_local_global.py --train False --ckpt_path './save_models/ckpt_path'
```
Replace './save_models/ckpt_path' with the actual path to your saved model checkpoint.

## Citation
If you find this repository useful in your research, please consider citing our paper (citation info to be updated after publication).
