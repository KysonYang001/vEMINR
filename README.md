# vEMINR: Faster Isotropic Reconstruction for Voluem Electron Microscopy with Implicit Neural Representation

## Contents
- [S1 Overview](#s1-overview)
- [S2 Installation](#s2-installation)
- [S3 Walkthrough](#s3-walkthrough)
  - [S3.1 Training](#s31-training)
  - [S3.2 Testing](#s32-testing)
- [S4 Availability of data](#s4-availability-of-data)

## S1 Overview
Here, we introduce vEMINR, a faster isotropic reconstruction method based on implicit neural representations (INR). 
The method improves the reconstruction quality of vEM images by learning the true degradation patterns of low-resolution images, 
and significantly accelerates the reconstruction process by utilizing the efficient parameterization and continuous function representation of INR. 
Experimental results on eight public datasets demonstrate that vEMINR outperforms existing mainstream methods in both reconstruction speed and quality.

## S2 Installation
- Clone this repository
  ```bash
  git clone https://github.com/KysonYang001/vEMINR.git
  ```
- Create conda environment and activate
  ```bash
  conda create -n vEMINR python=3.10
  conda activate vEMINR
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
- Install Random Fourier Features Pytorch
  ```bash
  pip install random-fourier-features-pytorch
  ```

- Install other dependencies
  ```bash
  pip install -r requirements.txt
  ```

## S3 Walkthrough

### S3.1 Training
We train the network by using the Adam optimizer on a NVIDIA A100 GPU. <br>
Before training, revise the absolute path in config file according to your setting.
You can change the model, training, data settings in the config file. You can also define a new config file as needed.
  
  ```bash
  configs/degradation.yaml
  configs/train_INR-volume.yaml
  ```

Training our model.
  ```bash
 ./train.sh
  ```

### S3.2 Testing
Before testing, modify the absolute path of the input data in the configuration file according to your settings.
You can change data-related hyperparameters such as reconstruction size and batch size in the configuration file. You can also define a new configuration file as needed.
  
  ```bash
  configs/test_yz.yaml
  configs/test_xz.yaml
  ```
Testing our model.
  ```bash
 ./test.sh
  ```

## S4 Availability of data
The EPFLdataset was downloaded from the EPFL website https://www.epfl.ch/labs/cvlab/data/data-em/.
<br>



