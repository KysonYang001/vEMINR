# vEMINR: Faster Isotropic Reconstruction for Volume Electron Microscopy with Implicit Neural Representation

**Paper:** [Advanced Science (2026)](https://doi.org/10.1002/advs.202511922)

## Contents
- [S1 Overview](#s1-overview)
- [S2 Installation](#s2-installation)
- [S3 Walkthrough](#s3-walkthrough)
  - [S3.1 Training](#s31-training)
  - [S3.2 Testing](#s32-testing)
  - [S3.3 Model Zoo](#s33-model-zoo)
- [S4 Availability of data](#s4-availability-of-data)
- [S5 Citation](#s5-citation)
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
  conda create -n vEMINR python=3.8
  conda activate vEMINR
  conda install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102

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
  configs/train_INR-rdn-liif-volume.yaml
  ```

Training our model.
  ```bash
 ./train.sh
  ```

For more details, please refer to the [Training Notebook](https://github.com/KysonYang001/vEMINR/blob/main/example/demo_training.ipynb).  
The training notebook includes the complete pipeline of model setup, data preparation, and the training procedure, which consists of two stages: training a degradation extractor to learn image degradation characteristics, and jointly training a feature extraction network and an INR-based upsampling network for isotropic reconstruction.

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

For more details, please refer to the [Inference Notebook](https://github.com/KysonYang001/vEMINR/blob/main/example/demo_inference.ipynb).  
The inference notebook demonstrates slice-wise isotropic reconstruction with a 2D architecture: it loads pretrained weights, takes anisotropic axial slices (XZ/YZ) as inputs, performs super-resolution on each slice under the INR-based upsampling module, and then reassembles the predictions into a 3D isotropic volume. It further supports orthogonal-view visualization for before/after comparison.

### S3.3 Model Zoo

We provide the trained models on the EPFL dataset at BaiduYun and GoogleDrive.

| Methods         | Models                     | Download                                                                 |
|-----------------|----------------------------|--------------------------------------------------------------------------|
| The degradation predictor    | epoch-best.ckpt   | [BaiduYun](https://pan.baidu.com/s/1KWFARNWuFXW2pCrpuxt20g?) (Access code: kysy) or [GoogleDrive](https://drive.google.com/file/d/1mAv4LlfPImMc_G9I5fvPdvT9lvrBQQ-e/view?usp=sharing) |
| The super-resolution model    | epoch-best.ckpt   | [BaiduYun](https://pan.baidu.com/s/1KWFARNWuFXW2pCrpuxt20g?) (Access code: kysy) or [GoogleDrive](https://drive.google.com/file/d/1mAv4LlfPImMc_G9I5fvPdvT9lvrBQQ-e/view?usp=sharing) |

## S4 Availability of data
The EPFLdataset was downloaded from the EPFL website https://www.epfl.ch/labs/cvlab/data/data-em/.
<br>

## S5 Citation
If you find our code or paper useful for your research, please cite our work:

```bibtex
@article{yang2026veminr,
  title={vEMINR: Ultra-Fast Isotropic Reconstruction for Volume Electron Microscopy With Implicit Neural Representation},
  author={Yang, Jibin and Huo, Jie and Liu, Muyu and Feng, Chenjie and Zhang, Yan and Pan, Gang and Meng, Wenjia and Han, Renmin},
  journal={Advanced Science},
  pages={e11922},
  year={2026},
  publisher={Wiley Online Library}
}
```




