### Set up the python environment

```
# python 3.7 or 3.8 are both fine. 
conda create -n nhp python=3.x
conda activate nhp

# make sure that the pytorch cuda is consistent with the system cuda
# install pytorch via conda
https://pytorch.org/get-started/locally/

# install additional requirements
pip install -r requirements.txt

# install requirement for mesh rendering
pip install PyOpenGL PyOpenGL_accelerate
sudo apt-get install freeglut3-dev

# install spconv
# Sparse Convolution now provides easy installation via pip
# Go to the SpConv Github and install the proper version
https://github.com/traveller59/spconv
```

### Set up datasets



#### ZJU-Mocap dataset

1. To request the ZJU-Mocap dataset download link, please fill in the [agreement](https://zjueducn-my.sharepoint.com/:b:/g/personal/pengsida_zju_edu_cn/EUPiybrcFeNEhdQROx4-LNEBm4lzLxDwkk1SBcNWFgeplA?e=BGDiQh), and email to the Neural Body author (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn).
2. Create a soft link:
    ```
    ROOT=/path/to/Neural_Human_Performer
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```
3. Our framework uses the newly fitted SMPL parameters and vertices of the Neural Body.  
Rename the `new_params` folder into `params` and the `new_vertices` folder into `vertices`.
4. Download the visibility maps at [here](https://drive.google.com/file/d/1I6GpC9BRiJq9IhOCMUE8oZBHcd-pbCfg/view?usp=sharing).
5. Create a soft link:
    ```
    ROOT=/path/to/Neural_Human_Performer
    cd $ROOT/data
    ln -s /path/to/zju_rasterization zju_rasterization
    ```
6. Change the file name format of the subject 313 and 315:
    ```
    python lib/utils/modify_313_315_filename.py
    ```
#### Download SMPL
1. Go to [SMPL website](https://smpl.is.tue.mpg.de/) and sign up.
2. Download the SMPL 1.0.0
3. Create a directory with the following structure:
```bash
data
└── smplx
   ├── smpl
   │   ├── SMPL_FEMALE.pkl
   │   └── SMPL_MALE.pkl
   │   └── SMPL_NEUTRAL.pkl
   └── J_regressor_body25.npy
```