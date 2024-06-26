# DDT: Dual-branch Deformable Transformer for image denoising (ICME2023)

Kangliang Liu, Xiangcheng Du, Sijie Liu, Yingbin Zheng, Xingjiao Wu, Cheng jin

Fudan University, Videt Technology

[[paper](https://ieeexplore.ieee.org/document/10220014)]

---



## Installation

1. Clone our repository

   ```
   git clone https://github.com/Merenguelkl/DDT.git
   cd DDT
   ```
2. Make conda environment

   ```
   conda create -n DDT python=3.8
   conda activate DDT
   ```

   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   pip3 install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
   pip3 install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips timm fvcore
   ```
4. Install basicsr

   ```
   python setup.py develop --no_cuda_ext
   ```
## Data Preparation

1. Real Denoising

   Download SIDD dataset and generate patches from full-resolution training images

   ```
   python download_data.py --data train --noise real
   python download_data.py --noise real --data test --dataset SIDD
   python generate_patches_sidd.py 
   ```

2. Synthetic Denoising

   Download training (DIV2K, Flickr2K, WED, BSD) and testing datasets and generate patches from full-resolution training images

   ```
   python download_data.py --data train-test --noise gaussian
   python generate_patches_dfwb.py 
   ```

## Training

1. Real Denoising

   ```
   ./train.sh Denoising/Options/RealDenoising_DDT.yml
   ```

2. Synthetic Denoising

   ```
   ./train.sh Denoising/Options/GaussianColorDenoising_DDT.yml
   ```

**Note:** This training script uses 4 GPUs by default. To use any other number of GPUs, modify ```DDT/train.sh``` and ```DDT/Denoising/Options/*.yml``` 

## Evaluation & Visualization
Download pretrained model from [Google Drive](https://drive.google.com/file/d/1GGeb_-NcUQkHeJkfoTttUYhk4N1Tqb97/view?usp=sharing]).

Update path in ```eval.py```.

```
python eval.py
```

The visualized outputs will be gererated in ```DDT/visualization```

## Citation
If you are interested in this work, please consider citing:

    @inproceedings{liu2023ddt,
     title={DDT: Dual-branch deformable transformer for image denoising},
     author={Liu, Kangliang and Du, Xiangcheng and Liu, Sijie and Zheng, Yingbin and Wu, Xingjiao and Jin, Cheng},
     booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
     pages={2765--2770},
     year={2023},
     organization={IEEE}
    }
  
## Acknowledgment
This code is based on the [BasicSR](https://github.com/XPixelGroup/BasicSR) and  [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome works.

## Contact
If you have any question or suggestion, please contact klliu21@m.fudan.edu.cn.
