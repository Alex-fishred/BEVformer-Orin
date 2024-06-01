<div align="center">   
  
# BEVFormer: a Cutting-edge Baseline for Camera-based Detection
</div>

https://github.com/Alex-fishred/BEVformer-Orin/assets/76515591/eb17a585-99c4-4699-950a-988889fc80f2





> - [HackMD上的教學]()


# News
- [2024/5/20]: 🚀BEVFormer-Orin 釋出
</br>


# Abstract
As autonomous driving technology continues to advance, the bird's-eye view offers an intuitive representation of the environment for autonomous vehicles, aiding in the enhancement of their decision-making capabilities. However, BEV detection systems require significant computational resources, making implementation on embedded platforms challenging. This study aims to optimize existing BEV detection models using pruning and quantization techniques to achieve a BEV detection system on the NVIDIA Orin platform. Experimental results show that when the model is deployed on the Orin single GPU embedded platform, the inference speed of the original architecture model is 1.05 FPS, while the inference speed of the pruned and quantized model can reach 2.32 FPS, nearly a 120% increase in speed. This study successfully lightens the BEV detection model through pruning and quantization techniques, achieving implementation on the NVIDIA Orin platform and improving inference speed.


# Methods
![本研究架構圖2](https://github.com/Alex-fishred/BEVformer-Orin/assets/76515591/3d55144c-2085-470e-a216-0f0b3bc44886)


# Getting Started
- [Installation](install.md) 
- [Prepare Dataset](prepare_dataset.md)
- [Run and Eval](getting_started.md)

# Model Zoo

| Backbone | Method | Lr Schd | NDS| mAP|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| R50 | BEVFormer-tiny_fp16 | 24ep | 35.9|25.7 | - |[config](projects/configs/bevformer_fp16/bevformer_tiny_fp16.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-small | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-base | 24ep | 51.7|41.6 |28500M |[config](projects/configs/bevformer/bevformer_base.py) | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.log) |



# Acknowledgement

Many thanks to these excellent open source projects:
- [dd3d](https://github.com/TRI-ML/dd3d) 
- [detr3d](https://github.com/WangYueFt/detr3d) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

