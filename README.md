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
![method](figs/arch.png "model arch")


# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

# Model Zoo

| Backbone | Method | Lr Schd | NDS| mAP|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| R50 | BEVFormer-tiny_fp16 | 24ep | 35.9|25.7 | - |[config](projects/configs/bevformer_fp16/bevformer_tiny_fp16.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-small | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-base | 24ep | 51.7|41.6 |28500M |[config](projects/configs/bevformer/bevformer_base.py) | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.log) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1-base | 24ep | 42.6 | 35.1 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py) | [model/log](https://drive.google.com/drive/folders/1nts_1XxAagCEN_Ub7W2f-507SiDdVS_u?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1-base | 48ep | 43.9 | 35.9 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-48ep.py) | [model/log](https://drive.google.com/drive/folders/1nts_1XxAagCEN_Ub7W2f-507SiDdVS_u?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1 | 24ep | 45.3 | 38.1 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py) | [model/log](https://drive.google.com/drive/folders/1uVzQCJq6gYbRLhBde09yzEBeU5l1hAxk?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1 | 48ep | 46.5 | 39.5 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-48ep.py) | [model/log](https://drive.google.com/drive/folders/1uVzQCJq6gYbRLhBde09yzEBeU5l1hAxk?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t2 | 24ep | 51.8 | 42.0 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-24ep.py) | [model/log](https://drive.google.com/drive/folders/1bSyuFWxfJSIidGV7bC8jx2NR7idRN9-s?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t2 | 48ep | 52.6 | 43.1 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py) | [model/log](https://drive.google.com/drive/folders/1bSyuFWxfJSIidGV7bC8jx2NR7idRN9-s?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t8 | 24ep | 55.3 | 46.0 | 40392M |[config](projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py) | [model/log](https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto?usp=sharing) |

The Baidu Driver Link for (BEVFormerV2 model and log)[https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv] is here.

# Catalog
- [ ] BEVFormerV2 HyperQuery
- [ ] BEVFormerV2 Optimization, including memory, speed, inference.
- [x] BEVFormerV2 Release
- [ ] BEV Segmentation checkpoints
- [ ] BEV Segmentation code
- [x] 3D Detection checkpoints
- [x] 3D Detection code
- [x] Initialization


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{li2022bevformer,
  title={BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng}
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
@article{Yang2022BEVFormerVA,
  title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
  author={Chenyu Yang and Yuntao Chen and Haofei Tian and Chenxin Tao and Xizhou Zhu and Zhaoxiang Zhang and Gao Huang and Hongyang Li and Y. Qiao and Lewei Lu and Jie Zhou and Jifeng Dai},
  journal={ArXiv},
  year={2022},
}
```

# Acknowledgement

Many thanks to these excellent open source projects:
- [dd3d](https://github.com/TRI-ML/dd3d) 
- [detr3d](https://github.com/WangYueFt/detr3d) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

