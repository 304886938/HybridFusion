# HybridFusion: A Universal Multimodal Data Fusion Framework for 3-D Detection and Tracking
This repo implements the paper HybridFusion: A Universal Multimodal Data Fusion Framework for 3-D Detection and Tracking.

We built our implementation upon MMdetection3D. The major part of the code is in the directory `HybridFusion`. 

## Environment
### Prerequisite
<ol>
<li> mmcv-full>=1.5.2, <=1.7.0 </li>
<li> mmdet>=2.24.0, <=3.0.0</li>
<li> mmseg>=0.20.0, <=1.0.0</li>
</ol>

### Installation

There is no neccesary to install mmdet3d separately, please install based on this repo:

```
pip3 install -v -e .
```


### Data

 Please follow the mmdet3d to process the data. 

## Train

For example, to train HybirdFusion with Camer-LiDAR-Radar-Fusion on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075_900q.py 8
```

For LiDAR-Cam and Cam-Radar version, we need pre-trained model. 

The Cam-Radar uses DETR3D model as pre-trained model, please check [DETR3D](https://github.com/WangYueFt/detr3d).

The LiDAR-Cam uses fused LiDAR-only and Cam-only model as pre-trained model. You can use

```
python tools/fuse_model.py --img <cam checkpoint path> --lidar <lidar checkpoint path> --out <out model path>
```
to fuse cam-only and lidar-only models.

## Evaluate

For example, to evalaute FUTR3D with LiDAR-cam on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/lidar_0075_cam_res101.py ../lidar_cam.pth 8 --eval bbox
```
```

Contact: Xuanyao Chen at: `xuanyaochen19@fudan.edu.cn` or `ixyaochen@gmail.com`


