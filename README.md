# HybridFusion: A Universal Multimodal Data Fusion Framework for 3-D Detection and Tracking
## Demo
<img src="demo/GIF.gif" width="900"/> 
Due to file size limitations, only a 3-second preview is shown above.  
The full-length demonstration video is available at: https://youtu.be/el_5jtxDO2I

## Introduction
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

For example, to train HybirdFusion with Camera-LiDAR-Radar-Fusion on 8 GPUs, please use

```
bash tools/dist_train.sh HybridFusion/FusionDetection/configs/cam_lidar_radar.py 8
```

## Evaluate

For example, to evalaute HybirdFusion with Camera-LiDAR-Radar-Fusion on 8 GPUs, please use

```
bash tools/dist_test.sh HybridFusion/FusionDetection/configs/cam_lidar_radar.py 8 --eval bbox
```

Contact: Cheng Zhang at: 2112004009@stmail.ujs.edu.cn or chengzhang971011@gmail.com
