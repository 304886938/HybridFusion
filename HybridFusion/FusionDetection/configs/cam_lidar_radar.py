_base_ = [
    '../../../../configs/_base_/datasets/vod-3d-3class.py',
    '../../../../configs/_base_/default_runtime.py'
]

class_names = ['Pedestrian', 'Cyclist', 'Car']

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)
voxel_size = [0.075, 0.075, 0.2]
pillar_size = [0.16, 0.16, 4]
point_cloud_range = [0, -25, -3, 50, 25, 1]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)


model = dict(
    type='FusionDetection',
    use_lidar=True,
    use_camera=True,
    use_radar=True,
    use_grid_mask=True,
    freeze_backbone=True,
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=['stage2', 'stage3', 'stage4', 'stage5']),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False)),
    vtransform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=128,
        image_size=[256, 704],
        feature_size=[32,88],
        xbound=[0, 50, 0.4],
        ybound=[-25, 25, 0.4],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),    
    lidar_pts_voxel_layer=dict(
        max_num_points=10, 
        voxel_size=voxel_size, 
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    lidar_pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    lidar_pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[21, 667, 667],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    lidar_pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    lidar_pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    radar_pts_pillar_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        pillar_size=pillar_size,
        max_pillars=(16000, 40000)  # (training, testing) max_voxels
    ),
    radar_pts_pillar_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        pillar_size=pillar_size,
        point_cloud_range=point_cloud_range),
    radar_pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    radar_pts_backbone=dict(
        type='DiNAT',
        embed_dim=128,
        mlp_ratio=2.0,
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.6,
        kernel_size=7,
        dilations=[[1, 28, 1], [1, 7, 1, 14], [1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7], [1, 3, 1, 3, 1]],
        layer_scale=1e-5),
    radar_feature_adjustment=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    radar_pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),

    pts_bbox_head=dict(
        type='FusionDetectionHead',
        fuse_img=True,
        num_views=num_views,
        in_channels_img=256,
        out_size_factor_img=4,
        lidar_num_proposals=100,
        cam_num_proposals=50,
        radar_num_proposals=50,
        auxiliary=True,
        num_classes=3,
        num_heads=8,
        num_encoder_layers=1,
        modalitiies = 3,
        dec_n_points=4,
        num_decoder_layers=1,
        modalities_fusion=1,
        enc_n_points=4,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='FusionDetectionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[0, -25, -10.0, 50, 25, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),

    # training and testing settings
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range,
        pc_range=point_cloud_range,
        grid_size=[1440, 1440, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0)))),
    test_cfg=dict(pts=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=[0, -25, -10.0, 50, 25, 10.0]],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
        max_num=300,
        score_threshold=0,
        post_center_range=[0, -25, -10.0, 50, 25, 10.0],
    )))

dataset_type = 'VoDDataset'
data_root = 'data/vod/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        #type='CBGSDataset',
        #dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vod_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
        #),
    val=dict(
        pipeline=test_pipeline, 
        classes=class_names, 
        ann_file=data_root + 'vod_infos_val.pkl',
        modality=input_modality,),
    test=dict(
        pipeline=test_pipeline, 
        classes=class_names, 
        ann_file=data_root + 'vod_infos_val.pkl',
        modality=input_modality,))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1)

find_unused_parameters=True

#custom_hooks = [dict(type='FadeOjectSampleHook', num_last_epochs=5)]
runner = dict(type='EpochBasedRunner', max_epochs=6)
optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_neck': dict(lr_mult=0.1),
            'vtransform': dict(lr_mult=0.1),
            'lidar_pts_middle_encoder': dict(lr_mult=0.1),
            'lidar_pts_backbone': dict(lr_mult=0.1),
            'lidar_pts_neck': dict(lr_mult=0.1),
            'radar_pts_pillar_encoder': dict(lr_mult=0.1),
            'radar_pts_middle_encoder': dict(lr_mult=0.1),
            'radar_pts_backbone': dict(lr_mult=0.1),
            'radar_feature_adjustment': dict(lr_mult=0.1),
            'radar_pts_neck': dict(lr_mult=0.1),
        }),

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
