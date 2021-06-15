_base_ = './vfnet_r50_fpn_mstrain_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
checkpoint = 'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth'