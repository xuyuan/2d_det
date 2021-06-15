_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'
checkpoint = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(type='Res2Net', depth=101, scales=4, base_width=26))