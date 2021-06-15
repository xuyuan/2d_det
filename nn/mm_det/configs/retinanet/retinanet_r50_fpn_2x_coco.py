_base_ = './retinanet_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
checkpoint = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'