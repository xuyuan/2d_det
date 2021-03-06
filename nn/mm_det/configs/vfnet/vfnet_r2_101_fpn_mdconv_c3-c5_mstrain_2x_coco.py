_base_ = './vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
checkpoint = 'https://drontheimerstr.synology.me/model_zoo/mmdetection/vfnet_r2_101_dcn_ms_2x_51.1-76ad4bbc.pth'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))