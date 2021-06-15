import torch
import math
from pathlib import Path
import numpy as np
from torch.nn import functional
from PIL import Image
from mmcv import Config, imnormalize, impad_to_multiple
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from ..basenet.basic import InputNormalization
from ..detnet import DetNet
from .models.necks.pafpnx import PAFPNX


def get_config_file(config_path):
    # relative path first
    config_path = Path(config_path)
    if config_path.exists():
        return config_path

    config_file = Path(__file__).parent / "configs" / config_path
    return config_file


def model_zoo_get(config_path, classnames: list, pretrained: str = 'coco', freeze_at: int = 2, frozen_bn: bool = True):
    """replicated `model_zoo.get` with additional config modification"""
    cfg_file = get_config_file(config_path)

    cfg = Config.fromfile(cfg_file)

    cfg.model.backbone.frozen_stages = freeze_at - 1
    cfg.model.backbone.norm_eval = frozen_bn

    if not classnames and 'classnames' in cfg:
        # read classnames from config
        classnames = cfg.classnames

    if classnames:
        if 'roi_head' in cfg.model:
            bbox_head = cfg.model.roi_head.bbox_head
        else:
            bbox_head = cfg.model.bbox_head

        if isinstance(bbox_head, list):
            for bh in bbox_head:
                bh.num_classes = len(classnames)
        else:
            bbox_head.num_classes = len(classnames)
    else:
        from mmdet.datasets.coco import CocoDataset
        classnames = CocoDataset.CLASSES

    if 'roi_head' in cfg.model:
        if 'mask_head' in cfg.model.roi_head:
            cfg.model.roi_head.mask_head = None
        if 'semantic_head' in cfg.model.roi_head:
            cfg.model.roi_head.semantic_head = None

    if pretrained != 'imagenet':
       cfg.model.pretrained = None

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    if pretrained == 'coco':
        print(f'load pretrained weights {cfg.checkpoint}')
        load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    return cfg, model, classnames


class MMDet(DetNet):
    size_divisible = 32

    def __init__(self, arch, classnames, freeze_pretrained=2, frozen_bn=True, pretrained='coco'):
        super().__init__()

        self.cfg, self.model, self.classnames = model_zoo_get(arch, classnames, pretrained=pretrained, freeze_at=freeze_pretrained, frozen_bn=frozen_bn)
        self.input_norm = InputNormalization(mean=self.cfg.img_norm_cfg.mean,
                                             std=self.cfg.img_norm_cfg.std, inplace=False)

    def divisible_padding(self, size):
        """return size to pad for divisible"""
        if self.size_divisible > 1:
            return int(math.ceil(size / self.size_divisible) * self.size_divisible) - size
        return 0

    def forward(self, x):
        ori_shape = (x.shape[2], x.shape[3], x.shape[1])  # original shape of the image as a tuple (h, w, c)

        if self.size_divisible > 1:
            pad_bottom = self.divisible_padding(ori_shape[0])
            pad_right = self.divisible_padding(ori_shape[1])
            if pad_bottom > 0 or pad_right > 0:
                x = functional.pad(x, [0, pad_right, 0, pad_bottom])

        pad_shape = (x.shape[2], x.shape[3], x.shape[1])  # image shape after padding
        img_shape = pad_shape  # shape of the image input to the network as a tuple (h, w, c)
        img_metas = [dict(img_shape=img_shape,
                          ori_shape=ori_shape,
                          pad_shape=pad_shape,
                          scale_factor=1,
                          flip=False,
                          flip_direction='')]

        net_param = next(self.parameters())
        x = x.to(net_param)

        x = self.input_norm(x)  # image normalization

        return x, img_metas * len(x)

    def _convert_bbox_results(self, ltrbs, img_shape):
        lt = ltrbs[:, 0:2] / img_shape
        rb = ltrbs[:, 2:4] / img_shape
        xy = (lt + rb) * 0.5
        wh = rb - lt
        return torch.from_numpy(np.hstack((ltrbs[:, -1:], xy, wh)))

    def _predict(self, x, **kwargs):
        with torch.no_grad():
            single_image_input = False
            if isinstance(x, Image.Image):
                x = np.float32(x)
                x = torch.as_tensor(x.transpose(2, 0, 1))
                x.unsqueeze_(0)
                single_image_input = True

            if len(x) > 1:
                raise NotImplementedError('mmdet only supports batch 1 predict')

            img, img_metas = self.forward(x)

            bbox_results = self.model.simple_test(img, img_metas)  # list[list[np.ndarray]]: N, C, n, 4
            ori_shape = img_metas[0]['ori_shape']
            ori_shape = np.asarray([ori_shape[1], ori_shape[0]])  # (w, h)
            bbox_results = [self._convert_bbox_results(bbox, ori_shape) for bbox in bbox_results[0]]

            if not single_image_input:
                bbox_results = [bbox_results]

            return bbox_results

    def criterion(self, args):
        return self.loss

    def loss(self, inputs, target):
        img, img_metas = inputs
        gt_bboxes = target['boxes']
        # In MMDetection 2.0, label “K” means background,
        # and labels [0, K-1] correspond to the K = num_categories object categories.
        gt_labels = [labels - 1 for labels in target['labels']]
        losses = self.model.forward_train(img, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

        losses = {k: v for k, v in losses.items() if 'loss' in k}  # filter non loss part, 'acc' was returned!

        # sum sublist loss
        for k, v in losses.items():
            if isinstance(v, list):
                losses[k] = sum(v)
        return losses

