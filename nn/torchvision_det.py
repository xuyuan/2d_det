import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from .detnet import DetNet


class TorchVisionDet(DetNet):
    def __init__(self, arch, classnames, pretrained=False):
        super().__init__()
        pretrained = pretrained == 'coco'
        pretrained_backbone = pretrained == 'imagenet'
        if classnames is None:
            from ..data.coco import COCO_CLASSNAMES
            self.classnames = COCO_CLASSNAMES
        else:
            self.classnames = classnames
        self.num_classes = len(self.classnames) + 1

        if arch == 'fasterrcnn_resnet50_fpn':
            self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
            if self.num_classes != 91:
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        elif arch == 'retinanet_resnet50_fpn':
            from torchvision.models.detection import retinanet_resnet50_fpn
            self.model = retinanet_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
            if self.num_classes != 91:
                from torchvision.models.detection.retinanet import RetinaNetHead
                in_features = self.model.backbone.out_channels
                num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
                self.model.head = RetinaNetHead(in_features, num_anchors, self.num_classes)
        else:
            raise NotImplementedError(arch)

    def forward(self, x):
        return x / 255  # To range 0-1

    def _predict(self, x, **kwargs):
        """
        During inference, the model requires only the input tensors, and returns the post-processed
        predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
        follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
        """
        with torch.no_grad():

            single_image_input = False
            if isinstance(x, Image.Image):
                x = np.float32(x)
                x = torch.as_tensor(x.transpose(2, 0, 1)).unsqueeze(0)
                single_image_input = True

            net_param = next(self.parameters())
            x = x.to(net_param)

            x = self.forward(x)
            detections = self.model(x)

            output = []
            for det in detections:
                scores = det['scores'].unsqueeze(1)
                box = det['boxes']
                image_size = box.new((x.shape[-1], x.shape[-2]))
                center = (box[:, 0:2] + box[:, 2:4]) / 2 / image_size
                wh = (box[:, 2:4] - box[:, 0:2]) / image_size
                bbox = torch.cat((scores, center, wh), dim=1)
                labels = det['labels']
                bbox_cls = []
                for cls in range(1, self.num_classes):
                    bbox_cls.append(bbox[labels == cls])
                output.append(bbox_cls)
            if single_image_input:
                output = output[0]

            return output

    def criterion(self, args):
        return self.loss

    def loss(self, output, target):
        """
            During training, the model expects both the input tensors, as well as a targets (list of dictionary),
            containing:
                - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
                  between ``0`` and ``H`` and ``0`` and ``W``
                - labels (``Int64Tensor[N]``): the class label for each ground-truth box
            """
        batch_size = len(output)
        images = []
        targets = []
        for i in range(batch_size):
            labels = target['labels'][i]
            if labels.numel() > 0:
                targets.append({'boxes': target['boxes'][i],
                               'labels': labels})
                images.append(output[i])

        return self.model(images, targets)


class TorchVisionTrans(object):
    def __call__(self, sample):
        ret = {'image_id': sample['image_id'],
               'input': sample['input'],
               'boxes': sample['bbox'][:, :4],
               'labels': sample['bbox'][:, 4].astype(int)}
        return ret
