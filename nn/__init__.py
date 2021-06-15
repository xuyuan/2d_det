from ..trainer.utils import add_save
from .ssd import SSD_CONFIG, SingleShotDetector
from .detnet import NoneNet
from .basenet import load_from_file_or_model_zoo


__all__ = ['register_model', 'create', 'load']

REGISTERED_MODEL = {}


def register_model(arch_prefix, model):
    REGISTERED_MODEL[arch_prefix] = model


@add_save
def create(arch, classnames, basenet='vgg16', pretrained='imagenet', freeze_pretrained=0, frozen_bn=False,
           score_thresh_test=0):

    # compatibility for old models
    if classnames and classnames[0] == 'background':
        #print('Warning: removing "background" from classnames')
        classnames = classnames[1:]

    if arch in SSD_CONFIG.keys():
        net = SingleShotDetector(classnames, basenet=basenet, version=arch, pretrained=pretrained, frozen_bn=frozen_bn)
        net.set_pretrained_frozen(freeze_pretrained)
    elif arch.split(':')[0].lower() == 'xsd':
        from .xsd import XSD
        cls_add = arch.split(':')[0] == 'xsd'
        net = XSD(arch.split(':')[1], classnames=classnames, freeze_pretrained=freeze_pretrained, frozen_bn=frozen_bn,
                  pretrained=pretrained, cls_add=cls_add)
    elif arch == 'pointdet':
        from .point_det import create_model
        net = create_model(classnames, basenet=basenet, frozen_bn=frozen_bn)
    elif arch.split(':')[0] == 'torchvision':
        from .torchvision_det import TorchVisionDet
        net = TorchVisionDet(arch.split(':')[1], classnames, pretrained=pretrained)
    elif arch.split(':')[0] == 'detectron2':
        from .detectron2_det import Detectron2Det
        net = Detectron2Det(arch.split(':')[1], classnames, freeze_pretrained, frozen_bn, pretrained=pretrained,
                            score_thresh_test=score_thresh_test)
    elif arch.split(':')[0] == 'mmdet':
        from .mm_det import MMDet
        net = MMDet(arch.split(':')[1], classnames, freeze_pretrained, frozen_bn, pretrained=pretrained)
    elif arch.startswith('yolov5'):
        from .yolov5 import YoloV5
        net = YoloV5(arch)
    elif arch.split(':')[0] in REGISTERED_MODEL:
        net = REGISTERED_MODEL[arch.split(':')[0]](arch.split(':')[1], classnames=classnames,
                                                   freeze_pretrained=freeze_pretrained,
                                                   frozen_bn=frozen_bn,
                                                   pretrained=pretrained)
    else:
        raise NotImplementedError(arch)

    return net


def load(filename, score_thresh_test=0):
    print('load {}'.format(filename))

    if isinstance(filename, str):
        if (filename.startswith('torchvision:')
                or filename.startswith("detectron2:")
                or filename.startswith("mmdet:")
                or filename.startswith('yolov5')):
            return create(filename, classnames=None, pretrained='coco', score_thresh_test=score_thresh_test)
        elif filename == 'none':
            return NoneNet()

    data = load_from_file_or_model_zoo(filename)

    if ('args' not in data) and ('kwargs' not in data):
        # loading old model
        net = SingleShotDetector(classnames=data['classes'], basenet=data['basenet'], version=data['version'],
                                 pretrained=False)
    else:
        data['kwargs']['pretrained'] = None

        if score_thresh_test > 0:
            data['kwargs']['score_thresh_test'] = score_thresh_test

        net = create(*data['args'], **data['kwargs'])
    net.load_state_dict(data['state_dict'])
    return net
