from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torchvision
from .basic import Sequential, DummyModule, get_num_of_channels
from .utils import load_pretrained


def resnet(name, pretrained, in_channels=3, pretrained_prefix=None, **kwargs):
    pool_in_2nd = name[0].isupper()
    name = name.lower()

    imagenet_pretrained = pretrained == 'imagenet'

    if 'replace_stride_with_dilation' in kwargs:
        if name in ('resnet18', 'resnet34') or (name in ('resnet50', 'resnet101') and pretrained in ('coco_fcn', 'coco_deeplabv3')):
            replace_stride_with_dilation = kwargs['replace_stride_with_dilation']
            warnings.warn(f"{name} ignore replace_stride_with_dilation={replace_stride_with_dilation}")
            del kwargs['replace_stride_with_dilation']

    # ignore deprecated args
    if 'drop_last' in kwargs:
        del kwargs['drop_last']

    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet50':
        if pretrained == 'coco_fcn':
            model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, **kwargs).backbone
        elif pretrained == 'coco_deeplabv3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, **kwargs).backbone
        elif pretrained == 'coco_fasterrcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs).backbone.body
        elif pretrained == 'coco_maskrcnn':
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, **kwargs).backbone.body
        elif pretrained == 'virtex':
            model = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True, **kwargs)
        else:
            model = torchvision.models.resnet50(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet101':
        if pretrained == 'coco_fcn':
            model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, **kwargs).backbone
        elif pretrained == 'coco_deeplabv3':
            model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, **kwargs).backbone
        else:
            model = torchvision.models.resnet101(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=imagenet_pretrained, **kwargs)
    elif '_ibn_'  in name:
        # Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net", ECCV2018.
        # https://github.com/XingangPan/IBN-Net
        model = torch.hub.load('XingangPan/IBN-Net', name, pretrained=imagenet_pretrained)
    elif name == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=imagenet_pretrained, **kwargs)
    elif name in ('resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d'):
        model = resnext_wsl(name, pretrained=imagenet_pretrained, **kwargs)
    elif name == 'wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'wide_resnet101_2':
        model = torchvision.models.wide_resnet101_2(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnest50':
        from resnest.torch import resnest50
        model = resnest50(pretrained=imagenet_pretrained)
    elif name == 'resnest101':
        from resnest.torch import resnest101
        model = resnest101(pretrained=imagenet_pretrained)
    elif name == 'resnest200':
        from resnest.torch import resnest200
        model = resnest200(pretrained=imagenet_pretrained)
    elif name == 'resnest269':
        from resnest.torch import resnest269
        model = resnest269(pretrained=imagenet_pretrained)
    elif name == 'oct_resnet50':
        from .octconv import oct_resnet50
        model = oct_resnet50(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'oct_resnet101':
        from .octconv import oct_resnet101
        model = oct_resnet101(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'oct_resnet152':
        from .octconv import oct_resnet152
        model = oct_resnet152(pretrained=imagenet_pretrained, **kwargs)
    elif name in ('seresnet18', 'seresnet34'):
        import timm
        model = timm.create_model('legacy_' + name, pretrained=imagenet_pretrained)
        model.conv1 = model.layer0.conv1
        model.bn1 = model.layer0.bn1
        model.relu = model.layer0.relu1
        model.maxpool = model.pool0
    elif name in ('resnet18d', 'resnet26d', 'resnet34d', 'resnet50d', 'resnet101d', 'resnet152d', 'resnet200d', 'resnet200d_320'):
        import timm
        model = timm.create_model(name, pretrained=imagenet_pretrained)
        model.relu = model.act1
    else:
        raise NotImplementedError(name)

    load_pretrained(model, pretrained, key_prefix=pretrained_prefix)

    model.conv1 = change_in_channels(model.conv1, in_channels, pretrained)

    # for maxpool1 in ResNext_IBN
    maxpool = model.maxpool if hasattr(model, 'maxpool') else model.maxpool1
    if pool_in_2nd:
        layer0 = Sequential(model.conv1, model.bn1, model.relu)
    else:
        layer0 = Sequential(model.conv1, model.bn1, model.relu, maxpool)

    layer0[-1].out_channels = get_num_of_channels(model.conv1)

    def get_out_channels_from_resnet_block(layer):
        block = layer[-1]
        block_name = block.__class__.__name__
        if 'BasicBlock' in block_name or 'SEResNetBlock' in block_name:
            return block.conv2.out_channels
        elif 'Bottleneck' in block_name:
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block.__class__))

    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4

    layer1.out_channels = layer1[-1].out_channels = get_out_channels_from_resnet_block(model.layer1)
    layer2.out_channels = layer2[-1].out_channels = get_out_channels_from_resnet_block(model.layer2)
    layer3.out_channels = layer3[-1].out_channels = get_out_channels_from_resnet_block(model.layer3)
    layer4.out_channels = layer4[-1].out_channels = get_out_channels_from_resnet_block(model.layer4)

    if pool_in_2nd:
        layer1 = nn.Sequential(maxpool, layer1)
        layer1.out_channels = layer1[-1].out_channels

    n_pretrained = 5 if imagenet_pretrained else 0

    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained



def resnext_wsl(arch, pretrained, progress=True, **kwargs):
    """
    models trained in weakly-supervised fashion on 940 million public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset.
    https://github.com/facebookresearch/WSL-Images/
    """
    from torch.hub import load_state_dict_from_url
    from torchvision.models.resnet import ResNet, Bottleneck

    model_args = {'resnext101_32x8d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=8),
                  'resnext101_32x16d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=16),
                  'resnext101_32x32d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=32),
                  'resnext101_32x48d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=48)}

    args = model_args[arch]
    args.update(kwargs)
    model = ResNet(**args)

    if pretrained:
        model_urls = {
            'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
            'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
            'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
            'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
        }
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

    return model


def se_net(name, pretrained, in_channels=3):
    pool_in_2nd = name.startswith('S')
    name = name.lower()

    abn = name.endswith('_abn')
    if abn:
        name = name[:-4]

    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None

    if name in ('se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154'):
        import pretrainedmodels
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    layer0 = [senet.layer0[i] for i in range(len(senet.layer0))]
    if pool_in_2nd:
        layer0 = layer0[:-1]
    layer0 = nn.Sequential(*layer0)
    layer0 = change_in_channels(layer0, in_channels, pretrained)

    layer1 = senet.layer1
    if pool_in_2nd:
        layer1 = nn.Sequential(senet.layer0[-1], layer1)
    layer1.out_channels = layer1[-1].out_channels = senet.layer1[-1].conv3.out_channels
    layer0.out_channels = layer0[-1].out_channels = senet.layer1[0].conv1.in_channels

    layer2 = senet.layer2
    layer2.out_channels = layer2[-1].out_channels = senet.layer2[-1].conv3.out_channels

    layer3 = senet.layer3
    layer3.out_channels = layer3[-1].out_channels = senet.layer3[-1].conv3.out_channels

    layer4 = senet.layer4
    layer4.out_channels = layer4[-1].out_channels = senet.layer4[-1].conv3.out_channels

    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def change_in_channels(model, in_channels, pretrained):
    if isinstance(model, nn.Sequential):
        model[0] = change_in_channels(model[0], in_channels, pretrained)
        return model

    assert isinstance(model, nn.Conv2d)
    if in_channels == model.in_channels:
        return model

    bias = model.bias is not None
    new_conv1 = nn.Conv2d(in_channels, model.out_channels, kernel_size=model.kernel_size,
                          stride=model.stride, padding=model.padding, bias=bias)

    if pretrained:
        # copy weights
        new_conv1.weight.data[:] = 0
        new_conv1.weight.data[:, :model.in_channels] = model.weight.data[:, :in_channels]
        if bias:
            new_conv1.bias.data[:] = model.bias.data[:]
    return new_conv1

