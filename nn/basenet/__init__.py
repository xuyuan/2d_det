import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from .basic import Sequential, ConvBnRelu, ConvRelu, get_num_of_channels, Swish, HardSwish, Mish, convert_activation, AdaptiveConcatPool2d, InputNormalization
from .utils import load_pretrained


BASENET_CHOICES = ('vgg11', 'vgg13', 'vgg16', 'vgg19',
                   'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                   #
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152',
                   #
                   'resnet18d', 'resnet26d', 'resnet34d', 'resnet50d', 'resnet101d', 'resnet152d', 'resnet200d', 'resnet200d_320',
                   'Resnet18d', 'Resnet26d', 'Resnet34d', 'Resnet50d', 'Resnet101d', 'Resnet152d', 'Resnet200d', 'Resnet200d_320',
                   #
                   'resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a',
                   'resnet18_ibn_b', 'resnet34_ibn_b', 'resnet50_ibn_b', 'resnet101_ibn_b',
                   'resnext101_ibn_a', 'se_resnet101_ibn_a',
                   #
                   'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
                   'Resnext50_32x4d', 'Resnext101_32x8d', 'Resnext101_32x16d', 'Resnext101_32x32d', 'Resnext101_32x48d',
                   'resnext101_32x4d', 'resnext101_64x4d',
                   'Resnext101_32x4d', 'Resnext101_64x4d',
                   #
                   'seresnet18', 'seresnet34',
                   'Seresnet18', 'Seresnet34',
                   #
                   'se_resnet50', 'se_resnet101', 'se_resnet152',
                   'Se_resnet50', 'Se_resnet101', 'Se_resnet152',
                   #
                   'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154',
                   'Se_resnext50_32x4d', 'Se_resnext101_32x4d', 'Senet154',
                   #
                   'wide_resnet50_2', 'wide_resnet101_2',
                   'Wide_resnet50_2', 'Wide_resnet101_2',
                   #
                   'resnest50', 'resnest101', 'resnest200', 'resnest269',
                   'Resnest50', 'Resnest101', 'Resnest200', 'Resnest269',
                   #
                   'densenet121', 'densenet161', 'densenet169', 'densenet201',
                   #
                   'oct_resnet50', 'oct_resnet101', 'oct_resnet152',
                   #
                   'efficientnet-b0', 'Efficientnet-b0', 'efficientnet-b0-advprop', 'Efficientnet-b0-advprop',
                   'efficientnet-b1', 'Efficientnet-b1', 'efficientnet-b1-advprop', 'Efficientnet-b1-advprop',
                   'efficientnet-b2', 'Efficientnet-b2', 'efficientnet-b2-advprop', 'Efficientnet-b2-advprop',
                   'efficientnet-b3', 'Efficientnet-b3', 'efficientnet-b3-advprop', 'Efficientnet-b3-advprop',
                   'efficientnet-b4', 'Efficientnet-b4', 'efficientnet-b4-advprop', 'Efficientnet-b4-advprop',
                   'efficientnet-b5', 'Efficientnet-b5', 'efficientnet-b5-advprop', 'Efficientnet-b5-advprop',
                   'efficientnet-b6', 'Efficientnet-b6', 'efficientnet-b6-advprop', 'Efficientnet-b6-advprop',
                   'efficientnet-b7', 'Efficientnet-b7', 'efficientnet-b7-advprop', 'Efficientnet-b7-advprop',
                   'efficientnet-b8-advprop', 'Efficientnet-b8-advprop',
                   #
                   'regnetx-200MF', 'regnetx-400MF', 'regnetx-600MF', 'regnetx-800MF', 'regnetx-1.6GF', 'regnetx-3.2GF',
                   'regnetx-4.0GF', 'regnetx-6.4GF', 'regnetx-8.0GF', 'regnetx-12GF', 'regnetx-16GF', 'regnetx-32GF',
                   'regnety-200MF', 'regnety-400MF', 'regnety-600MF', 'regnety-800MF', 'regnety-1.6GF', 'regnety-3.2GF',
                   'regnety-4.0GF', 'regnety-6.4GF', 'regnety-8.0GF', 'regnety-12GF', 'regnety-16GF', 'regnety-32GF',
                   #
                   'bninception',
                   #
                   'mobilenet_v2', 'Mobilenet_v2',
                   'mobilenet_v3',
                   #
                   'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                   #
                   'squeezenet1_0', 'squeezenet1_1',
                   #
                   'darknet',
                   #
                   'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6',
                   #
                   'hrnet18s_v1', 'hrnet18s_v2', 'hrnet18', 'hrnet30', 'hrnet32', 'hrnet40', 'hrnet44', 'hrnet48', 'hrnet64',
                   )


MODEL_ZOO_URL = 'https://drontheimerstr.synology.me/model_zoo/'

MODEL_URLS = {
    'vgg16': {'voc': 'SSDv2_vgg16_c21-96ae2bf2.pth',
              'coco': 'SSDretina_vgg16_c81-de29503d.pth'},
    'resnet50': {#'voc':  'SSDretina_resnet50_c21-fb6036d1.pth',  # SSDretina_resnet50_c81-a584ead7.pth pretrained
                 'voc': 'SSDretina_resnet50_c21-1c85a349.pth',  # SSDretina_resnet50_c501-06095077.pth pretrained
                 'coco': 'SSDretina_resnet50_c81-a584ead7.pth',
                 'oid': 'SSDretina_resnet50_c501-06095077.pth',
                 'visdrone': 'SSDdrone_resnet50_c12-9777e250.pth',
                 'efffeu': 'SSDdrone_resnet50_c6_job2854-99e7388e.pth' # SSDretina_resnet50_c6_job2833_399f6639.pth
                },
    'resnet101': {'coco': 'SSDretina_resnet101_c81-d515d740.pth'},
    'resnext101_32x4d': {'coco': 'SSDretina_resnext101_32x4d_c81-fdb37546.pth'},
    'se_resnext50_32x4d': {'coco': 'SSDretina_se_resnext50_32x4d-c280aa00.pth'},
    'se_resnext101_32x4d': {'coco': 'SSDretina_se_resnext101_32x4d_c81-14b8f37.pth'},
    'senet154': {'coco': 'SSDretina_senet154_c81-e940bc59.pth'},
    'mobilenet_v2': {'imagenet': 'mobilenet_v2-ecbe2b56.pth',
                     'coco': 'SSDretina_mobilenet_v2_c81-4d8aaad4.pth'},
    'darknet': {'coco': 'pytorch_yolov3-60bd5e05.pth'},

    'hrnet18s_v1': {'imagenet': 'hrnet_w18_small_model_v1-9d2048da.pth'},
    'hrnet18s_v2': {'imagenet': 'hrnet_w18_small_model_v2-d3923c3d.pth'},
    'hrnet18': {'imagenet': 'hrnetv2_w18_imagenet_pretrained-00eb2006.pth'},
    'hrnet30': {'imagenet': 'hrnetv2_w30_imagenet_pretrained-11fb7730.pth'},
    'hrnet32': {'imagenet': 'hrnetv2_w32_imagenet_pretrained-dc9eeb4f.pth'},
    'hrnet40': {'imagenet': 'hrnetv2_w40_imagenet_pretrained-ed0b031c.pth'},
    'hrnet44': {'imagenet': 'hrnetv2_w44_imagenet_pretrained-8c55086c.pth'},
    'hrnet48': {'imagenet': 'hrnetv2_w48_imagenet_pretrained-0efec102.pth'},
    'hrnet64': {'imagenet': 'hrnetv2_w64_imagenet_pretrained-41ed675b.pth'},
}

MODEL_URLS['Resnet50'] = MODEL_URLS['resnet50']


def load_from_file_or_model_zoo(filename):
    if isinstance(filename, str) and filename.startswith('model_zoo:'):
        model_url = filename.replace('model_zoo:', MODEL_ZOO_URL)
        data = model_zoo.load_url(model_url, map_location=lambda storage, loc: storage)
    else:
        data = torch.load(filename, map_location=lambda storage, loc: storage)
    return data


def get_model_zoo_url(basenet, pretrained):
    return MODEL_ZOO_URL + MODEL_URLS[basenet][pretrained]


def vgg_base_extra(bn):
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    block = ConvBnRelu if bn else ConvRelu
    conv6 = block(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = block(1024, 1024, kernel_size=1)
    return [pool5, conv6, conv7]


def vgg(name, pretrained, ssd_variant=True):
    if name == 'vgg11':
        net_class = torchvision.models.vgg11
    elif name == 'vgg13':
        net_class = torchvision.models.vgg13
    elif name == 'vgg16':
        net_class = torchvision.models.vgg16
    elif name == 'vgg19':
        net_class = torchvision.models.vgg19
    elif name == 'vgg11_bn':
        net_class = torchvision.models.vgg11_bn
    elif name == 'vgg13_bn':
        net_class = torchvision.models.vgg13_bn
    elif name == 'vgg16_bn':
        net_class = torchvision.models.vgg16_bn
    elif name == 'vgg19_bn':
        net_class = torchvision.models.vgg19_bn
    else:
        raise RuntimeError("unknown model {}".format(name))

    imagenet_pretrained = pretrained == 'imagenet'
    vgg = net_class(pretrained=imagenet_pretrained)

    # for have exact same layout as original paper
    if ssd_variant and name == 'vgg16':
        vgg.features[16].ceil_mode = True

    bn = name.endswith('bn')
    layers = []
    l = []
    for i in range(len(vgg.features) - 1):
        if isinstance(vgg.features[i], nn.MaxPool2d):
            layers.append(l)
            l = []
        l.append(vgg.features[i])
    if ssd_variant:
        l += vgg_base_extra(bn=bn)
    layers.append(l)

    block = ConvBnRelu if bn else ConvRelu
    if ssd_variant:
        # layers of feature scaling 2**5
        layer5 = [block(1024, 256, 1, 1, 0),
                  block(256, 512, 3, 2, 1)]
    else:
        layer5 = [nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                  block(512, 1024, kernel_size=3, padding=1),
                  block(1024, 1024, kernel_size=3, padding=1)]
    layers.append(layer5)

    layers = [Sequential(*l) for l in layers]
    n_pretrained = 4 if imagenet_pretrained else 0
    return layers, bn, n_pretrained


def resnext(name, pretrained):
    pool_in_2nd = name.startswith('R')
    name = name.lower()

    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        import pretrainedmodels
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    resnext_features = resnext.features
    layer0 = [resnext_features[i] for i in range(4)]
    if pool_in_2nd:
        layer0 = layer0[:-1]
    layer0 = nn.Sequential(*layer0)
    layer0.out_channels = layer0[-1].out_channels = 64

    layer1 = resnext_features[4]
    if pool_in_2nd:
        layer1 = nn.Sequential(resnext_features[3], layer1)
    layer1.out_channels = layer1[-1].out_channels = 256

    layer2 = resnext_features[5]
    layer2.out_channels = layer2[-1].out_channels = 512

    layer3 = resnext_features[6]
    layer3.out_channels = layer3[-1].out_channels = 1024

    layer4 = resnext_features[7]
    layer4.out_channels = layer4[-1].out_channels = 2048
    n_pretrained = 5 if imagenet_pretrained else 0

    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def densenet(name, pretrained, pretrained_prefix=None, **kwargs):
    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
    if name == 'densenet121':
        net = torchvision.models.densenet121(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'densenet161':
        net = torchvision.models.densenet161(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'densenet169':
        net = torchvision.models.densenet169(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'densenet201':
        net = torchvision.models.densenet201(pretrained=imagenet_pretrained, **kwargs)
    else:
        raise NotImplementedError(name)

    load_pretrained(net, pretrained, key_prefix=pretrained_prefix)

    layer0 = Sequential(net.features.conv0, net.features.norm0, net.features.relu0, net.features.pool0)
    layer1 = Sequential(net.features.denseblock1, net.features.transition1)
    layer2 = Sequential(net.features.denseblock2, net.features.transition2)
    layer3 = Sequential(net.features.denseblock3, net.features.transition3)
    layer4 = Sequential(net.features.denseblock4, net.features.norm5)

    layers = [layer0, layer1, layer2, layer3, layer4]

    n_pretrained = len(layers) if imagenet_pretrained else 0
    return layers, True, n_pretrained


def efficientnet(name, pretrained, in_channels=3, **kwargs):
    from efficientnet_pytorch import EfficientNet
    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None

    advprop = False
    if name.endswith('-advprop'):
        advprop = True
        name = name[:-len('-advprop')]

    if imagenet_pretrained:
        model = EfficientNet.from_pretrained(name.lower(), advprop=advprop)
    else:
        model = EfficientNet.from_name(name.lower())

    if 'memory_efficient' in kwargs:
        if hasattr(model, 'set_swish'):
            model.set_swish(memory_efficient=kwargs['memory_efficient'])

    return convert_efficientnet(model, name, pretrained, in_channels)


def convert_efficientnet(model, name, pretrained, in_channels=3):
    from efficientnet_pytorch.model import MBConvBlock
    layers = []

    # set input channels
    conv_stem = model._conv_stem
    if in_channels != 3:
        conv_type = type(conv_stem)
        image_size = model._global_params.image_size
        conv_stem = conv_type(in_channels=in_channels, out_channels=conv_stem.out_channels,
                              kernel_size=conv_stem.kernel_size, stride=conv_stem.stride,
                              bias=conv_stem.bias, image_size=image_size)

    stem = Sequential(conv_stem, model._bn0, Swish(inplace=True))

    blocks = [stem]
    for idx, block in enumerate(model._blocks):
        if block._depthwise_conv.stride[0] == 2:
            l = nn.Sequential(*blocks)
            layers.append(l)
            blocks = []

        drop_connect_rate = model._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(idx) / len(model._blocks)
        block.drop_connect_rate = drop_connect_rate
        blocks.append(block)
    if blocks:
        layers.append(nn.Sequential(*blocks))

    layers += [Sequential(model._conv_head, model._bn1, Swish(inplace=True))]

    for l in layers:
        if isinstance(l[-1], MBConvBlock):
            l.out_channels = l[-1]._bn2.num_features
        else:
            l.out_channels = get_num_of_channels(l)

    if name[0] == 'E':
        # make sure output size are different
        layers_merged = nn.Sequential(layers[4], layers[5])
        layers_merged.out_channels = layers[5].out_channels
        layers = layers[:4] + [layers_merged]
    else:
        layers = layers[:5]

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def darknet(pretrained):
    from .darknet import KitModel as DarkNet
    net = DarkNet()
    if pretrained:
        state_dict = model_zoo.load_url(get_model_zoo_url('darknet', 'coco'), map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
    n_pretrained = 3 if pretrained else 0
    return [net.model0, net.model1, net.model2], True, n_pretrained


def mobilenet_v2(pretrained):
    from .mobile_net_v2 import MobileNetV2
    net = MobileNetV2()
    if pretrained:
        state_dict = model_zoo.load_url(get_model_zoo_url('mobilenet_v2', pretrained), map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
        pretrained = True
    else:
        pretrained = False

    splits = [0, 2, 4, 7, 14]
    layers = [net.features[i:j] for i, j in zip(splits, splits[1:] + [len(net.features)-1])]
    for l in layers:
        l.out_channels = l[-1].out_channels

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def torch_vision_mobilenet_v2(pretrained):
    from torchvision.models import mobilenet_v2
    imagenet_pretrained = pretrained == 'imagenet'
    net = mobilenet_v2(imagenet_pretrained)
    splits = [0, 2, 4, 7, 14]
    layers = [net.features[i:j] for i, j in zip(splits, splits[1:] + [len(net.features)-1])]
    for l in layers:
        l.out_channels = l[-1].conv[-2].out_channels

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def mobilenet_v3(pretrained):
    imagenet_pretrained = pretrained == 'imagenet'
    net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'mobilenetv3_rw', pretrained=imagenet_pretrained)

    stem = Sequential(net.conv_stem, net.bn1, Swish())
    layers = [stem]

    splits = [0, 2, 3, 5, 7]
    blocks = [net.blocks[i:j] for i, j in zip(splits[:-1], splits[1:])]
    for block, out_channels in zip(blocks, [24, 40, 112, 960]):
        block.out_channels = out_channels
        layers.append(block)

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def shufflenet_v2(name, pretrained):
    from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    imagenet_pretrained = pretrained == 'imagenet'

    if name == 'shufflenet_v2_x0_5':
        net = shufflenet_v2_x0_5(imagenet_pretrained)
    elif name == 'shufflenet_v2_x1_0':
        net = shufflenet_v2_x1_0(imagenet_pretrained)
    elif name == 'shufflenet_v2_x1_5':
        net = shufflenet_v2_x1_5(imagenet_pretrained)
    elif name == 'shufflenet_v2_x2_0':
        net = shufflenet_v2_x2_0(imagenet_pretrained)
    else:
        raise NotImplementedError(name)

    stage1 = nn.Sequential(net.conv1, net.maxpool)
    layers = [stage1, net.stage2, net.stage3, net.stage4, net.conv5]
    for stage, out in zip(layers, net._stage_out_channels):
        stage.out_channels = out

    n_pretrained = len(layers) if imagenet_pretrained else 0
    return layers, True, n_pretrained


def squeezenet1(name, pretrained):
    from torchvision.models import squeezenet1_0, squeezenet1_1
    imagenet_pretrained = pretrained == 'imagenet'

    if name == 'squeezenet1_0':
        net = squeezenet1_0(imagenet_pretrained)
        splits = [0, 3, 7, 12, 13]
        out_channels = [96, 256, 512, 512]
    elif name == 'squeezenet1_1':
        net = squeezenet1_1(imagenet_pretrained)
        splits = [0, 3, 6, 9, 13]
        out_channels = [64, 128, 256, 512]
    else:
        raise NotImplementedError(name)

    layers = [net.features[i:j] for i, j in zip(splits[:-1], splits[1:])]
    for block, o in zip(layers, out_channels):
        block.out_channels = o

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


class MockModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.backbone = nn.ModuleList(layers)


def load_pretrained_weights(layers, name, dataset_name):
    state_dict = model_zoo.load_url(get_model_zoo_url(name, dataset_name))
    mock_module = MockModule(layers)
    mock_module.load_state_dict(state_dict, strict=False)


def create_basenet(name, pretrained, activation=None, frozen_batchnorm=False, frozen_batchnorm_requires_grad=False,
                   in_channels=3, trainable_layers=None, frozen_layers=0,
                   **kwargs):
    """
    Parameters
    ----------
    name: model name
    pretrained: dataset name

    Returns
    -------
    list of modules, is_batchnorm, num_of_pretrained_module
    """
    if name.startswith('vgg'):
        layers, bn, n_pretrained = vgg(name, pretrained)
    elif name.lower() in ('resnext101_32x4d', 'resnext101_64x4d'):
        layers, bn, n_pretrained = resnext(name, pretrained)
    elif name.lower().startswith('resnet') or name.lower().startswith('resnext') or name.lower().startswith("wide_resnet") or name.lower().startswith('resnest') or name.lower().startswith('oct_resnet') or name.lower().startswith('seresnet') or ('_ibn_' in name):
        from .resnet import resnet
        layers, bn, n_pretrained = resnet(name, pretrained, in_channels, **kwargs)
    elif name.lower().startswith('se'):
        from .resnet import se_net
        layers, bn, n_pretrained = se_net(name, pretrained, in_channels)
    elif name.lower().startswith('densenet'):
        layers, bn, n_pretrained = densenet(name, pretrained, **kwargs)
    elif name.lower().startswith('efficientnet'):
        layers, bn, n_pretrained = efficientnet(name, pretrained, in_channels=in_channels, memory_efficient=True)
    elif name.lower().startswith('regnet'):
        from .regnet import regnet
        layers, bn, n_pretrained = regnet(name, pretrained)
    elif name == 'bninception':
        from .bnincetion import bninception
        layers, bn, n_pretrained = bninception(pretrained)
    elif name == 'darknet':
        layers, bn, n_pretrained = darknet(pretrained)
    elif name == 'mobilenet_v2':
        layers, bn, n_pretrained = mobilenet_v2(pretrained)
    elif name == 'Mobilenet_v2':
        layers, bn, n_pretrained = torch_vision_mobilenet_v2(pretrained)
    elif name == 'mobilenet_v3':
        layers, bn, n_pretrained = mobilenet_v3(pretrained)
    elif name.startswith('shufflenet_v2'):
        layers, bn, n_pretrained = shufflenet_v2(name, pretrained)
    elif name.startswith('squeezenet1'):
        layers, bn, n_pretrained = squeezenet1(name, pretrained)
    elif name.startswith('dm_nfnet_f'):
        from .nfnet import dm_nfnet
        layers, bn, n_pretrained = dm_nfnet(name, pretrained)
    else:
        raise NotImplementedError(name)

    if pretrained in ('voc', 'coco', 'oid'):
        load_pretrained_weights(layers, name, pretrained)
        n_pretrained = len(layers)

    if activation:
        layers = [convert_activation(activation, l) for l in layers]

    if frozen_batchnorm:
        from .batch_norm import FrozenBatchNorm2d
        if frozen_batchnorm is True:
            layers = [FrozenBatchNorm2d.convert_frozen_batchnorm(l, frozen_batchnorm_requires_grad) for l in layers]
        elif isinstance(frozen_batchnorm, int):
            for i in range(frozen_batchnorm):
                layers[i] = FrozenBatchNorm2d.convert_frozen_batchnorm(layers[i], frozen_batchnorm_requires_grad)
        else:
            raise RuntimeError(f'unknown type {frozen_batchnorm}')

    if trainable_layers is not None:
        if frozen_layers > 0:
            raise RuntimeError('trainable_layers and frozen_layers are set at the same time! it is not possible.')
        frozen_layers = len(layers) - trainable_layers

    for i in range(frozen_layers):
        for parameter in layers[i].parameters():
            parameter.requires_grad_(False)

    return layers, bn, n_pretrained


class PyramidBackbone(nn.ModuleList):
    def forward(self, x):
        y = []
        for m in self:
            x = m(x)
            y.append(x)
        return y

    @property
    def out_channels(self):
        return [m.out_channels for m in self]


def create_pyramid_backbone(name, pretrained, activation=None, frozen_batchnorm=False, in_channels=3,
                            trainable_layers=None, frozen_layers=0,
                            **kwargs):
    if name.startswith('hrnet'):
        from .hrnet import hrnet
        assert in_channels == 3

        model = hrnet(name, **kwargs)

        weights = None
        if pretrained:
            model_url = get_model_zoo_url(name, pretrained)
            weights = model_zoo.load_url(model_url, map_location=lambda storage, loc: storage)

        model.init_weights(weights)

        if activation:
            raise NotImplementedError(activation)

        if frozen_batchnorm:
            raise NotImplemented

        if trainable_layers is not None:
            if frozen_layers > 0:
                raise RuntimeError('trainable_layers and frozen_layers are set at the same time! it is not possible.')
            frozen_layers = 4 - trainable_layers

        if frozen_layers > 0:
            raise NotImplemented

        return model

    layers, bn, n_pretrained = create_basenet(name, pretrained, activation=activation, frozen_batchnorm=frozen_batchnorm, in_channels=in_channels,
                                              trainable_layers=trainable_layers, frozen_layers=frozen_layers,
                                              **kwargs)
    return PyramidBackbone(layers)


def create_fpn_backbone(name, pretrained, activation=None, frozen_batchnorm=False, in_channels=3,
                        trainable_layers=None, frozen_layers=0,
                        out_channels=256, n_fpn_layers=4, bifpn_stack=0,
                        **kwargs):
    layers, bn, n_pretrained = create_basenet(name=name, pretrained=pretrained, activation=activation,
                                              frozen_batchnorm=frozen_batchnorm, in_channels=in_channels,
                                              trainable_layers=trainable_layers, frozen_layers=frozen_layers,
                                              **kwargs)
    if len(layers) < n_fpn_layers:
        raise NotImplementedError(f'FPN requires >={n_fpn_layers} layers backbone, but {name} has {len(layers)} layers!')

    backbone = nn.ModuleDict()
    for i in range(0, len(layers)):
        backbone[f'layer{i}'] = layers[i]

    in_channels_list = []
    return_layers = {}
    for name in list(backbone.keys())[-n_fpn_layers:]:
        return_layers[name] = str(len(in_channels_list))
        in_channels_list.append(backbone[name].out_channels)

    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    model = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    if bifpn_stack > 0:
        from .bifpn import BiFPN
        model.fpn = BiFPN(in_channels_list, out_channels, stack=bifpn_stack)

    return model

