from .basic import Sequential


def dm_nfnet(name, pretrained):
    from timm import create_model
    imagenet_pretrained = pretrained == 'imagenet'
    model = create_model(name, pretrained=imagenet_pretrained)
    stem = model.stem

    layer0 = Sequential(stem.conv1,
                        stem.act2, stem.conv2,
                        stem.act3, stem.conv3)
    layers = [layer0]

    in_channels = stem.conv4.out_channels
    for s in model.stages:
        s.in_channels = in_channels
        s.out_channels = s[0].conv3.out_channels
        layers.append(s)
        in_channels = s.out_channels

    layers[1] = Sequential(stem.act4, stem.conv4, layers[1])
    layers[-1] = Sequential(layers[-1], model.final_conv, model.final_act)
    n_pretrained = len(layers) if imagenet_pretrained else 0
    return layers, False, n_pretrained
