"""
Designing Network Design Spaces
"""


def regnet(name, pretrained):
    from pycls import models
    imagenet_pretrained = pretrained == 'imagenet'
    model_name, model_size = name.split('-')
    if model_name == 'regnetx':
        model = models.regnetx(model_size, imagenet_pretrained)
    elif model_name == 'regnety':
        model = models.regnety(model_size, imagenet_pretrained)
    else:
        raise NotImplementedError(model_name)

    layers = [model.stem, model.s1, model.s2, model.s3, model.s4]

    layers[0].out_channels = 32
    for l in layers[1:]:
        l.out_channels = l.b1.f.a.out_channels

    n_pretrained = len(layers) if imagenet_pretrained else 0
    return layers, True, n_pretrained
