from .__init__ import BASENET_CHOICES, create_pyramid_backbone

if __name__ == '__main__':
    import argparse
    from torchsummary import summary
    import torch
    parser = argparse.ArgumentParser(description="show network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', nargs=1, choices=BASENET_CHOICES, help='load saved model')
    parser.add_argument('--pretrained', default=False, type=str, help='pretrained dataset, e.g. imagenet, instagram, voc, coco, oid')
    parser.add_argument('--pretrained-prefix', type=str, help='key prefix in pretrained state dict')
    parser.add_argument('--activation', type=str, choices=('mish', 'relu', 'swish', 'hardswish'), help='convert activation functions')
    parser.add_argument('--frozen-batchnorm', action='store_true', help='Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.')
    parser.add_argument('--input', type=lambda s: tuple(int(v) for v in s.split('x')), default=(3, 224, 224),
                        help='input image size, e.g. 3x224x224')
    parser.add_argument("-v", "--verbose", action='count', default=0, help="level of debug messages")

    args = parser.parse_args()
    model = create_pyramid_backbone(args.model[0], args.pretrained, activation=args.activation, in_channels=args.input[0],
                                    frozen_batchnorm=args.frozen_batchnorm, pretrained_prefix=args.pretrained_prefix)

    if args.verbose == 1:
        print(model)

    if args.verbose == 2:
        summary(model, input_size=args.input, device='cpu')

    print('out_channels', model.out_channels)

    x = torch.zeros(args.input).unsqueeze(0)
    model.eval()
    Y = model(x)
    shapes = [x.shape] + [y.shape for y in Y]
    shapes = [str(s) for s in shapes]
    print('\n-->'.join(shapes))
