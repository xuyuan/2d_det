

# ImageNet pretrained models

| model               | origin               | FLOPS | No. Param | Top 5 Acc | Top 1 Acc |
|:--------------------|:---------------------|:------|:----------|:----------|:----------|
| vgg11               | torchvision          |       |           |     88.63 |     69.93 |
| vgg19               | torchvision          |       |           |     90.88 |     72.38 |
| vgg19_bn            | torchvision          |       |           |     91.85 |     74.24 |
| resnet18            | torchvision          |       |           |     89.08 |     69.76 |
| resnet34            | torchvision          |       |           |     91.42 |     73.30 |
| resnet50            | torchvision          |       |           |     92.97 |     76.15 |
| resnet101           | torchvision          |       |           |     93.56 |     77.37 |
| resnet152           | torchvision          |       |           |     94.06 |     78.31 |
| resnet18d           | timm                 |       |           |     90.69 |     72.27 |
| resnet26d           | timm                 |       |           |     93.14 |     76.68 |
| resnet34d           | timm                 |       |           |     93.38 |     77.11 |
| resnet50d           | timm                 |       |           |     95.15 |     80.54 |
| resnet101d          | timm                 |       |           |     96.06 |     82.31 |
| resnet152d          | timm                 |       |           |     96.35 |     83.13 |
| resnet200d          | timm                 |       |           |     96.48 |     83.24 |
| resnet200d_320      | timm                 |       |           |     96.81 |     83.96 |
| oct_resnet50        | octconv.pytorch      |  2.4G |     25.6M |     93.66 |     77.64 |
| oct_resnet101       | octconv.pytorch      |    4G |     44.5M |     94.33 |     78.79 |
| oct_resnet152       | octconv.pytorch      |  5.6G |     60.2M |     94.48 |     79.26 |
| efficientnet-b0     | efficientnet_pytorch |       |      5.3M |           |     76.3  |
| efficientnet-b1     | efficientnet_pytorch |       |      7.8M |           |     78.8  |
| efficientnet-b2     | efficientnet_pytorch |       |      9.2M |           |     79.8  |
| efficientnet-b3     | efficientnet_pytorch |       |       12M |           |     81.1  |
| efficientnet-b4     | efficientnet_pytorch |       |       19M |           |     82.6  |
| efficientnet-b5     | efficientnet_pytorch |       |       30M |           |     83.3  |
| efficientnet-b6     | efficientnet_pytorch |       |       43M |           |     84.0  |
| efficientnet-b7     | efficientnet_pytorch |       |       66M |           |     84.4  |
| seresnet18          | timm                 |       |     11.8M |     90.33 |     71.74 |
| seresnet34          | timm                 |       |       22M |     92.12 |     74.81 |
| se_resnet50         | pretrainedmodels     |       |           |     93.75 |     77.63 |
| se_resnext50_32x4d  | pretrainedmodels     |       |           |     94.43 |     79.08 |
| se_resnext101_32x4d | pretrainedmodels     |       |           |     95.03 |     80.24 |
| senet154            | pretrainedmodels     |       |           |     95.49 |     81.30 |
| resnext50_32x4d     | torchvision          |       |           |     93.7  |     77.62 |
| resnext101_32x4d    | pretrainedmodels     |       |           |     93.89 |     78.19 |
| resnext101_64x4d    | pretrainedmodels     |       |           |     94.25 |     78.95 |
| resnext101_32x8d    | WSL-Images Instagram |  16B  |       88M |     96.4  |     82.2  |
| resnext101_32x16d   | WSL-Images Instagram |  36B  |      193M |     97.2  |     84.2  |
| resnext101_32x32d   | WSL-Images Instagram |  87B  |      466M |     97.5  |     85.1  |
| resnext101_32x48d   | WSL-Images Instagram | 153B  |      829M |     97.6  |     85.4  |
| resnet50_ibn_a      | IBN-Net              |       |           |     93.59 |     77.24 |
| resnet101_ibn_a     | IBN-Net              |       |           |     94.41 |     78.61 |
| wide_resnet50_2     | torchvision          |       |           |     94.09 |     78.51 |
| wide_resnet101_2    | torchvision          |       |           |     94.28 |     78.84 |
| densenet121         | torchvision          |       |           |     92.17 |     74.65 |
| densenet161         | torchvision          |       |           |     93.80 |     77.65 |
| densenet169         | torchvision          |       |           |     93.00 |     76.00 |
| densenet201         | torchvision          |       |           |     93.57 |     77.20 |
| resnest50           | resnest              |       |           |           |     81.03 |
| resnest101          | resnest              |       |           |           |     82.83 |
| resnest200          | resnest              |       |           |           |     83.84 |
| resnest269          | resnest              |       |           |           |     84.54 |
| mobilenet_v2        | tonylins/pytorch-mobilenet-v2 |  | 3.5M  |           |     71.80 |
| Mobilenet_v2        | torchvision          |       |           |     90.29 |     71.88 |
| mobilenet_v3        | rwightman/gen-efficientnet-pytorch | 219M | 5.5M | 92.71 | 75.63 |
| regnetx-200MF       | pycls                |  0.2B |      2.7M |           |     68.9  |
| regnetx-400MF       | pycls                |  0.4B |      5.2M |           |     72.6  |
| regnetx-600MF       | pycls                |  0.6B |      6.2M |           |     74.1  |
| regnetx-800MF       | pycls                |  0.8B |      7.3M |           |     75.2  |
| regnetx-1.6GF       | pycls                |  1.6B |      9.2M |           |     77.0  |
| regnetx-3.2GF       | pycls                |  3.2B |     15.3M |           |     78.3  |
| regnetx-4.0GF       | pycls                |  4.0B |     22.1M |           |     78.6  |
| regnetx-6.4GF       | pycls                |  6.5B |     26.2M |           |     79.2  |
| regnetx-8.0GF       | pycls                |  8.0B |     39.6M |           |     79.3  |
| regnetx-12GF        | pycls                | 12.1B |     46.1M |           |     79.7  |
| regnetx-16GF        | pycls                | 15.9B |     54.3M |           |     80.0  |
| regnetx-32GF        | pycls                | 31.7B |    107.8M |           |     80.5  |
| regnety-200MF       | pycls                |  0.2B |      3.2M |           |     70.3  |
| regnety-400MF       | pycls                |  0.4B |      4.3M |           |     74.1  |
| regnety-600MF       | pycls                |  0.6B |      6.1M |           |     75.5  |
| regnety-800MF       | pycls                |  0.8B |      6.3M |           |     76.3  |
| regnety-1.6GF       | pycls                |  1.6B |     11.2M |           |     77.9  |
| regnety-3.2GF       | pycls                |  3.2B |     19.4M |           |     78.9  |
| regnety-4.0GF       | pycls                |  4.0B |     20.6M |           |     79.4  |
| regnety-6.4GF       | pycls                |  6.4B |     30.6M |           |     79.9  |
| regnety-8.0GF       | pycls                |  8.0B |     39.2M |           |     79.9  |
| regnety-12GF        | pycls                | 12.1B |     51.8M |           |     80.3  |
| regnety-16GF        | pycls                | 15.9B |     83.6M |           |     80.4  |
| regnety-32GF        | pycls                | 31.3B |    145.0M |           |     80.9  |
| hrnet18s_v1         | HRNet                | 1.49G |     13.2M |      90.7 |     72.3  |
| hrnet18s_v2         | HRNet                | 2.42G |     15.6M |      92.4 |     75.1  |
| hrnet18             | HRNet                | 3.99G |     21.3M |      93.4 |     76.8  |
| hrnet30             | HRNet                | 7.55G |     37.7M |      94.2 |     78.2  |
| hrnet32             | HRNet                | 8.31G |     41.2M |      94.2 |     78.5  |
| hrnet40             | HRNet                | 11.8G |     57.6M |      94.5 |     78.9  |
| hrnet44             | HRNet                | 13.9G |     67.1M |      94.4 |     78.9  |
| hrnet48             | HRNet                | 16.1G |     77.5M |      94.4 |     79.3  |
| hrnet64             | HRNet                | 26.9G |    128.1M |      94.6 |     79.5  |
| dm_nfnet_f0         | timm                 |       |           |           |     83.4  |
| dm_nfnet_f1         | timm                 |       |           |           |     84.6  |
| dm_nfnet_f2         | timm                 |       |           |           |     85.1  |
| dm_nfnet_f3         | timm                 |       |           |           |     85.6  |
| dm_nfnet_f4         | timm                 |       |           |           |     85.8  |
| dm_nfnet_f5         | timm                 |       |           |           |     86.1  |
| dm_nfnet_f6         | timm                 |       |           |           |     86.3  |