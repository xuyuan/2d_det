
import warnings
from collections.abc import Iterable
import numbers
import random

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as tvt
import skimage.transform
from skimage.exposure import equalize_adapthist
import cv2


_pil_interpolation_to_str = {
    Image.NEAREST: 'NEAREST',
    Image.BILINEAR: 'BILINEAR',
    Image.BICUBIC: 'BICUBIC',
    Image.LANCZOS: 'LANCZOS',
    None: None
}


def cv2_border_mode_value(border):
    border_value = 0
    if border == 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    elif border == 'reflect':
        border_mode = cv2.BORDER_REFLECT_101
    else:
        border_mode = cv2.BORDER_CONSTANT
        border_value = border
    return dict(borderMode=border_mode, borderValue=border_value)


# normalization for `torchvision.models`
# see http://pytorch.org/docs/master/torchvision/models.html
TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
TORCH_VISION_NORMALIZE = tvt.Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)
TORCH_VISION_DENORMALIZE = tvt.Normalize(mean=-TORCH_VISION_MEAN/TORCH_VISION_STD, std=1/TORCH_VISION_STD)


def _random(name, data=(-1, 1)):
    if name == 'choice':
        return np.random.choice(data)
    elif name == 'uniform':
        return np.random.uniform(data[0], data[1])
    else:
        raise NotImplementedError(name)


class VisionTransform(object):
    def __repr__(self): return self.__class__.__name__ + '()'

    def __call__(self, sample):
        """
        :param sample: dict of data, key is used to determine data type, e.g. image, bbox, mask
        :return: transformed sample in dict
        """
        sample = self.pre_transform(sample)
        output_sample = {}
        for k, v in sample.items():
            if k == 'input':
                if isinstance(v, Image.Image) or torch.is_tensor(v):
                    output_sample[k] = self.transform_image(v)
                elif isinstance(v, Iterable):
                    output_sample[k] = [self.transform_image(vi) for vi in v]
                else:
                    output_sample[k] = self.transform_image(v)
            elif k == 'image_h':
                output_sample[k] = self.transform_image_h(v)
            elif k == 'bbox':
                output_sample[k] = self.transform_bbox(v) if len(v) > 0 else v
            elif k.startswith('mask'):
                output_sample[k] = self.transform_mask(v, k)
            elif k == 'transformed':
                output_sample[k] = sample[k] + [repr(self)]
            else:
                output_sample[k] = sample[k]

        output_sample = self.post_transform(output_sample)
        return output_sample

    def redo(self, sample):
        raise NotImplementedError

    @staticmethod
    def get_input_size(sample):
        width = height = None
        if 'input' in sample:
            image = sample['input']
            if isinstance(image, Iterable):
                image = image[0]
            width, height = image.size
        else:
            for k in sample:
                if k.startswith('mask'):
                    mask = sample[k]
                    height = mask.shape[0]
                    width = mask.shape[1]
                    break

        assert width is not None  # sample must have input or mask
        return width, height


    def pre_transform(self, sample):
        return sample

    def transform_image(self, image):
        return image

    def transform_image_h(self, image):
        raise NotImplementedError

    def transform_bbox(self, bbox):
        return bbox

    def transform_mask(self, mask, name):
        return mask

    def post_transform(self, sample):
        # if 'input' in sample:
        #     w, h = sample['input'].size
        #     for k, v in sample.items():
        #         if k.startswith('mask'):
        #             if v.shape[0] != h or v.shape[1] != w:
        #                 raise RuntimeError(f'{repr(self)}\n mask size mismatch {(h, w)} != {(v.shape)}')
        return sample

    @staticmethod
    def size_from_number_or_iterable(size, n=2):
        if isinstance(size, numbers.Number):
            return (size,) * n
        elif isinstance(size, Iterable):
            return size
        else:
            raise RuntimeError(type(size))


class Resize(VisionTransform):
    BILINEAR = Image.BILINEAR

    def __init__(self, size=None, scale_factor=None, interpolation=None, max_size=None):
        """
        :param size: (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio.
        :param scale_factor: (sequence or int): multiplier for spatial size.
        :param max_size: int: limit the maximum size of longer edge of image, it is useful to avoid OOM
        :param interpolation: interpolation method of PIL, `None` means random
        """
        self.size = size
        self.scale_factor = scale_factor
        self.interpolation = interpolation
        self.max_size = max_size

        assert size is None or scale_factor is None

        if self.max_size:
            assert isinstance(size, int) or isinstance(scale_factor, int)  # max_size only works with not fixed size

    def redo(self, sample):
        return self(sample)

    def __repr__(self):
        arg_str = []
        if self.size is not None:
            arg_str += [f'size={self.size}']
        if self.scale_factor is not None:
            arg_str += [f'scale_factor={self.scale_factor}']
        if self.interpolation:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
            arg_str += [f'interpolation={interpolate_str}']
        if self.max_size:
            arg_str += [f'max_size={self.max_size}']

        arg_str = ', '.join(arg_str)
        return self.__class__.__name__ + '(' + arg_str + ')'

    @staticmethod
    def compute_scaled_image_size(sample, size, scale_factor=None, max_size=None):
        w, h = VisionTransform.get_input_size(sample)

        if isinstance(scale_factor, Iterable):
            size = (int(i*s) for i, s in zip((h, w), scale_factor))
        elif scale_factor is not None:
            size = int(min(w, h) * scale_factor)

        if isinstance(size, int):
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            if max_size:
                if oh > max_size:
                    ow = int(max_size / oh * ow)
                    oh = max_size
                if ow > max_size:
                    oh = int(max_size / ow * oh)
                    ow = max_size

            return oh, ow
        return size

    def pre_transform(self, sample):
        self.out_size = Resize.compute_scaled_image_size(sample, self.size, self.scale_factor, self.max_size)
        self._image_interpolation = self.interpolation if self.interpolation is not None else random.randint(0, 5)
        if 'image_h' in sample:
            self.out_size_h = tuple(int(s1 * s2 / s0) for s0, s1, s2 in zip(sample['input'].size[::-1], sample['image_h'].size[::-1], self.out_size))
        return sample

    def transform_image(self, image):
        if self.out_size[0] != image.height or self.out_size[1] != image.width:
            return F.resize(image, self.out_size, self._image_interpolation)
        return image

    def transform_image_h(self, image):
        if self.out_size_h[0] != image.height or self.out_size_h[1] != image.width:
            interpolation = self.interpolation if self.interpolation is not None else Image.BILINEAR
            return F.resize(image, self.out_size_h, interpolation)
        return image

    def transform_mask(self, mask, name):
        if mask.shape[0] != self.out_size[0] or mask.shape[1] != self.out_size[1]:
            return skimage.transform.resize(mask, self.out_size,
                                            order=0, preserve_range=True,
                                            mode='constant', anti_aliasing=False
                                            ).astype(mask.dtype)
        return mask


class RecordImageSize(VisionTransform):
    def pre_transform(self, sample):
        image = sample['input']
        sample['image_size'] = (image.height, image.width)
        return sample


class _ImageHighResSync(VisionTransform):
    """the high resolution image has to be transformed the same way as low resolution image"""
    def transform_image(self, image):
        raise NotImplementedError

    def transform_image_h(self, image):
        return self.transform_image(image)

    def redo(self, sample):
        return self(sample)


class HorizontalFlip(_ImageHighResSync):
    def transform_image(self, image):
        return F.hflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 0:4:2] = 1 - bbox[:, 2::-2]
        return bbox

    def transform_mask(self, mask, name):
        return np.fliplr(mask)


class VerticalFlip(_ImageHighResSync):
    def transform_image(self, image):
        return F.vflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 1:4:2] = 1 - bbox[:, -2::-2]
        return bbox

    def transform_mask(self, mask, name):
        return np.flipud(mask)


class Transpose(_ImageHighResSync):
    def transform_image(self, image):
        return image.transpose(Image.TRANSPOSE)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox = bbox[:, [1, 0, 3, 2, 4]]
        return bbox

    def transform_mask(self, mask, name):
        return mask.T


class ToRGB(_ImageHighResSync):
    def transform_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


class ToYUV(_ImageHighResSync):
    def transform_image(self, image):
        if image.mode != 'YCbCr':
            image = image.convert('YCbCr')
        return image


class ChannelShuffle(_ImageHighResSync):
    def __init__(self, channels_shuffled):
        self.channels_shuffled = channels_shuffled

    def transform_image(self, image):
        a = np.asarray(image)
        a = a[..., self.channels_shuffled]
        return Image.fromarray(a)
class ToBGR(ChannelShuffle):
    def __init__(self):
        super().__init__([2, 1, 0])


class AutoContrast(VisionTransform):
    def transform_image(self, image):
        try:
            return ImageOps.autocontrast(image)
        except IOError as e:
            warnings.warn(str(e))
            return image


class Equalize(VisionTransform):
    def transform_image(self, image):
        return ImageOps.equalize(image)


def clahe(image):
    with warnings.catch_warnings():
        # ignore skimage/util/dtype.py:135: UserWarning: Possible precision loss when converting from float64 to uint16
        warnings.simplefilter("ignore")
        return Image.fromarray((equalize_adapthist(np.asarray(image)) * 255).astype(np.uint8))


class CLAHE(VisionTransform):
    """Contrast Limited Adaptive Histogram Equalization"""
    def transform_image(self, image):
        return clahe(image)


class ToByteTensor(_ImageHighResSync):
    def transform_image(self, image):
        a = np.asarray(image)
        t = torch.from_numpy(a)
        return t


class ToTensor(_ImageHighResSync):
    def __init__(self, scaling=True, normalize=TORCH_VISION_NORMALIZE):
        """
        convert image to torch.tensor
        Args:
            scaling: if tensors are scaled in the range [0.0, 1.0]
            normalize: `torchvision.transforms.Normalize` or None
        """
        self.scaling = scaling
        self.normalize = normalize

    def transform_image(self, image):
        # 3D image in format CDHW, e.g. CT scans
        image_3d = isinstance(image, np.ndarray) and (image.ndim in {4})

        if image_3d:
            # CDHW --> CD(H*W)
            image_3d_shape = image.shape
            image = image.reshape((image_3d_shape[0], image_3d_shape[1], -1))
            image_t = torch.from_numpy(image)
        elif isinstance(image, (list, tuple)):
            images = [ToTensor.to_tensor(i, self.scaling) for i in image]
            image_t = torch.stack(images)
            image_t = image_t.permute((1, 0, 2, 3))  # DCHW --> CDHW
            # CDHW --> CD(H*W)
            image_3d = True
            image_3d_shape = image_t.shape
            image_t = image_t.view((image_3d_shape[0], image_3d_shape[1], -1))
        else:
            image_t = ToTensor.to_tensor(image, self.scaling)

        if self.normalize:
            image_t = self.normalize(image_t)

        if image_3d:
            image_t = image_t.view(image_3d_shape)

        return image_t

    def transform_mask(self, mask, name):
        # numpy to tensor
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(mask)

    @staticmethod
    def to_tensor(pil_img, scaling):
        if scaling:
            return F.to_tensor(pil_img)
        else:
            np_img = np.float32(pil_img)
            t_img = torch.as_tensor(np_img)
            t_img = t_img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
            # put it from HWC to CHW format
            return t_img.permute((2, 0, 1)).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + f'(scaling={self.scaling}, normalize={self.normalize})'


