
import torch
from torch import nn

from ..utils.box_utils import point_form
from torchvision.ops import nms


class DetNet(nn.Module):
    TEST_WINDOW = None
    TEST_WINDOW_IGNORE_BORDER = 0.25

    @property
    def info(self):
        return {'ModelClass': self.__class__.__name__,
                'classnames': self.classnames}

    def _predict(self, image, **kwargs):
        """return: [[array(N, 5), ...]] in B, C, N, (conf, cx, cy, w, h)"""
        raise NotImplemented

    def slide(self, size, window_size, stride):
        if window_size > size:
            yield 0, size
        else:
            for i in range(0, size - window_size + stride, stride):
                ii = i + window_size
                if ii > size:
                    ii = size
                    i = size - window_size
                yield i, ii

    def sliding_window(self, pil_image, size, step):
        for upper, lower in self.slide(pil_image.height, size[0], step[0]):
            for left, right in self.slide(pil_image.width, size[1], step[1]):
                image = pil_image.crop((left, upper, right, lower))
                yield left, upper, image

    def predict(self, image, slide_window=None, nms_thresh=0.3, **kwargs):
        """predict using sliding windows"""
        if slide_window is None:
            slide_window = self.TEST_WINDOW

        detections = []
        if slide_window is None:
            detections = self._predict(image, nms_thresh=nms_thresh, **kwargs)
        else:
            # step_size = slide_window * 3 // 4  # 1/4 overlap
            step_size = slide_window // 2 # 1/2 overlap

            for left, top, img in self.sliding_window(image, slide_window, step_size):
                det = self._predict(img, image_size=slide_window, nms_thresh=nms_thresh, **kwargs)

                valid_right = 1 if left + img.width >= image.width else 1 - self.TEST_WINDOW_IGNORE_BORDER
                valid_bottom = 1 if top + img.height >= image.height else 1 - self.TEST_WINDOW_IGNORE_BORDER
                valid_left = 0 if left <= 0 else self.TEST_WINDOW_IGNORE_BORDER
                valid_top = 0 if top <= 0 else self.TEST_WINDOW_IGNORE_BORDER

                scale_height = img.height / image.height
                scale_width = img.width / image.width
                left /= image.width
                top /= image.height
                scale = det[0].new([1, scale_width, scale_height, scale_width, scale_height])
                offset = scale.new([0, left, top, 0, 0])

                for c, d in enumerate(det):
                    # d = offset + d * scale
                    valid = ((d[:, 1] < valid_right) & (d[:, 1] > valid_left) &
                             (d[:, 2] < valid_bottom) & (d[:, 2] > valid_top))
                    d = d[valid]
                    if len(d) == 0:
                        continue

                    d = torch.addcmul(offset, d, scale)
                    if len(detections) > c:
                        detections[c].append(d)
                    else:
                        detections.append([d])
            detections = [torch.cat(d, dim=0) for d in detections]
            # NMS
            detections = [d[nms(point_form(d[:, 1:5]), d[:, 0], nms_thresh)] for d in detections]

        if torch.is_tensor(image) and image.dim() == 4:
            detections = [[d.cpu().numpy() for d in det] for det in detections]
        else:
            detections = [d.cpu().numpy() for d in detections]
        return detections


class NoneNet(DetNet):
    def _predict(self, image, **kwargs):
        return []
