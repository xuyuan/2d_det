import torch
from PIL import Image
from .detnet import DetNet


class YoloV5(DetNet):
    def __init__(self, model: str):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', model)

    @property
    def classnames(self):
        return self.model.names

    def _predict(self, image, **kwargs):
        single_image_input = isinstance(image, Image.Image)
        image_size = image.size if single_image_input else image[0].size

        with torch.no_grad():
            det = self.model(image, size=max(image_size))

        results = []
        for xywhn in det.xywhn:
            xywhc = xywhn[:, :5]
            cxywh = torch.roll(xywhc, 1, dims=1)
            cls = xywhn[:, 5].int()
            boxes = [cxywh[cls==c] for c in range(len(self.classnames))]
            results.append(boxes)

        if single_image_input:
            results = results[0]
        return results
