import warnings
from ..trainer.data import ImageFolder as ImageFolderBase
from .coco import predition2dict


class ImageFolder(ImageFolderBase):
    def __init__(self, root, sort_by_image_size=False):
        super().__init__(root, sort_by_image_size=sort_by_image_size)
        self.image_size = {}  # cache image size

    def getitem(self, index):
        sample = super().getitem(index)
        self.image_size[sample['image_id']] = sample['input'].size
        return sample

    def load_prediction(self, predictions):
        """return as COCO annotation format
        """
        if len(self.image_size) != len(self):
            warnings.warn("load all images for image_size")
            for i in range(len(self)):
                self.getitem(i)

        images = []
        annotations = []
        for file_name, (width, height) in self.image_size.items():
            det = predictions[str(file_name)]
            image_id = len(images) + 1

            img = {'id': image_id, 'file_name': file_name, 'width': width, 'height': height}
            images.append(img)
            annotations += predition2dict(det, img)

        # assign id to annotations
        for i, anno in enumerate(annotations):
            anno['id'] = i + 1

        ret = dict(images=images, annotations=annotations)

        classnames = predictions.info['model'].get('classnames', [])
        categories = [{'id': i + 1, 'name': name, 'supercategory': ''} for i, name in enumerate(classnames)]
        if categories:
            ret['categories'] = categories

        return ret
