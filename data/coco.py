
import warnings
import itertools
from functools import cmp_to_key
from pathlib import Path
from collections import Iterable, defaultdict
import numpy as np
from PIL import Image
import pandas as pd
from tabulate import tabulate

from .anchor_box_dataset import AnchorBoxDataset
from ..trainer.data import WeightedRandomDataset


COCO_CATEGORIES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

COCO_CLASSNAMES = [COCO_CATEGORIES.get(i, "unknown") for i in range(1, 91)]


def get_label_map():
    """get label map which ignores unknown classes, e.g. total 80 classes
    """
    label_map = {}
    classnames = []
    for coco_id, coco_name in COCO_CATEGORIES.items():
        label_map[coco_id] = len(classnames)
        classnames.append(coco_name)
    return label_map, classnames


def category_count(cocoGt):
    data = [ann['category_id'] for ann in cocoGt.anns.values() if ann['image_id'] in cocoGt.imgs]
    value_counts = pd.Series(data).value_counts()
    return value_counts


def table_of_AP(E, classnames):
    # Compute per-category AP
    precisions = E.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)

    results_per_category = []
    for idx, c in enumerate(E.params.catIds):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        name = classnames[c]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap * 100)))

    # tabulate it
    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (N_COLS // 2),
        numalign="left",
    )
    print("Per-category AP: \n" + table)


def reassign_catId(cocoGt, dt):
    coco_cats = {c['name']: i for i, c in cocoGt.cats.items()}
    pred_cats = {c['id']: c['name'] for c in dt['categories']}
    pred_id_to_coco = {i: coco_cats.get(c, 0) for i, c in pred_cats.items()}
    for r in dt['annotations']:
        r['category_id'] = pred_id_to_coco[r['category_id']]
    dt['annotations'] = [r for r in dt['annotations'] if r['category_id'] > 0]


def coco_eval(cocoGt, anno, catIds=None):
    from pycocotools.cocoeval import COCOeval
    cocoDt = cocoGt.loadRes(anno)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')  # initialize CocoEval object
    if catIds:
        E.params.catIds = catIds
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E


def filter_coco(coco, cats):
    cat_ids = [c for c in coco.cats.values() if c['name'] in cats]
    cat_ids = {i + 1: c for i, c in enumerate(cat_ids)}
    cat_ids_map = {c['id']: t for t, c in cat_ids.items()}
    anns_filtered = [ann for ann in coco.anns.values() if ann['category_id'] in cat_ids_map]

    # reid of annotations
    for i, ann in enumerate(anns_filtered):
        ann['id'] = i + 1
        ann['category_id'] = cat_ids_map[ann['category_id']]

    # reid of categories
    for k, c in cat_ids.items():
        c['id'] = k

    dataset = coco.dataset.copy()
    dataset['annotations'] = anns_filtered
    dataset['categories'] = list(cat_ids.values())

    from pycocotools.coco import COCO
    new_coco = COCO()
    new_coco.dataset = dataset
    new_coco.createIndex()
    return new_coco


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, coco, label_map=None, segmentation=False):
        self.coco = coco
        self.label_map = label_map
        self.segmentation = segmentation

    def __call__(self, target, img_size):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            img_size (int, int): width, height
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        boxes = []
        labels = []

        width, height = img_size
        scale = np.asarray([width, height, width, height], dtype=np.float32)

        if self.segmentation:
            masks = np.zeros((height, width), dtype=np.int32)
        else:
            masks = None

        for obj in target:
            if 'bbox' in obj:
                bbox = np.asarray(obj['bbox'], dtype=np.float32)  # (left_top_x, left_top_y, width, height)
                bbox[2:] += bbox[:2]  # (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
            else:
                warnings.warn("no bbox!")
                continue

            label_idx = obj['category_id']
            if self.label_map:
                label_idx = self.label_map[label_idx]

            if self.segmentation:
                if 'segmentation' in obj:
                    mask_obj = self.coco.annToMask(obj)
                    masks[mask_obj > 0] = label_idx
                else:
                    warnings.warn("no segmentation!")

            bbox /= scale  # [xmin, ymin, xmax, ymax] in percentage
            boxes.append(bbox)
            labels.append(label_idx)

        if boxes:
            boxes=np.asarray(boxes)
            labels=np.asarray(labels)
            bbox = np.hstack((boxes, labels[:, None]))
            bbox = bbox[labels > 0]
            if bbox.size > 0:
                bbox = np.unique(bbox, axis=0)
        else:
            bbox = np.empty((0, 5))

        if self.segmentation:
            return bbox, masks

        return bbox


def predition2dict(det, img):
    image_id, width, height = img['id'], img['width'], img['height']
    scale = np.asarray([1, width, height, width, height])

    results = []
    for cls, bbox in enumerate(det):
        bbox = bbox * scale  # [conf, cx, cy, w, h]
        bbox[:, 1:3] -= (bbox[:, 3:5] / 2)
        for box in bbox:
            # score = round(box[0], 5)
            score = float(box[0])
            results.append(dict(image_id=image_id, category_id=cls + 1, bbox=box[1:5].tolist(), score=score))
    return results


class COCODetection(AnchorBoxDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2014>`_ Dataset.
    Args:
        images_root (string): Root directory where images are downloaded to.
        annotations (string, COCO): annotations file of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """
    metric = 'COCO'

    def __init__(self, images_root, annotations, dataset_name=None, image_ids=None, categories=None,
                 return_image_file=False, sort_by_image_size=False, ignore_empty_images=False,
                 segmentation=False):
        from pycocotools.coco import COCO
        self.return_image_file = return_image_file
        self.root = Path(images_root)
        if isinstance(annotations, COCO):
            self.coco = annotations
        else:
            self.coco = COCO(annotations)

        if categories:
            self.coco = filter_coco(self.coco, categories)

        self.ids = sorted(self.coco.getImgIds())
        if image_ids:
            self.include(image_ids)

        if ignore_empty_images:
            total_images = len(self)
            self.ids = [i for i in self.ids if self._get_target(i)]
            self.ignored_empty_images = total_images - len(self)
        else:
            self.ignored_empty_images = None

        label_map = None
        self.classnames = None
        if dataset_name == 'MS COCO':
            label_map, self.classnames = get_label_map()

        self.target_transform = None
        if self.coco.anns:
            self.target_transform = COCOAnnotationTransform(self.coco, label_map, segmentation)
        if self.classnames is None:
            cats = self.coco.cats
            cats_ids = [c['id'] for c in cats.values()]
            self.classnames = ['background'] * (max(cats_ids) + 1)
            for c in cats.values():
                self.classnames[c['id']] = c['name']

        if sort_by_image_size:
            self.ids = sorted(self.ids, key=cmp_to_key(self._image_size_compare))

    def exclude(self, image_ids):
        self.ids = list(set(self.ids) - set(image_ids))
        self._update_coco_by_ids()

    def include(self, image_ids):
        self.ids = list(set(self.ids) & set(image_ids))
        self._update_coco_by_ids()

    def _update_coco_by_ids(self):
        self.coco.imgs = {k: self.coco.imgs[k] for k in self.ids}
        ids_set = set(self.ids)
        self.coco.anns = {k: v for k, v in self.coco.anns.items() if v['image_id'] in ids_set}
        self.coco.imgToAnns = defaultdict(list, {k: self.coco.imgToAnns[k] for k in self.ids})

    def _image_size_compare(self, x, y):
        x = self._read_image(x)
        y = self._read_image(y)
        if x.height == y.height:
            return x.width - y.width
        return x.height - y.height

    def __subset__(self, indices):
        if isinstance(indices, slice):
            indices = range(len(self))[indices]

        image_ids = [self.ids[i] for i in indices]
        return self.__class__(self.root, self.coco, image_ids=image_ids)

    def __len__(self):
        return len(self.ids)

    def _get_image_path(self, img_id):
        return self.root / self.coco.imgs[img_id]['file_name']

    def _read_image(self, img_id):
        path = self._get_image_path(img_id)
        assert path.exists(), 'Image path does not exist: {}'.format(path)
        image = Image.open(path)
        return image

    def _get_target(self, img_id):
        return self.coco.imgToAnns[img_id]

    def getitem(self, index):
        img_id = self.ids[index]
        sample = dict(image_id=str(img_id))

        img = self._read_image(img_id)
        sample['input'] = img

        target = self._get_target(img_id)
        bbox = self.target_transform(target, img.size)
        sample['bbox'] = bbox

        if self.return_image_file:
            sample['image_file'] = self._get_image_path(img_id)

        return sample

    @property
    def annotations(self):
        for img_id in self.ids:
            img = self.coco.imgs[img_id]
            width, height = img['width'], img['height']
            target = self._get_target(img_id)
            bbox = self.target_transform(target, (width, height))  # bbox in pixels
            yield {'image_id': str(img_id), 'width': width, 'height': height, 'bbox': bbox}

    def __str__(self):
        fmt_str = [self.__class__.__name__,
                   'No. of images: {}'.format(len(self)),
                   'No. of classes: {}'.format(len(self.classnames) - 1),
                   'No. of bboxes {}'.format(len(self.coco.anns)),
                   'Root Location: {}'.format(self.root)]

        if self.ignored_empty_images is not None:
            fmt_str.append('Ignored empty images: {}'.format(self.ignored_empty_images))

        fmt_str = ('\n' + self.repr_indent).join(fmt_str)
        return fmt_str + '\n'

    def get_category_id(self, name):
        for cat in self.coco.cats.values():
            if cat['name'] == name:
                return cat['id']
        return 0

    def load_prediction(self, predictions):
        """return list of dict
            {
            "image_id" : str,
            "category_id" : int,
            "bbox" : [ x, y, width, height ],
            "score" : float
            }
        """
        images = []
        annotations = []
        for image_id in self.ids:
            img = self.coco.imgs[image_id]
            images.append(img)
            det = predictions[str(image_id)]
            annotations += predition2dict(det, img)

        # assign id to annotations
        for i, anno in enumerate(annotations):
            anno['id'] = i + 1

        res = dict(images=images, annotations=annotations)

        classnames = predictions.info.get('model', {}).get('classnames', None)
        if classnames:
            res['categories'] = [{'id': i + 1, 'name': name} for i, name in enumerate(classnames)]
        return res

    def evaluate(self, predictions, num_processes=1, metric=None):
        if metric is None:
            metric = self.metric

        if isinstance(metric, Iterable) and not isinstance(metric, str):
            # mulit metric, return last one as major score
            score = dict(score=0)
            for m in self.metric:
                score = self.evaluate(predictions, num_processes=num_processes, metric=m)
            return score

        if metric.lower() == 'voc':
            print_out = metric[0].isupper()
            from .metric import evaluate_detections
            voc_metric = evaluate_detections(predictions, self, num_processes, print_out=print_out)
            print(f'{metric} metric:', voc_metric['mean'])
            return voc_metric
        elif metric == 'COCO':
            res = self.load_prediction(predictions)
            if res['annotations']:
                # map categories
                if 'categories' in res:
                    reassign_catId(self.coco, res)

                E = coco_eval(self.coco, res['annotations'])
                table_of_AP(E, self.classnames)

                return dict(score=E.stats[0],
                            AP=E.stats[0], AP50=E.stats[1], AP75=E.stats[2],
                            APs=E.stats[3], APm=E.stats[4], APl=E.stats[5],
                            AR=E.stats[8],
                            ARs=E.stats[9], ARm=E.stats[10], ARl=E.stats[11])
            else:
                print('WARN no detection at all!')
            return dict(score=0)
        else:
            raise NotImplementedError(metric)

    def resample(self, num_samples=None):
        """resample dataset by weighting classes"""
        # weight by class count
        class_count = category_count(self.coco)
        class_weights = 1. / class_count
        # limit max weight
        max_weight = 1. / len(class_count)
        class_weights[class_weights > max_weight] = max_weight

        # sample weights = sum of class weight each sample
        weights = np.zeros(len(self))
        for index in range(len(self)):
            img_id = self.ids[index]
            target = self._get_target(img_id)
            for obj in target:
                cat_id = obj['category_id']
                weights[index] += class_weights[cat_id]
        num_samples = num_samples or len(self)
        return WeightedRandomDataset(self, weights, num_samples)

    def find_class(self, classname, n=10, return_index=False):
        cls = self.classnames.index(classname)
        indices = []
        for i in range(len(self)):
            image_id = self.ids[i]
            target = self._get_target(image_id)
            for obj in target:
                label = obj['category_id']
                if label == cls:
                    indices.append(i)
                    break
            if 0 < n <= len(indices):
                break

        if return_index:
            return indices
        return [self[i] for i in indices]