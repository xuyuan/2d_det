"""
test script for object detection
"""

import argparse
from pathlib import Path
import functools
import cProfile
import torch
from torch import nn
import time
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .trainer.test import Tester
from .trainer.utils import get_num_workers, arg2bool
from .trainer.utils.logger import Logger, default_log_dir
from .trainer.transforms.vision import ToTensor, Resize, AutoContrast, CLAHE
from .nn import load as load_model
from .utils.box_utils import nms, point_form
from .nn.tta import TTA


def add_test_argument(parser):
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu', 'half'], help='running with cpu, cuda, or float16')
    parser.add_argument("-m", "--model", type=str, default='', help='pre/trained model file')

    parser.add_argument("--threshold", type=float, default=0,
                        help='overwrite confidence threshold for accepting detection')
    parser.add_argument("--max-bbox", type=int, default=0, help='maximum number of bbox output per image if positive')
    parser.add_argument('--max-bbox-per-class', type=int, default=4000, help='max number of detection for each class')
    parser.add_argument("--nms-thresh", type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--soft-nms', action='store_true', help='use soft NMS')
    parser.add_argument('--bbox-voting', type=float, default=0,
                        help='threshold for bbox voting, non positive for disabling')
    parser.add_argument('--tta', type=str, default='', help='Test Time Augementation: orig,hflip,vflip,dflip,autocontrast,equalize')

    parser.add_argument("--num-dataloader-workers", type=int, help='number of dataloader workers per process')
    parser.add_argument("--batch-size", type=int, default=0, help='batch size')
    parser.add_argument('--test-window', type=lambda x: [int(v) for v in x.split(',')],
                        help='size of sliding window for testing')
    parser.add_argument("--resize", type=lambda s: [int(item) for item in reversed(s.split('x'))] if 'x' in s else int(s),
                        help='resize the smaller edge of the image if positive number given, or resize to given size if 2 numbers given')
    parser.add_argument("--max-image-size", type=int, help='limit the maximum size of longer edge of image, it is useful to avoid OOM')
    parser.add_argument("--auto-contrast", type=arg2bool, help='Apply auto contrast to input image')
    parser.add_argument("--clahe", type=arg2bool, help='Apply Contrast Limited Adaptive Histogram Equalization to input image')


class ArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_test_argument(self.parser)
        self.parser.add_argument("-i", "--input", type=str, help='root directory of input images')
        self.parser.add_argument("-o", "--output", type=str, help='root directory of output')
        self.parser.add_argument("--exclusive", type=str, help='txt file contains list of exclusive images')
        self.parser.add_argument('-j', '--jobs', type=int, help='number of processes', default=1)
        self.parser.add_argument('--resume', type=str, help='path to prediction for resuming')
        self.parser.add_argument('--eval', action='store_true', help='evaluation after detection')
        self.parser.add_argument('--export', type=str, help='path of export file')
        self.parser.add_argument('--export-format', type=str, choices=('Image', 'Image100', 'ImageSmallBBox', 'json'),
                                default='json', help='the format of input and output')
        self.parser.add_argument('--profile', action='store_true', help='profile the performance')
        self.parser.add_argument('--cudnn-benchmark', default=True, type=arg2bool, help='enable cudnn benchmark')
        self.parser.add_argument('--log-dir', type=str, default=default_log_dir(),
                                 help='Location to save logs and checkpoints')
        self.parser.add_argument('--mlflow-tracking-uri', type=str,
                                 help='URI for MLFlow to which to persist experiment and run data.')

    def add_argument(self, *kargs, **kwargs):
        return self.parser.add_argument(*kargs, **kwargs)

    def add_argument_group(self, *kargs, **kwargs):
        return self.parser.add_argument_group(*kargs, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, args=None, namespace=None, raise_warning=True):
        args = self.parser.parse_args(args=args, namespace=namespace)

        if raise_warning:
            if not args.output and not args.eval and not args.export:
                raise UserWarning("Please specify at least one path for output / evaluation / export")

        return args


class PredictModel(nn.Module):
    """wrapper to call detect function with args"""
    def __init__(self, model, detect_args):
        super().__init__()
        self.model = model

        test_window = detect_args.get('test_window', 0)
        if test_window:
            if len(test_window) == 1:
                test_window = test_window * 2
            assert len(test_window) == 2
            self.model.TEST_WINDOW = np.asarray(test_window)

        if detect_args.get('tta', None):
            if 'detectron2' in detect_args['tta']:
                # assume model is Detectron2Det
                min_size = min(detect_args['image_size']) / 8
                min_sizes = [min_size * 7, min_size * 8, min_size * 9]
                min_sizes = [int(i) for i in min_sizes]
                self.model.enable_tta(min_sizes)
            else:
                self.model = TTA(self.model, detect_args['tta'], nms_thresh=detect_args['nms_thresh'])

        self.detect_args = detect_args

    @property
    def info(self):
        return self.model.info

    @property
    def classnames(self):
        return self.model.classnames

    def forward(self, sample):
        detections = self.model.predict(sample, nms_thresh=self.detect_args['nms_thresh'])
        # return self.model.predict(sample, **self.detect_args)

        if isinstance(sample, Image.Image):
            return self.post_process(detections)
        else:
            # batch mode
            return [self.post_process(d) for d in detections]

    def pack_detections(self, detections: [np.ndarray]) -> (torch.Tensor, torch.Tensor):
        all_labels = [[i] * len(bbox) for i, bbox in enumerate(detections)]
        all_labels = torch.tensor(sum(all_labels, []))  # flat
        all_detections = torch.from_numpy(np.vstack(detections))
        return all_detections, all_labels

    def unpack_detections(self, all_detections: torch.Tensor, all_labels: torch.Tensor, num_classes: int) -> [np.ndarray]:
        detections = []
        for i in range(num_classes):
            indices = all_labels == i
            det = all_detections[indices]
            detections.append(det.numpy())
        return detections

    def post_process(self, detections: [np.ndarray]) -> [np.ndarray]:
        max_bbox = self.detect_args.get('max_bbox', 0)
        if max_bbox > 0:
            all_detections, all_labels = self.pack_detections(detections)
            if len(all_labels) > max_bbox:
                all_conf = all_detections[:, 0]
                top_conf, top_indices = all_conf.topk(max_bbox)
                all_detections = all_detections[top_indices]
                all_labels = all_labels[top_indices]

                detections = self.unpack_detections(all_detections, all_labels, num_classes=len(detections))
        return detections


def create_model(model_file, conf_thresh=0, **detect_args):
    net = load_model(model_file, score_thresh_test=conf_thresh)
    return PredictModel(net, detect_args)


def nms_merge(predicitons):
    v = [torch.from_numpy(i) for i in predicitons]
    d = torch.cat(v, dim=0)
    # xywh --> ltrd
    boxes = point_form(d[:, 1:5])
    scores = d[:, 0]
    keep, scores = nms(boxes, scores)  # TODO: soft nms, bbox vote...

    d = d[keep]
    return [d.cpu().numpy()]


def inference(dataset, args):

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    detect_args = dict(test_window=args.test_window,
                       top_k=args.max_bbox_per_class,
                       max_bbox=args.max_bbox,
                       conf_thresh=args.threshold,
                       nms_thresh=args.nms_thresh,
                       soft_nms=args.soft_nms,
                       bbox_voting=args.bbox_voting,
                       tta=args.tta.split(',') if args.tta else None)

    cm = functools.partial(create_model, **detect_args)

    tester = Tester(create_model=cm, device=args.device, jobs=args.jobs, disable_tqdm=False, cudnn_benchmark=args.cudnn_benchmark,
                    num_dataloader_workers=args.num_dataloader_workers)

    # dataset = dataset[:3]
    if args.auto_contrast:
        dataset = dataset >> AutoContrast()

    if args.clahe:
        dataset = dataset >> CLAHE()

    if args.resize:
        dataset = dataset >> Resize(args.resize, interpolation=Image.BILINEAR, max_size=args.max_image_size)

    if args.batch_size > 0:
        dataset = dataset >> ToTensor(scaling=False, normalize=None)

    print(dataset, flush=True)

    if args.eval:
        logger = Logger(args.log_dir, mlflow_tracking_uri=args.mlflow_tracking_uri)

    start_time = time.time()
    predictions = tester.test(args.model, dataset, args.output, batch_size=args.batch_size, resume=args.resume)

    if args.eval:
        inference_time = time.time() - start_time
        num_processes = get_num_workers(0)
        metrics = dataset.evaluate(predictions, num_processes=num_processes)
        metrics['inference_time'] = inference_time
        metrics['FPS'] = len(dataset) / inference_time
        nonhparam_args = ('profile', 'log_dir', 'mlflow_tracking_uri')
        hparam_dict = {k: v for k, v in vars(args).items() if k not in nonhparam_args and v is not None}
        logger.add_hparams(hparam_dict, metrics)
        logger.close()

    if args.export:
        from .export import export
        export_path = Path(args.export)
        export(predictions, export_path, dataset, args.export_format, args.threshold)

    if args.profile:
        pr.disable()
        pr.print_stats()
        pr.dump_stats('test.profile')
    print(f'inference done in {time.time() - start_time} seconds')
    return predictions


if __name__ == '__main__':
    from .data import create_test_dataset
    parser = ArgumentParser()
    args = parser.parse_args()

    dataset = create_test_dataset(args.input)
    inference(dataset, args)

