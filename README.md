# 2D Object Detection Demo & Benchmark

## Demo
```shell
python -m 2d_det.demo
```

```
usage: demo.py [-h] [--device {cuda,cpu}] [-m MODEL] [--threshold THRESHOLD] [-i INPUT] [--resize RESIZE]
               [--max-image-size MAX_IMAGE_SIZE] [--interval INTERVAL]

demo for object detection

optional arguments:
  -h, --help            show this help message and exit
  --device {cuda,cpu}   running with cpu or cuda (default: auto)
  -m MODEL, --model MODEL
                        pre/trained model file (default: torchvision:fasterrcnn_resnet50_fpn)
  --threshold THRESHOLD
                        threshold for accepting detection (default: 0.5)
  -i INPUT, --input INPUT
                        input images, video, folder, youtube url, etc. (default: 0)
  --resize RESIZE       resize the smaller edge of the image if positive number given, or resize to given size if 2 numbers given
                        (default: None)
  --max-image-size MAX_IMAGE_SIZE
                        limit the maximum size of longer edge of image, it is useful to avoid OOM (default: None)
  --interval INTERVAL   detection interval between frames (default: 1)
```

## Benchmark
```shell
python -m 2d_det.inference -h
```

```
usage: inference.py [-h] [--device {cuda,cpu,half}] [-m MODEL] [--threshold THRESHOLD] [--max-bbox MAX_BBOX]
                    [--max-bbox-per-class MAX_BBOX_PER_CLASS] [--nms-thresh NMS_THRESH] [--soft-nms] [--bbox-voting BBOX_VOTING]
                    [--tta TTA] [--num-dataloader-workers NUM_DATALOADER_WORKERS] [--batch-size BATCH_SIZE]
                    [--test-window TEST_WINDOW] [--resize RESIZE] [--max-image-size MAX_IMAGE_SIZE] [--auto-contrast AUTO_CONTRAST]
                    [--clahe CLAHE] [-i INPUT] [-o OUTPUT] [--exclusive EXCLUSIVE] [-j JOBS] [--resume RESUME] [--eval]
                    [--export EXPORT] [--export-format {Image,Image100,ImageSmallBBox,json}] [--profile]
                    [--cudnn-benchmark CUDNN_BENCHMARK] [--log-dir LOG_DIR] [--mlflow-tracking-uri MLFLOW_TRACKING_URI]

test script for object detection

optional arguments:
  -h, --help            show this help message and exit
  --device {cuda,cpu,half}
                        running with cpu, cuda, or float16 (default: auto)
  -m MODEL, --model MODEL
                        pre/trained model file (default: )
  --threshold THRESHOLD
                        overwrite confidence threshold for accepting detection (default: 0)
  --max-bbox MAX_BBOX   maximum number of bbox output per image if positive (default: 0)
  --max-bbox-per-class MAX_BBOX_PER_CLASS
                        max number of detection for each class (default: 4000)
  --nms-thresh NMS_THRESH
                        threshold for NMS (default: 0.5)
  --soft-nms            use soft NMS (default: False)
  --bbox-voting BBOX_VOTING
                        threshold for bbox voting, non positive for disabling (default: 0)
  --tta TTA             Test Time Augementation: orig,hflip,vflip,dflip,autocontrast,equalize (default: )
  --num-dataloader-workers NUM_DATALOADER_WORKERS
                        number of dataloader workers per process (default: None)
  --batch-size BATCH_SIZE
                        batch size (default: 0)
  --test-window TEST_WINDOW
                        size of sliding window for testing (default: None)
  --resize RESIZE       resize the smaller edge of the image if positive number given, or resize to given size if 2 numbers given
                        (default: None)
  --max-image-size MAX_IMAGE_SIZE
                        limit the maximum size of longer edge of image, it is useful to avoid OOM (default: None)
  --auto-contrast AUTO_CONTRAST
                        Apply auto contrast to input image (default: None)
  --clahe CLAHE         Apply Contrast Limited Adaptive Histogram Equalization to input image (default: None)
  -i INPUT, --input INPUT
                        root directory of input images (default: None)
  -o OUTPUT, --output OUTPUT
                        root directory of output (default: None)
  --exclusive EXCLUSIVE
                        txt file contains list of exclusive images (default: None)
  -j JOBS, --jobs JOBS  number of processes (default: 1)
  --resume RESUME       path to prediction for resuming (default: None)
  --eval                evaluation after detection (default: False)
  --export EXPORT       path of export file (default: None)
  --export-format {Image,Image100,ImageSmallBBox,json}
                        the format of input and output (default: json)
  --profile             profile the performance (default: False)
  --cudnn-benchmark CUDNN_BENCHMARK
                        enable cudnn benchmark (default: True)
  --log-dir LOG_DIR     Location to save logs and checkpoints (default: runs/Jun15_02-14-46_dai149)
  --mlflow-tracking-uri MLFLOW_TRACKING_URI
                        URI for MLFlow to which to persist experiment and run data. (default: None)

```