from pathlib import Path
import itertools
from PIL import Image
from torch.utils.data import IterableDataset


class TransformedStream(IterableDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            yield self.transform(sample)

    def __rshift__(self, other):
        """transformed_dataset = dataset >> transform"""
        if not callable(other):
            raise RuntimeError('Dataset >> callable only!')
        return TransformedStream(dataset=self, transform=other)


class ImageStream(IterableDataset):
    def __init__(self, path, interval=1):
        self.interval = interval
        self.videos = []
        self.images = []
        self.ros_topic = None

        if isinstance(path, int):
            # try to open camera device
            self.videos = [path]
        elif isinstance(path, str):
            if path.isnumeric():
                # camera device
                self.videos = [int(path)]
            elif path.startswith('http'):
                import pafy
                video_pafy = pafy.new(path)
                print(video_pafy.title)
                best = video_pafy.getbest()
                self.videos.append(best.url)
            elif path.startswith('ros:'):
                self.ros_topic = path[4:]
            else:
                path = Path(path)
                if path.is_dir():
                    self.images = itertools.chain(*(path.glob(f"**/*.{suffix}") for suffix in ("JPG", 'jpg', "png")))
                    self.videos = [str(p) for p in path.glob("**/*.mp4")]
                elif path.suffix.lower() in ('.jpg', '.png'):
                    self.images = [path]
                elif path.suffix.lower() in ('.mp4', '.webm'):
                    self.videos = [str(path)]
        assert self.videos or self.images or self.ros_topic

    def __iter__(self):
        for i, image in enumerate(self.images):
            if i % self.interval == 0:
                yield {'image_id': image, 'input': Image.open(image), 'file_name': image}
        for video in self.videos:
            import cv2
            reader = cv2.VideoCapture(video)
            frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 0
            while True:
                _, image = reader.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if i % self.interval == 0:
                    yield {'image_id': f"{video}_{i}", 'input': Image.fromarray(image), 'file_name': video}
                i += 1
                if 0 < frame_count <= i:
                    break

        if self.ros_topic:
            import threading
            from collections import deque
            from sensor_msgs.msg import Image as ImageMsg
            import rospy
            from .ros_numpy_image import image_to_numpy

            image_msg_queue = deque(maxlen=1)
            image_msg_event = threading.Event()
            def imgmsg_callback(imgmsg):
                image_msg_queue.append(imgmsg)
                image_msg_event.set()

            rospy.init_node(self.ros_topic.replace('/', '_') + '_listener')
            rospy.Subscriber(self.ros_topic, ImageMsg, imgmsg_callback)
            while not rospy.is_shutdown():
                image_msg_event.wait()
                imgmsg = image_msg_queue.pop()
                image_msg_event.clear()

                image = image_to_numpy(imgmsg)
                if imgmsg.encoding.startswith('bgr'):
                    if image.shape[-1] == 3:
                        image = image[..., (2, 1, 0)]
                    elif image.shape[-1] == 4:
                        image = image[..., (2, 1, 0, 3)]
                
                yield {'image_id': f"{imgmsg.header.seq}", 'input': Image.fromarray(image), 'file_name': self.ros_topic}

    def __rshift__(self, other):
        """transformed_dataset = dataset >> transform"""
        if not callable(other):
            raise RuntimeError('Dataset >> callable only!')
        return TransformedStream(dataset=self, transform=other)
