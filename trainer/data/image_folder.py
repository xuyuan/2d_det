import copy
from functools import cmp_to_key
from pathlib import Path
from PIL import Image
from .dataset import Dataset


def get_image_files(directory, full_path=False, suffix=None):
    """Returns list of image file names in the given directory."""
    root_path = Path(directory)
    all_files = []
    suffix = suffix or ['png', 'jpg', 'jpeg', 'tif']
    suffix += [i.upper() for i in suffix]
    for ext in suffix:
        files = root_path.glob("**/*." + ext)
        if full_path:
            files = [str(f) for f in files]
        else:
            files = [str(f.relative_to(root_path)) for f in files]
        all_files += files

    all_files.sort()
    return all_files


class ImageFolder(Dataset):
    def __init__(self, root, sort_by_image_size=False, suffix=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.sort_by_image_size = sort_by_image_size
        if self.root.is_dir():
            self.ids = get_image_files(self.root, suffix=suffix)

            if sort_by_image_size:
                self.ids = sorted(self.ids, key=cmp_to_key(self._image_size_compare))
        else:
            # single file
            self.ids = [self.root.name]
            self.root = self.root.parent

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += '(root={0}, sort_by_image_size={1})\n'.format(self.root, self.sort_by_image_size)
        fmt_str += '    len: {}\n'.format(len(self))
        return fmt_str

    def exclusive(self, image_ids):
        s = set(self.ids)
        t = set(image_ids)
        s -= t
        self.ids = list(s)

    def __subset__(self, indices):
        if isinstance(indices, slice):
            indices = range(len(self))[indices]

        subset = copy.copy(self)
        subset.ids = [subset.ids[i] for i in indices]
        return subset

    def getitem(self, index):
        img_file = self.ids[index]
        image_id, img = self._read_image(img_file)

        sample = dict(input=img, image_id=image_id)
        return sample

    def __len__(self):
        return len(self.ids)

    def _image_size_compare(self, x, y):
        _, x = self._read_image(x)
        _, y = self._read_image(y)
        if x.height == y.height:
            return x.width - y.width
        return x.height - y.height

    def _read_image(self, img_file):
        image_id = str(Path(img_file))
        img = Image.open(self.root / img_file)
        return image_id, img


if __name__ == '__main__':
    import sys
    import numpy as np
    from tqdm import tqdm
    from torchvision.transforms.functional import to_tensor
    dataset = ImageFolder(sys.argv[1])
    print(dataset)
    images = []
    for sample in tqdm(dataset):
        #print(sample)
        img = sample['input']
        img_np = to_tensor(img).numpy()
        images.append(img_np.reshape(3, -1))

    images = np.concatenate(images, 1)
    mean = images.mean(1)
    std = images.std(1)
    print(f'mean={mean}, std={std}')


