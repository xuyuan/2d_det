
import torch
from torch import nn


class OpTTA(object):
    """Abstract class"""
    def pre_process(self, x):
        """
        Args:
            x: [[N, C, H, W], ...]

        Returns:
            [[N, C, H, W], ...] * K
        """
        raise NotImplementedError

    def post_process(self, y):
        """
        Args:
            y: [C, [n, 5]] * K

        Returns:
            [C, [n, 5]]
        """
        return y

    def pprint_tree(self, file=None, _prefix="", _last=True):
        sub_tree = isinstance(self, Compose)
        node_value = '+' if sub_tree else str(self)
        print(_prefix, "`- " if _last else "|- ", node_value, sep="", file=file)
        _prefix += "   " if _last else "|  "

        if sub_tree:
            child_count = len(self)
            for i, child in enumerate(self):
                _last = i == (child_count - 1)
                child.pprint_tree(file, _prefix, _last)


class Compose(OpTTA, list):
    pass


class SequentialTTA(Compose):
    def pre_process(self, x):
        for tta in self:
            x = tta.pre_process(x)
        return x

    def post_process(self, y):
        for tta in self[::-1]:
            y = tta.post_process(y)
        return y


class ParallelTTA(Compose):
    def __init__(self, ttas, ensemble_func):
        super().__init__(ttas)
        self.merge_func = ensemble_func

    def pre_process(self, x):
        X = [tta.pre_process(x) for tta in self]
        return sum(X, [])

    def post_process(self, y):
        l = len(y) // len(self)
        Y = [y[i*l:i*l+l] for i in range(len(self))]
        Y = [tta.post_process(yi) for yi, tta in zip(Y, self)]

        return self.merge_func(Y)


class OrigTTA(OpTTA):
    def pre_process(self, x):
        return x


class HFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.flip(xi, [3]) for xi in x]


class VFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.flip(xi, [2]) for xi in x]


class DFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.transpose(xi, 2, 3) for xi in x]


class ResizeTTA(OpTTA):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'

    def pre_process(self, x):
        return [torch.nn.functional.interpolate(xi, scale_factor=self.scale_factor, mode='bilinear',  align_corners=False) for xi in x]


class BatchTTA(OpTTA):
    """pack TTA inputs into batch"""
    def pre_process(self, x):
        batch = []
        self._indices = []
        for b in self._batch(x):
            if len(b) > 0:
                l = [len(bi) for bi in b]
                self._indices.append(l)

                b = torch.cat(b, dim=0)
                batch.append(b)
        return batch

    def post_process(self, y):  # [N, C, n*5]
        z = []
        for l, yi in zip(self._indices, y):
            i = 0
            for j in l:
                z.append(yi[i:i+j])
                i += j
        return z

    def _batch(self, x):
        b = x[:1]
        h, w = x[0].shape[-2:]
        for xi in x[1:]:
            if xi.shape[-1] != w or xi.shape[-2] != h:
                yield b
                b = [xi]
                h, w = b[0].shape[-2:]
            else:
                b.append(xi)
        yield b


class TTA(nn.Module):
    def __init__(self, model, data_aug, ensemble_func):
        super().__init__()
        self.model = model

        ttas = []
        brute_mode = 'brute' in data_aug

        if 'orig' in data_aug:
            ttas.append(OrigTTA())

        for aug in data_aug:
            if aug.startswith('x'):
                scale_factor = float(aug[1:])
                ttas.append(ResizeTTA(scale_factor))

        if brute_mode and len(ttas) > 1:
            ttas = [ParallelTTA(ttas, ensemble_func=ensemble_func)]

        if 'dflip' in data_aug:
            ttas.append(DFlipTTA())
        if 'hflip' in data_aug:
            ttas.append(HFlipTTA())
        if 'vflip' in data_aug:
            ttas.append(VFlipTTA())

        if brute_mode and len(ttas) > 1:
            ttas = ttas[:1] + [ParallelTTA([OrigTTA(), tta], ensemble_func=ensemble_func) for tta in ttas[1:]]

        if 'batch' in data_aug:
            ttas.append(BatchTTA())
        self.tta = SequentialTTA(ttas)

        self.tta.pprint_tree()

    @property
    def info(self):
        return {'ModelClass': self.__class__.__name__,
                'model': getattr(self.model, "info", self.model.__class__.__name__),
                'classnames': self.classnames}

    def predict(self, x, **kwargs):
        X = self.tta.pre_process([x])
        Y = [self.model.predict(xi, **kwargs) for xi in X]
        y = self.tta.post_process(Y)
        return y[0]

    @property
    def classnames(self):
        return getattr(self.model, 'classnames', None)
