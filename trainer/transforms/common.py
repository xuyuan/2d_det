
import torchvision.transforms as tvt


def _compose_repr(self, args_string=''):
    format_string = self.__class__.__name__ + '('
    indent = ' ' * len(format_string)
    trans_strings = [repr(t).replace('\n', '\n' + indent) for t in self.transforms]
    if args_string:
        trans_strings.insert(0, args_string)

    format_string += (',\n'+indent).join(trans_strings)
    format_string += ')'
    return format_string


class Compose(tvt.Compose):
    def __repr__(self): return _compose_repr(self)

    def redo(self, sample):
        for t in self.transforms:
            sample = t.redo(sample)
        return sample


class PassThough(object):
    def __call__(self, sample): return sample

    def __repr__(self): return self.__class__.__name__ + '()'
