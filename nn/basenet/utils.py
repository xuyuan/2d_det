
from itertools import repeat
import warnings
import torch
import torch.utils.model_zoo as model_zoo
from torch._six import container_abcs


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


tup_single = _ntuple(1)
tup_pair = _ntuple(2)
tup_triple = _ntuple(3)
tup_quadruple = _ntuple(4)


def load_part_state_dict(model: torch.nn.Module, pretrained_dict: dict, key_prefix: str = ''):
    """
    :param model: `torch.nn.Module`
    :param pretrained_dict: dict
    :param key_prefix: 'key/to/prefix'
    :return:
    """
    if key_prefix:
        if '/' in key_prefix:
            # get sub state_dict
            key_prefix = key_prefix.split('/')
            for k in key_prefix[:-1]:
                pretrained_dict = pretrained_dict[k]
            key_prefix = key_prefix[-1]
        pretrained_dict = {k.replace(key_prefix, '', 1): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    missing_keys = []
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            missing_keys.append(k)
        else:
            if pretrained_dict[k].size() == v.size():
                # 2. overwrite entries in the existing state dict
                model_dict[k] = pretrained_dict[k]
            else:
                warnings.warn(f"'{k}' has shape {pretrained_dict[k].size()} in the checkpoint but {model_dict[k].size()} in the model! Skipped.")
    if missing_keys:
        warnings.warn(f'Missing key(s) in state_dict: {missing_keys}. ')
    ignored_keys = [k for k in pretrained_dict if k not in model_dict]
    if ignored_keys:
        warnings.warn(f'Ignore key(s) in state_dict: {ignored_keys}. ')

    # 3. load the new state dict
    model.load_state_dict(model_dict)


def load_pretrained(model: torch.nn.Module, pretrained: str, key_prefix: str = ''):
    if pretrained:
        state_dict = None
        if pretrained.startswith('https://') or pretrained.startswith('http://'):
            state_dict = model_zoo.load_url(pretrained, map_location=lambda storage, loc: storage)
        elif pretrained.endswith('.pth'):
            print('load weights from file:', pretrained)
            state_dict = torch.load(pretrained,map_location=lambda storage, loc: storage)
        if state_dict:
            load_part_state_dict(model, state_dict, key_prefix=key_prefix)
