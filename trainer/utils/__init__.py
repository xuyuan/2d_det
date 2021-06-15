import functools
import argparse
import traceback
import warnings
import sys
import os
import torch
from torch import nn


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """use it with `warnings.showwarning = warn_with_traceback`"""
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def choose_device(device):
    if not isinstance(device, str):
        return device

    if device not in ['cuda', 'cpu', 'half']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device in ["cuda", 'half']:
        assert torch.cuda.is_available()
        device = 'cuda'

    device = torch.device(device)
    return device


def get_num_workers(jobs, device='cpu'):
    """
    Parameters
    ----------
    jobs How many jobs to be paralleled. Negative or 0 means number of cpu cores left.

    Returns
    -------
    How many subprocess to be used
    """
    num_workers = jobs

    if isinstance(device, torch.device):
        device = device.type

    if device == 'cpu':
        max_workers = os.cpu_count()
    elif device in ('gpu', 'cuda'):
        max_workers = torch.cuda.device_count()
    else:
        raise RuntimeError(f"unknown device {device}")
    if num_workers <= 0:
        num_workers = max_workers + jobs
    if num_workers < 0 or num_workers > max_workers:
        raise RuntimeError("System doesn't have so many {}: {} vs {}".format(device, jobs, os.cpu_count()))
    return num_workers


def arg2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save(filename, net, args, kwargs):
    if isinstance(net, nn.DataParallel):
        net = net.module

    data = dict(args=args,
                kwargs=kwargs,
                state_dict=net.state_dict())
    torch.save(data, filename)


def load(filename, create, **kwargs):
    print('load {}'.format(filename))
    data = torch.load(filename, map_location='cpu')
    data['kwargs'].update(kwargs)

    net = create(*data['args'], **data['kwargs'])
    net.load_state_dict(data['state_dict'])
    return net


def add_save(create_func):
    @functools.wraps(create_func)
    def extended_create_func(*args, **kwargs):
        net = create_func(*args, **kwargs)
        net.save = functools.partial(save, net=net, args=args, kwargs=kwargs)
        return net
    return extended_create_func
