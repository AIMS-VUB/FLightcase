"""
General DL functions
"""

import torch
import warnings


def get_device(device_str):
    """
    This function gets the desired device
    """
    if device_str == 'cpu':
        device = torch.device(device_str)
    elif device_str.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(device_str)
            print(f'{device_str} available!')
        else:
            warnings.warn('cuda is not available, continuing with cpu instead')
            device = torch.device('cpu')
    else:
        raise ValueError(f'Device request {device_str} not understood. Choose from: "cpu", "cuda", "cuda:N", '
                         f'with N an integer value indicating which GPU to use.')
    return device
