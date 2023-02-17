"""
Utilities related to the model
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import torch
import torch.nn as nn


def get_weights(net, state_dict_path=None):
    """ Assign weights to a network architecture

    :param net: torch network architecture
    :param state_dict_path: str, path to state dict
    :return: torch network with weights
    """
    # Assign weights to net
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location='cpu')  # Load state dict
        net.load_state_dict(state_dict)
    else:
        net.apply(init_weights)

    return net


def init_weights(m):
    """
    Function source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
