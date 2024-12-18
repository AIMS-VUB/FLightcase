"""
Utilities related to the model
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import copy
import torch


def copy_net(net):
    """ Copy torch network

    Source: https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
    """
    net_copy = copy.deepcopy(net)
    return net_copy


def get_weights(net, state_dict_path):
    """ Assign weights to a network architecture

    :param net: torch network architecture
    :param state_dict_path: str, path to state dict
    :return: torch network with weights
    """

    copied_net = copy_net(net)

    # Assign weights to net
    state_dict = torch.load(state_dict_path, map_location='cpu')  # Load state dict
    copied_net.load_state_dict(state_dict)

    return copied_net
