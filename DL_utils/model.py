"""
Utilities related to the model
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import torch


def get_weights(net, state_dict_path):
    """ Assign weights to a network architecture

    :param net: torch network architecture
    :param state_dict_path: str, path to state dict
    :return: torch network with weights
    """
    # Assign weights to net
    state_dict = torch.load(state_dict_path, map_location='cpu')  # Load state dict
    net.load_state_dict(state_dict)

    return net
