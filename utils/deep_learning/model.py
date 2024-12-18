"""
Utilities related to the model
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import copy
import torch
import random
from monai.networks.nets import DenseNet
from collections import OrderedDict


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


def prepare_for_transfer_learning(net, method, print_trainable_params=False):
    """
    Prepare torch neural network for transfer learning
    ==> freeze all weights except those in the fully connected layer
    :param net: Torch net
    :param method: str, method of transfer learning. Choose from:
        ['no_freeze', 'freeze_up_to_trans_1', 'freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']
    :param print_trainable_params: bool
    """

    if method in ['freeze_up_to_trans_1', 'freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
        # Gradually freeze layers
        freeze(net.features.conv0.parameters())
        freeze(net.features.norm0.parameters())
        freeze(net.features.relu0.parameters())
        freeze(net.features.pool0.parameters())
        freeze(net.features.denseblock1.parameters())
        freeze(net.features.transition1.parameters())
        if method in ['freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
            freeze(net.features.denseblock2.parameters())
            freeze(net.features.transition2.parameters())
            if method in ['freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
                freeze(net.features.denseblock3.parameters())
                freeze(net.features.transition3.parameters())
                if method in ['freeze_up_to_norm_5']:
                    # Note: This is the same as only unfreezing weights in class_layers.out
                    # Relu, pool and flatten do not contain trainable parameters
                    freeze(net.features.denseblock4.parameters())
                    freeze(net.features.norm5.parameters())

    elif method == 'no_freeze':
        pass
    else:
        raise ValueError('Transfer learning method not recognised')

    # Print number of trainable parameters
    if print_trainable_params:
        print('Number of trainable parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    return net


def freeze(parameters):
    for param in parameters:
        param.requires_grad = False


def loss_to_contribution(loss_list):
    """
    Convert loss list into normalised weights.
    Higher loss = lower contribution.
    """
    contribution_weights = [1 / val for val in loss_list]
    contribution_weights_normalised = [val / (sum(contribution_weights)) for val in contribution_weights]
    return contribution_weights_normalised


def get_weighted_average_model(net_architecture, path_error_dict):
    """
    Get weighted average of multiple models.
    The contribution of a model is based on its loss (higher loss = lower contribution)
    """
    # Convert loss to normalised contribution (so that sum of the contributions is 1)
    path_contribution_dict = dict(zip(path_error_dict.keys(), loss_to_contribution(path_error_dict.values())))

    state_dict_avg = OrderedDict()
    for i, (path, contribution) in enumerate(path_contribution_dict.items()):
        # Load network weights
        net = get_weights(copy_net(net_architecture), path)
        net.to(torch.device('cpu'))
        state_dict = net.state_dict()

        for key in state_dict.keys():
            state_dict_contribution = state_dict[key] * contribution
            if i == 0:
                state_dict_avg[key] = state_dict_contribution
            else:
                state_dict_avg[key] += state_dict_contribution

    weighted_avg_net = DenseNet(3, 1, 1)
    weighted_avg_net.load_state_dict(state_dict_avg)
    return weighted_avg_net


def weighted_avg_local_models(state_dicts_dict, size_dict):
    """ Get weighted average of local models

    :param state_dicts_dict: dict, key: client ip address, value: local state dict
    :param size_dict: dict, key: client ip address, value: dataset size
    :return: weighted average state dict of local state dicts
    """

    n_sum = sum(size_dict.values())
    clients = list(state_dicts_dict.keys())
    state_dict_keys = state_dicts_dict.get(clients[0]).keys()

    state_dict_avg = OrderedDict()
    for i, client in enumerate(clients):
        local_state_dict = state_dicts_dict.get(client)
        n_client = size_dict.get(client)
        for key in state_dict_keys:
            state_dict_contribution = (n_client * local_state_dict[key]) / n_sum
            if i == 0:
                state_dict_avg[key] = state_dict_contribution
            else:
                state_dict_avg[key] += state_dict_contribution

    return state_dict_avg


def get_n_random_pairs_from_dict(input_dict, n, random_seed=None):
    """ Sample n random pairs from a dict without replacement

    :param input_dict: dict
    :param n: int, number of pairs to extract
    :param random_seed: int, random seed
    :return: dict with n pairs from input dict
    """

    if random_seed is not None:
        random.seed(random_seed)
    output_dict = {k: input_dict.get(k) for k in random.sample(list(input_dict.keys()), n)}

    return output_dict


def get_parameters(net, method):
    """ Get total and trainable parameters of a torch neural network
    Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    :param net: torch neural network
    :param method: str, method of transfer learning. Choose from:
        ['no_freeze', 'freeze_up_to_trans_1', 'freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']
    """

    if method in ['freeze_up_to_trans_1', 'freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
        # Gradually freeze layers
        freeze(net.features.conv0.parameters())
        freeze(net.features.norm0.parameters())
        freeze(net.features.relu0.parameters())
        freeze(net.features.pool0.parameters())
        freeze(net.features.denseblock1.parameters())
        freeze(net.features.transition1.parameters())
        if method in ['freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
            freeze(net.features.denseblock2.parameters())
            freeze(net.features.transition2.parameters())
            if method in ['freeze_up_to_trans_3', 'freeze_up_to_norm_5']:
                freeze(net.features.denseblock3.parameters())
                freeze(net.features.transition3.parameters())
                if method in ['freeze_up_to_norm_5']:
                    # Note: This is the same as only unfreezing weights in class_layers.out
                    # Relu, pool and flatten do not contain trainable parameters
                    freeze(net.features.denseblock4.parameters())
                    freeze(net.features.norm5.parameters())

    elif method == 'no_freeze':
        pass
    else:
        raise ValueError('Transfer learning method not recognised')

    total_parameters_dict = {name: p.numel() for name, p in net.named_parameters()}
    trainable_parameters_dict = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}

    return total_parameters_dict, trainable_parameters_dict
