"""
Utilities related to the model
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import sys
import copy
import json
import torch
import random
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: torch.load with weights_only=True in future


def import_net_architecture(architecture_file_path):
    """
    This function imports the network architecture from the architecture.py file at the given path
    :param architecture_file_path: path to the architecture.py file
    :return: torch network (architecture)
    """
    sys.path.append(os.path.dirname(architecture_file_path))
    from architecture import net_architecture
    return net_architecture


def copy_net(net):
    """ Copy torch network

    Source: https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
    :param net: torch network
    :return: copied network
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


def loss_to_contribution(loss_list):
    """
    Convert loss list into normalised weights. Higher loss = lower contribution.
    :param loss_list: list, each element is a loss value
    :return: list with normalised weight contribution
    """
    contribution_weights = [1 / val for val in loss_list]
    contribution_weights_normalised = [val / (sum(contribution_weights)) for val in contribution_weights]
    return contribution_weights_normalised


def get_weighted_average_model(model_error_dict):
    """
    Get weighted average of multiple models.
    The contribution of a model is based on its loss (higher loss = lower contribution)

    :param model_error_dict: dict, key: model, value: loss (error)
    :param: torch model with weighted average weights
    """
    # Convert loss to normalised contribution (so that sum of the contributions is 1)
    model_contribution_dict = dict(zip(model_error_dict.keys(), loss_to_contribution(model_error_dict.values())))

    state_dict_avg = OrderedDict()
    for i, (model, contribution) in enumerate(model_contribution_dict.items()):
        # Load network weights
        model.to(torch.device('cpu'))
        state_dict = model.state_dict()

        for key in state_dict.keys():
            state_dict_contribution = state_dict[key] * contribution
            if i == 0:
                state_dict_avg[key] = state_dict_contribution
            else:
                state_dict_avg[key] += state_dict_contribution

    # Load average net on model of last iteration
    weighted_avg_net = copy_net(model)
    weighted_avg_net.load_state_dict(state_dict_avg)

    return weighted_avg_net


def weighted_avg_local_models(client_info_dict_sample, fl_round):
    """ Get weighted average of local models

    :param client_info_dict_sample: dict, key: client name, value: client info
    :param fl_round: int, federated learning round
    :return: weighted average state dict of local state dicts
    """

    n_sum = sum([dct['dataset_size'] for dct in client_info_dict_sample.values()])
    clients = list(client_info_dict_sample.keys())
    first_client_model = client_info_dict_sample[clients[0]][f'round_{fl_round}']['model']
    state_dict_keys = first_client_model.state_dict().keys()

    state_dict_avg = OrderedDict()
    for i, client in enumerate(clients):
        local_state_dict = client_info_dict_sample[client][f'round_{fl_round}']['model'].state_dict()
        n_client = client_info_dict_sample[client]['dataset_size']
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


def get_model_param_info(net, print_info=False, save_info_path=None):
    """
    Get model information related to (trainable) parameters

    :param net: torch network
    :param print_info: bool, print the intormation?
    :param save_info_path: str, JSON path to save info dict to
    :return: dict, model information
    """
    # Get model info
    total_parameters_dict = {name: p.numel() for name, p in net.named_parameters()}
    trainable_parameters_dict = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}

    # Collect in one dict
    model_info_dict = {
        'total_parameters': total_parameters_dict,
        'trainable_parameters': trainable_parameters_dict,
        'total_n_parameters': sum(total_parameters_dict.values()),
        'total_n_trainable_parameters': sum(trainable_parameters_dict.values())
    }

    if print_info:
        print(f'Total number of parameters: {sum(total_parameters_dict.values())}\n'
              f'Number of trainable parameters: {sum(trainable_parameters_dict.values())}\n')

    if save_info_path is not None:
        with open(save_info_path, 'w') as file:
            json.dump(model_info_dict, file)

    return model_info_dict
