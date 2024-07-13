"""
Script for server of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9

Info:
- Run client scripts first, then server script
"""

import os
import json
import random
import torch
import argparse
import paramiko
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from scp import SCPClient
from collections import OrderedDict
from monai.networks.nets import DenseNet
from DL_utils.model import get_weights

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


def createSSHClient(server, port, user, password):
    """
    Function source: https://stackoverflow.com/questions/250283/how-to-scp-in-python
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def send_file(remote_ip_address, username, password, file_path):
    """ Send file to remote

    :param remote_ip_address: str, remote ip address
    :param username: str, username of remote
    :param password: str, password of remote
    :param file_path: str, path to local file to share
    """
    # Create ssh and scp client
    # Source to fix issue "scp.SCPException: Timeout waiting for scp response":
    # ==> https://github.com/ktbyers/netmiko/issues/1254
    ssh = createSSHClient(remote_ip_address, 22, username, password)
    scp = SCPClient(ssh.get_transport(), socket_timeout=60)

    # Share model with client
    scp.put(file_path, remote_path=file_path)

    # Share txt file with client that marks the end of the file transfer
    txt_file_path = file_path.replace(os.path.splitext(file_path)[1], '_transfer_completed.txt')
    with open(txt_file_path, 'w') as file:
        file.write(f'The following file was succesfully transferred: {txt_file_path}')
    scp.put(txt_file_path, txt_file_path)


def wait_for_file(file_path):
    """ This function waits for a file path to exist

    :param file_path: str, path to file
    """
    while not os.path.exists(file_path):
        pass


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


def freeze(parameters):
    for param in parameters:
        param.requires_grad = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Server',
        description='Server for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    parser.add_argument('--FL_plan_path', type=str, help='Path to the FL plan JSON')
    args = parser.parse_args()
    settings_path = args.settings_path
    FL_plan_path = args.FL_plan_path

    print('\n\n==============================\nStarting federated learning :)\n==============================\n\n')

    # FL start time
    fl_start_time = dt.datetime.now()

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path = settings_dict.get('workspace_path')                    # Path to workspace for server and clients
    initial_state_dict_path = settings_dict.get('initial_state_dict_path')  # Path to initial state dict
    client_credentials_dict = settings_dict.get('client_credentials')       # Client credentials dict

    # Create workspace folder
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    # Copy FL plan in workspace folder
    os.system(f'cp {FL_plan_path} {os.path.join(workspace_path, "FL_plan.json")}')

    # Send FL plan to all clients
    print(f'==> Sending FL plan to all clients...')
    for client_name, credentials in client_credentials_dict.items():
        print(f'    ==> Sending to {client_name} ...')
        client_ip_address = credentials.get('ip_address')
        send_file(client_ip_address, credentials.get('username'), credentials.get('password'),
                  os.path.join(workspace_path, 'FL_plan.json'))
        print(f'    ==> Sent to {client_name} ...')

    # Extract FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                    # Number of FL rounds
    n_clients_set = FL_plan_dict.get('n_clients_set')               # Number of clients in set for averaging
    patience_stop = int(FL_plan_dict.get('pat_stop'))               # N fl rounds stagnating val loss before stopping
    tl_method = FL_plan_dict.get('tl_method')                       # Get transfer learning method
    print('\n========\nFL plan:\n========\n')
    for k, v in FL_plan_dict.items():
        print(f'- {k}: {v}')
    print('\n')

    # Wait for all clients to share their dataset size
    print('==> Collecting all client dataset sizes...')
    client_dataset_size_dict = {}
    for client_name in client_credentials_dict.keys():
        client_dataset_txt_path = os.path.join(workspace_path, f'{client_name}_dataset_size.txt')
        wait_for_file(client_dataset_txt_path.replace('.txt', '_transfer_completed.txt'))
        with open(client_dataset_txt_path, 'r') as file:
            n_client = int(file.read())
            client_dataset_size_dict.update({client_name: n_client})
            print(f'     ==> {client_name}: n = {n_client}')

    # Load initial network and save
    net_architecture = DenseNet(3, 1, 1)
    if initial_state_dict_path is None:
        print('\nWarning: Do not have initial model !!!\n==> Default model has been used\n\n')
        global_net = net_architecture
        model_path = os.path.join(workspace_path, 'initial_model.pt')
        global_net = get_weights(net_architecture, model_path)
        torch.save(global_net.state_dict(), model_path)

    else:
        print('\n==> Initial model has been used\n\n')    
        global_net = get_weights(net_architecture, initial_state_dict_path)
        model_path = os.path.join(workspace_path, 'initial_model.pt')
        torch.save(global_net.state_dict(), model_path)
    

    # Print model information: total and trainable parameters
    total_parameters_dict, trainable_parameters_dict = get_parameters(global_net, method=tl_method)
    parameters_info_txt = f'Total number of parameters: {sum(total_parameters_dict.values())}\n' \
                          f'Number of trainable parameters: {sum(trainable_parameters_dict.values())}\n' \
                          f'More info trainable parameters: {trainable_parameters_dict}'
    print(parameters_info_txt)                                                          # Print
    with open(os.path.join(workspace_path, 'parameters_info.txt'), 'w') as txt_file:    # Save
        txt_file.write(parameters_info_txt)

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf       # Reference validation loss
    counter_stop = 0            # Counter for FL stop
    best_model_path = None      # Best model path with lowest avg validation loss
    avg_val_loss_clients = []   # Average validation loss across clients

    # Start federated learning
    for fl_round in range(1, n_rounds + 1):  # Start counting from 1
        print(f'\n*****************\nRound {fl_round}\n*****************\n')
        round_start_time = dt.datetime.now()

        # Send global model to all clients
        for client_name, credentials in client_credentials_dict.items():
            print(f'==> Sending global model to {client_name}...')
            client_ip_address = credentials.get('ip_address')
            send_file(client_ip_address, credentials.get('username'), credentials.get('password'), model_path)
        print('==> Model shared with all clients. Waiting for updated client models...')
        txt_file_paths = [os.path.join(workspace_path, f'model_{client_name}_round_{fl_round}_transfer_completed.txt')
                          for client_name in client_credentials_dict.keys()]
        for txt_file_path in txt_file_paths:
            wait_for_file(txt_file_path)

        # Create new global model by combining local models
        print('==> Combining local model weights and saving...')
        local_model_paths_dict = {client_name: os.path.join(workspace_path, f'model_{client_name}_round_{fl_round}.pt')
                                  for client_name in client_credentials_dict.keys()}
        local_state_dicts_dict = {k: torch.load(v, map_location='cpu') for k, v in local_model_paths_dict.items()}
        if n_clients_set is not None:
            local_state_dicts_dict = get_n_random_pairs_from_dict(local_state_dicts_dict, n_clients_set, fl_round)
            print(f'    ==> Clients in sample (random seed = {fl_round}): {list(local_state_dicts_dict.keys())}')
        new_global_state_dict = weighted_avg_local_models(local_state_dicts_dict,
                                                          {k: client_dataset_size_dict.get(k)
                                                           for k in local_state_dicts_dict.keys()})
        model_path = os.path.join(workspace_path, f'global_model_round_{fl_round}.pt')  # Overwrite model_path
        torch.save(new_global_state_dict, model_path)

        # Calculate average validation loss
        val_loss_avg = 0
        print('==> Average validation loss tracking...')
        for client_name in client_credentials_dict.keys():
            wait_for_file(os.path.join(
                workspace_path, f'train_results_{client_name}_round_{fl_round}_transfer_completed.txt'
            ))
            filename = f'train_results_{client_name}_round_{fl_round}.csv'
            train_results_client_df = pd.read_csv(os.path.join(workspace_path, filename))
            val_loss_avg += train_results_client_df['val_loss'].mean() / len(client_credentials_dict)
        print(f'     ==> val loss ref: {val_loss_ref} || val loss avg: {val_loss_avg}')
        avg_val_loss_clients.append(val_loss_avg)

        # Perform actions based on average validation loss
        if val_loss_avg < val_loss_ref:         # Improvement
            val_loss_ref = val_loss_avg
            counter_stop = 0
            best_model_path = model_path
        else:                                   # No improvement
            counter_stop += 1
            if counter_stop == patience_stop:
                stop_txt_file_path = os.path.join(workspace_path, 'stop_training.txt')
                with open(stop_txt_file_path, 'w') as txt_file:
                    txt_file.write('This file causes early FL stopping')
                for client_name, credentials in client_credentials_dict.items():
                    print(f'==> Sending stop txt file to {client_name}...')
                    client_ip_address = credentials.get('ip_address')
                    send_file(client_ip_address, credentials.get('username'), credentials.get('password'),
                              stop_txt_file_path)
                break
        print(f'     ==> lr stop counter: {counter_stop}')

        # Time tracking
        round_stop_time = dt.datetime.now()
        round_duration = round_stop_time - round_start_time
        ETA = (round_stop_time + round_duration * (n_rounds - fl_round - 1)).strftime('%Y/%m/%d, %H:%M:%S')
        print(f'Round time: {round_duration / 60} min || ETA: {ETA}')

    # Create dataframe with average validation loss across clients
    avg_val_loss_df = pd.DataFrame({'avg_val_loss_clients': avg_val_loss_clients,
                                    'fl_round': range(1, fl_round+1)})
    avg_val_loss_df.to_csv(os.path.join(workspace_path, 'avg_val_loss_clients.csv'), index = False)

    # Copy final model path and send to clients
    final_model_path = os.path.join(workspace_path, "final_model.pt")
    os.system(f'cp {best_model_path} {final_model_path}')
    print(f'==> Sending final model ({os.path.basename(best_model_path)}) to all clients...')
    with open(os.path.join(workspace_path, 'final_model.txt'), 'w') as txt_file:
        txt_file.write(best_model_path)
    for client_name, credentials in client_credentials_dict.items():
        print(f'     ==> Sending to {client_name}')
        client_ip_address = credentials.get('ip_address')
        send_file(client_ip_address, credentials.get('username'), credentials.get('password'), final_model_path)

    # Calculate overall test MAE
    print('==> Calculate overall test MAE...')
    test_mae_overall = 0
    for client_name, n_client in client_dataset_size_dict.items():
        print(f'    ==> Wait for test results {client_name}...')
        test_results_txt_path = os.path.join(workspace_path, f'test_results_{client_name}_transfer_completed.txt')
        wait_for_file(test_results_txt_path)
        test_df_client = pd.read_csv(test_results_txt_path.replace('_transfer_completed.txt', '.csv'))
        test_mae_client = test_df_client['test_mae'].iloc[0]
        test_mae_overall += test_mae_client * n_client / sum(client_dataset_size_dict.values())
    with open(os.path.join(workspace_path, 'overall_test_mae.txt'), 'w') as txt_file:
        txt_file.write(f'Overall test MAE: {test_mae_overall}')

    # Print total FL duration
    fl_stop_time = dt.datetime.now()
    fl_duration = fl_stop_time - fl_start_time
    print(f'Total federated learning duration: {fl_duration/3600} hrs')
