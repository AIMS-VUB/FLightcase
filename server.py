"""
Script for server of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9
"""

import os
import json
import random
import torch
import argparse
import paramiko
from scp import SCPClient
from collections import OrderedDict
from monai.networks.nets import DenseNet
from DL_utils.model import get_weights


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
    ssh = createSSHClient(remote_ip_address, 22, username, password)
    scp = SCPClient(ssh.get_transport())

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
    output_dict = {k: input_dict.get(k) for k in random.sample(input_dict.keys(), n)}

    return output_dict


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
    for ip_address, credentials in client_credentials_dict.items():
        print(f'    ==> Sending to {ip_address} ...')
        send_file(ip_address, credentials.get('username'), credentials.get('password'),
                  os.path.join(workspace_path, 'FL_plan.json'))

    # Extract FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                            # Number of FL rounds
    n_clients_set = FL_plan_dict.get('n_clients_set')                       # Number of clients in set for averaging

    # Wait for all clients to share their dataset size
    print('==> Collecting all client dataset sizes...')
    client_dataset_size_dict = {}
    for client_ip_address in client_credentials_dict.keys():
        client_dataset_txt_path = os.path.join(workspace_path, f'{client_ip_address}_dataset_size.txt')
        wait_for_file(client_dataset_txt_path.replace('.txt', '_transfer_completed.txt'))
        with open(client_dataset_txt_path, 'r') as file:
            n_client = int(file.read())
            client_dataset_size_dict.update({client_ip_address: n_client})
            print(f'     ==> {client_ip_address}: n = {n_client}')

    # Load initial network and save
    net_architecture = DenseNet(3, 1, 1)
    global_net = get_weights(net_architecture, initial_state_dict_path)
    model_path = os.path.join(workspace_path, 'initial_model.pt')
    torch.save(global_net.state_dict(), model_path)

    # Start federated learning
    for fl_round in range(n_rounds):
        print(f'\n*****************\nRound {fl_round}\n*****************\n')
        # Send global model to all clients
        for ip_address, credentials in client_credentials_dict.items():
            print(f'==> Sending global model to {ip_address}...')
            send_file(ip_address, credentials.get('username'), credentials.get('password'), model_path)
        print('==> Model shared with all clients. Waiting for updated client models...')
        txt_file_paths = [os.path.join(workspace_path, f'model_{ip_address}_round_{fl_round}_transfer_completed.txt')
                          for ip_address in client_credentials_dict.keys()]
        for txt_file_path in txt_file_paths:
            wait_for_file(txt_file_path)

        # Create new global model by combining local models
        print('==> Combining local model weights and saving...')
        local_model_paths_dict = {ip_address: os.path.join(workspace_path, f'model_{ip_address}_round_{fl_round}.pt')
                                  for ip_address in client_credentials_dict.keys()}
        local_state_dicts_dict = {k: torch.load(v, map_location='cpu') for k, v in local_model_paths_dict.items()}
        if n_clients_set is not None:
            seed = 42
            local_state_dicts_dict = get_n_random_pairs_from_dict(local_state_dicts_dict, n_clients_set, seed)
            client_dataset_size_dict = get_n_random_pairs_from_dict(client_dataset_size_dict, n_clients_set, seed)
            print(f'    ==> Clients in sample: {list(local_state_dicts_dict.keys())}')
        new_global_state_dict = weighted_avg_local_models(local_state_dicts_dict, client_dataset_size_dict)
        model_path = os.path.join(workspace_path, f'global_model_round_{fl_round}.pt')  # Overwrite model_path
        torch.save(new_global_state_dict, model_path)
