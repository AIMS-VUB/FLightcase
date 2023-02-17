"""
Script for server of the federated learning network
"""

import os
import json
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


def wait_for_clients(local_model_paths):
    """ Wait until all clients have shared their local model with the server

    :param local_model_paths: list, expected location (path) of each local model on the server
    """
    while not sum([os.path.exists(path) for path in local_model_paths]) == len(local_model_paths):
        pass


def FedAvg(global_state_dict, local_state_dicts):
    # Sum global and local state dicts
    state_dict_sum = global_state_dict.copy()
    for local_state_dict in local_state_dicts:
        for key, value in local_state_dict.items():
            state_dict_sum[key] = state_dict_sum[key] + local_state_dict[key]

    # Divide each key by the number of local state dicts + 1 (the global model)
    state_dict_avg = OrderedDict()
    for key, value in state_dict_sum.items():
        state_dict_avg[key] = state_dict_sum[key] / (len(local_state_dicts) + 1)

    return state_dict_avg


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
    print(FL_plan_path)
    print(os.path.join(workspace_path, "FL_plan.json"))
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
    n_rounds = int(FL_plan_dict.get('n_rounds'))  # Number of FL rounds

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
        wait_for_clients(txt_file_paths)

        # Create new global model by combining local models
        print('==> Combining local model weights and saving...')
        local_model_paths = [os.path.join(workspace_path, f'model_{ip_address}_round_{fl_round}.pt')
                             for ip_address in client_credentials_dict.keys()]
        local_state_dicts = [torch.load(path, map_location='cpu') for path in local_model_paths]
        new_global_state_dict = FedAvg(global_net.state_dict(), local_state_dicts)
        model_path = os.path.join(workspace_path, f'global_model_round_{fl_round}.pt')  # Overwrite model_path
        torch.save(global_net.state_dict(), model_path)
