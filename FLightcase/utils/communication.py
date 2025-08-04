"""
Functions related to communication between server and client
"""

import os
import re
import sys
import pathlib
import paramiko
import pandas as pd
import datetime as dt
from scp import SCPClient
# Add path to parent dir of this Python file: https://stackoverflow.com/questions/3430372/
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from deep_learning.model import get_weights


def createSSHClient(server, port, user, password):
    """
    Create an SSH client to connect to server.
    Note: terminology might be confusing as also used to connect from FL server to FL client
    Function source: https://stackoverflow.com/questions/250283/how-to-scp-in-python

    :param server: str, remote ip address (denoted as server)
    :param port: int, port
    :param user: str, remote username
    :param password: str, password corresponding to remote username
    :return: ssh client
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def send_file(remote_ip_address, username, password, sender_file_path, workspace_path_sender, workspace_path_receiver):
    """ Send file to remote

    :param remote_ip_address: str, remote ip address
    :param username: str, username of remote
    :param password: str, password of remote
    :param sender_file_path: str, path to local file to share
    :param workspace_path_sender: str, path to sender workspace
    :param workspace_path_receiver: str, path to receiver workspace
    """

    # Define paths
    receiver_file_path = sender_file_path.replace(workspace_path_sender, workspace_path_receiver)
    sender_txt_file_path = sender_file_path.replace(os.path.splitext(sender_file_path)[1], '_transfer_completed.txt')
    receiver_txt_file_path = sender_txt_file_path.replace(workspace_path_sender, workspace_path_receiver)

    # Prepare the txt file that marks the end of the file transfer
    with open(sender_txt_file_path, 'w') as file:
        file.write(f'The following file was succesfully transferred: {os.path.basename(sender_file_path)}')

    # Use "cp" command if local simulation
    # Note: Due to persisting Exception when running locally on Mac:
    # ==> "SSHException: Error reading SSH protocol banner"
    if remote_ip_address == '127.0.0.1':
        os.system(f'cp {sender_file_path} {receiver_file_path}')            # Send info to receiver
        os.system(f'cp {sender_txt_file_path} {receiver_txt_file_path}')    # Completion marker
    else:
        # Create ssh and scp client
        # Source to fix issue "scp.SCPException: Timeout waiting for scp response":
        # ==> https://github.com/ktbyers/netmiko/issues/1254
        ssh = createSSHClient(remote_ip_address, 22, username, password)
        scp = SCPClient(ssh.get_transport(), socket_timeout=60)

        scp.put(sender_file_path, remote_path=receiver_file_path)           # Send info to receiver
        scp.put(sender_txt_file_path, receiver_txt_file_path)               # Completion marker


def wait_for_file(file_path, stop_with_stop_file = False):
    """ This function waits for a file path to exist

    :param file_path: str, path to file
    :param stop_with_stop_file: bool, stop when "stop_training.txt" is present in the same directory?
    :return: bool, is a stop file present? Indicates stopping FL.
    """

    stop_file_present = False
    while not os.path.exists(file_path):
        if os.path.exists(os.path.join(os.path.dirname(file_path), 'stop_training.txt')):
            if stop_with_stop_file:
                stop_file_present = True
                break
            else:
                pass
        else:
            pass
    return stop_file_present


def send_to_all_clients(client_info_dict, path_to_file, ws_path_server):
    """
    This function sends a file to all clients

    :param client_info_dict: dict, k: client name, v: client information dict
    """
    file = os.path.basename(path_to_file)
    print(f'==> Sending {file} to all clients...')
    for client_name in client_info_dict.keys():
        print(f'    ==> Sending to {client_name} ...')
        ip_address = client_info_dict[client_name]['ip_address']
        username = client_info_dict[client_name]['username']
        password = client_info_dict[client_name]['password']
        workspace_path = client_info_dict[client_name]['ws_path']
        send_file(ip_address, username, password, path_to_file, ws_path_server, workspace_path)


def collect_client_info(client_info_dict, workspace_path_server, info_type, file_ext, fl_round=None, net_arch=None):
    """
    Collect workspace paths or dataset size from clients

    :param client_info_dict: dict, k: client name, v: dict with all info for specific client
    :param workspace_path_server: str, absolute path to server workspace
    :param info_type: str, which info to expect. Currently, supports 'dataset_size' or 'workspace_path'
    :param file_ext: str, file extension
    :param fl_round: int, federated learning round
    :param net_arch: torch model, network architecture. Will be updated with received state dict (.pt)
    :return: dict, enriched client information dict
    """

    # Prep
    print(f'==> Collecting all client {info_type}...')

    # Add client info to client_info_dict
    for client_name in client_info_dict.keys():
        print(f'     ==> Waiting for {client_name}...')
        # Define path to client info, extract file type and wait
        if fl_round is not None:
            client_info_path = os.path.join(workspace_path_server,
                                            f'{client_name}_round_{fl_round}_{info_type}{file_ext}')
        else:
            client_info_path = os.path.join(workspace_path_server, f'{client_name}_{info_type}{file_ext}')
        suffix = pathlib.Path(client_info_path).suffix
        wait_for_file(client_info_path.replace(suffix, '_transfer_completed.txt'))

        # Define action based on file type
        if suffix == '.txt':
            with open(client_info_path, 'r') as file:
                info = file.read()
                if info_type == 'dataset_size':
                    info = int(info)
        elif suffix == '.csv':
            info = pd.read_csv(client_info_path)
        elif suffix == '.pt':
            info = get_weights(net_arch, client_info_path)
        else:
            raise ValueError(f'No processing of file extension {suffix}')

        # Add to dictionary (add level if specific for FL round)
        if fl_round is not None:
            client_info_dict[client_name][f'round_{fl_round}'][info_type] = info
        else:
            client_info_dict[client_name][info_type] = info

    return client_info_dict


def send_client_info_to_server(client_n, client_ws_path, client_name, server_ip_address, server_username,
                               server_password, server_ws_path):
    """
    Send client information  to server

    :client_n: str, client dataset size
    :client_ws_path: str, path to client workspace
    :client_name: str, client name
    :server_ip_address: str, server ip address
    :server_username: str, server username
    :server_password: str, server password
    :server_ws_path: str, server workspace path
    """
    for tag, info in zip(['dataset_size', 'ws_path'], [client_n, client_ws_path]):
        print(f'==> Send {tag} to server...')
        info_txt_path = os.path.join(client_ws_path, f'{client_name}_{tag}.txt')
        with open(info_txt_path, 'w') as file:
            file.write(str(info))

        send_file(server_ip_address, server_username, server_password, info_txt_path, client_ws_path,
                  server_ws_path)


def send_test_df_to_server(test_df_for_server, client_name, workspace_path_client, server_username,
                           server_password, server_ip_address, workspace_path_server):
    """
    Send test dataframe to server

    :client_name: str, client name
    :workspace_path_client: str, path to client workspace
    :server_username: str, server username
    :server_password: str, server password
    :server_ip_address: str, server ip address
    :workspace_path_server: str, server workspace path
    """
    # Send results to server
    print('==> Sending test results to server...')
    test_df_path = os.path.join(workspace_path_client, f'{client_name}_test_results.csv')
    test_df_for_server.to_csv(test_df_path, index=False)
    send_file(server_ip_address, server_username, server_password, test_df_path, workspace_path_client,
              workspace_path_server)


def clean_up_workspace(workspace_dir_path, who):
    """
    Clean up workspace by removing, moving and copying files and dirs

    :param workspace_dir_path: str, path to workspace directory
    :param who: str, "server" or "client". Defines whether client or server workspace.
    """
    # Remove _transfer_completed.txt files and __pycache__
    remove_transfer_completion_files(workspace_dir_path)
    pycache_path = os.path.join(workspace_dir_path, '__pycache__')
    if os.path.exists(pycache_path):
        os.system(f'rm -r {pycache_path}')

    # Create unique experiment folder (date and time)
    date_time_folder_name = f'{str(dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))}'
    date_time_folder_path = os.path.join(workspace_dir_path, date_time_folder_name)
    if not os.path.exists(date_time_folder_path):
        os.mkdir(date_time_folder_path)

    # Create subdirectories per category
    subdirs = ['state_dicts', 'results', 'data', 'settings']
    if who == 'server':
        subdirs.remove('data')
        subdirs.append('log')
    for subdir in subdirs:
        subdir_path = os.path.join(date_time_folder_path, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

    # Move contents of workspace directory to subdirectories
    # Note: exceptions are FL plan, FL settings and architecture.
    # ==> Keep in server parent folder for ease of running more experiments
    for root, dirs, files in os.walk(workspace_dir_path):
        # Do not process elements already in experiment folder
        if experiment_folder_in_path(root):
            continue

        for file in files:
            src_file_path = os.path.join(root, file)
            # Result files
            if (any(file.endswith(ext) for ext in ['.png', '.csv'])
                    or file in ['overall_test_mae.txt', 'final_model.txt']):
                dest_file_path = os.path.join(date_time_folder_path, 'results', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            # Settings files
            elif any(file.endswith(ext) for ext in ['.json', 'ws_path.txt', 'dataset_size.txt', '.py',
                                                    'stop_training.txt']):
                dest_file_path = os.path.join(date_time_folder_path, 'settings', file)
                if file == f'FL_settings_{who}.json':
                    os.system(f'cp {src_file_path} {dest_file_path}')
                elif file in ['architecture.py', 'FL_plan.json'] and who == 'server':
                    os.system(f'cp {src_file_path} {dest_file_path}')
                else:
                    os.system(f'mv {src_file_path} {dest_file_path}')
            # Data files
            elif file.endswith('tsv'):
                dest_file_path = os.path.join(date_time_folder_path, 'data', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            # Log files
            elif file in ['FL_duration.txt', 'aggregation_samples.txt']:
                dest_file_path = os.path.join(date_time_folder_path, 'log', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            # State dicts
            elif file.endswith('.pt'):
                dest_file_path = os.path.join(date_time_folder_path, 'state_dicts', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
        break


def experiment_folder_in_path(path):
    """
    Check whether the directory has an experiment folder format

    :param path: str, path
    :return: bool
    """

    return sum([bool(re.fullmatch('\A[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}h[0-9]{2}m[0-9]{2}s\Z', i))
               for i in path.split(os.sep)]) > 0


def remove_transfer_completion_files(workspace_dir_path, print_tracking=False):
    """
    Remove files with '_transfer_completed.txt' suffix

    :param workspace_dir_path: str, path to workspace directory
    :param print_tracking: bool, track which files are removed?
    """
    for root, dirs, files in os.walk(workspace_dir_path):
        for file in files:
            if file.endswith('_transfer_completed.txt'):
                if print_tracking:
                    print(f'removing: {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))
