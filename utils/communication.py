"""
Functions related to communication between server and client
"""

import os
import re
import paramiko
import datetime as dt
from scp import SCPClient


def createSSHClient(server, port, user, password):
    """
    Function source: https://stackoverflow.com/questions/250283/how-to-scp-in-python
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
    # Create ssh and scp client
    # Source to fix issue "scp.SCPException: Timeout waiting for scp response":
    # ==> https://github.com/ktbyers/netmiko/issues/1254
    ssh = createSSHClient(remote_ip_address, 22, username, password)
    scp = SCPClient(ssh.get_transport(), socket_timeout=60)

    # Share model with receiver
    receiver_file_path = sender_file_path.replace(workspace_path_sender, workspace_path_receiver)
    scp.put(sender_file_path, remote_path=receiver_file_path)

    # Share txt file with client that marks the end of the file transfer
    sender_txt_file_path = sender_file_path.replace(os.path.splitext(sender_file_path)[1], '_transfer_completed.txt')
    receiver_txt_file_path = sender_txt_file_path.replace(workspace_path_sender, workspace_path_receiver)
    with open(sender_txt_file_path, 'w') as file:
        file.write(f'The following file was succesfully transferred: {os.path.basename(sender_file_path)}')
    scp.put(sender_txt_file_path, receiver_txt_file_path)


def wait_for_file(file_path, stop_with_stop_file = False):
    """ This function waits for a file path to exist

    :param file_path: str, path to file
    :param stop_with_stop_file: bool, stop when "stop_training.txt" is present in the same directory?
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


def clean_up_workspace(workspace_dir_path, server_or_client):
    # Remove _transfer_completed.txt files
    remove_transfer_completion_files(workspace_dir_path)

    # Create subdirectories per category
    subdirs = ['state_dicts', 'results', 'data']
    if server_or_client == 'server':
        subdirs.remove('data')
    for subdir in subdirs:
        subdir_path = os.path.join(workspace_dir_path, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

    # Move files to subdirectories
    for root, dirs, files in os.walk(workspace_dir_path):
        for file in files:
            src_file_path = os.path.join(root, file)
            if any(file.endswith(ext) for ext in ['.png', '.csv']):
                dest_file_path = os.path.join(root, 'results', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            elif file.endswith('tsv'):
                dest_file_path = os.path.join(root, 'data', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            elif file.endswith('.pt'):
                dest_file_path = os.path.join(root, 'state_dicts', file)
                os.system(f'mv {src_file_path} {dest_file_path}')
            elif any(file.endswith(ext) for ext in ['.json', '.txt', '.py', '.pyc']):
                pass
            else:
                raise ValueError(f'No handler specified for file extension (file = {file})')
        break

    # Move contents of workspace directory to data and time folder
    date_time_folder_name = f'{str(dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))}'
    date_time_folder_path = os.path.join(workspace_dir_path, date_time_folder_name)
    if not os.path.exists(date_time_folder_path):
        os.mkdir(date_time_folder_path)
    for element in os.listdir(workspace_dir_path):
        if is_experiment_folder(element):
            continue
        src_file_path = os.path.join(root, element)
        dest_file_path = os.path.join(root, date_time_folder_name, element)
        os.system(f'mv {src_file_path} {dest_file_path}')


def is_experiment_folder(dir_name):
    return re.fullmatch('\A[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}h[0-9]{2}m[0-9]{2}s\Z', dir_name)


def remove_transfer_completion_files(workspace_dir_path, print_tracking=False):
    for root, dirs, files in os.walk(workspace_dir_path):
        for file in files:
            if file.endswith('_transfer_completed.txt'):
                if print_tracking:
                    print(f'removing: {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))
