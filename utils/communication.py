"""
Functions related to communication between server and client
"""

import os
import paramiko
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
