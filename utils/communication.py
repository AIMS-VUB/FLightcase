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
