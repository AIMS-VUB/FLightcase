"""
Functions related to communication between server and client
"""

import os
import re
import sys
import time
import pathlib
import requests
import pandas as pd
import datetime as dt
# Add path to parent dir of this Python file: https://stackoverflow.com/questions/3430372/
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from deep_learning.model import get_weights


def file_present_in_moderator_ws(url, username, password):
    # Send credentials in "params" attribute instead of "auth", as it did not work outside local environment
    # "params" parsed with ".args" at server side
    response = requests.get(url, params={'username': username, 'password': password})
    return response.text != 'This file is not yet present.'


def download_file(url, download_location, username, password, download_if_exists=False):
    """
    Adapted from: https://realpython.com/python-download-file-from-url/
    Action: downloads.
    returns: Boolean (downloaded?)
    """

    response = requests.get(url, params={'username': username, 'password': password})

    # Check whether to proceed or not
    if response.status_code != 200:
        return False
    elif response.text in ['The downloader is not recognized.',
                           'The downloader is recognized, but the password is incorrect.',
                           'The downloader did not provide credentials.',
                           'This file is not yet present.']:
        return False

    # Get filename
    if "content-disposition" in response.headers:
        content_disposition = response.headers["content-disposition"]
        filename = content_disposition.split("filename=")[1]
    else:
        filename = url.split("/")[-1]

    # Define file path
    # Split filename source: https://stackoverflow.com/questions/541390/extracting-extension-from-filename
    file_path = os.path.join(download_location, filename)
    file_path_no_extension, ext = os.path.splitext(file_path)

    # Do not overwrite file if already exists. Add copy number.
    if download_if_exists:
        copy_nr = 1
        while os.path.exists(file_path):
            file_path = f'{file_path_no_extension} ({copy_nr}){ext}'
            copy_nr += 1
    else:
        if os.path.exists(file_path):
            return True

    content = response.content
    if b'404 Not Found' in content:
        return False
    else:
        with open(file_path, mode="wb") as file:
            file.write(response.content)
        return True


def upload_file(url_upload, local_path, username, password):
    """
    Sources:
    - https://stackoverflow.com/questions/68477/send-file-using-post-from-a-python-script
    - https://proxiesapi.com/articles/a-beginner-s-guide-to-uploading-files-with-python-requests
    """
    with open(local_path, 'rb') as f:
        file_bytes = f.read()
    files = {'file': (os.path.basename(local_path), file_bytes)}

    # Keep trying to upload (sometimes status code 500 returned by server)
    response_text = ''
    while response_text != 'Upload successful!':
        response = requests.post(os.path.join(url_upload, os.path.basename(local_path)), files=files,
                                 params={'username': username, 'password': password,
                                         'file_size': os.path.getsize(local_path)})
        response_text = response.text
        time.sleep(1)


def wait_for_file(file_path, moderator_download_folder_url, download_username, download_password, stop_with_stop_file=False):
    """ This function waits for a file path to exist

    :param file_path: str, path to file
    :param moderator_download_folder_url: str, path to download folder of the moderator
    :param download_username: str, remote username
    :param download_password: str, password corresponding to remote username
    :param stop_with_stop_file: bool, stop when "stop_training.txt" is present in the same directory?
    :return: bool, is a stop file present? Indicates stopping FL.
    """

    stop_training = False

    # Download the target file.
    # Note: Here, file completion does not need to be flagged as the path only exists after download
    file = os.path.basename(file_path)
    file_url = os.path.join(moderator_download_folder_url, file)
    workspace_receiver = os.path.dirname(file_path)
    while not download_file(file_url, workspace_receiver, download_username, download_password):
        if download_file(os.path.join(moderator_download_folder_url, 'stop_training.txt'), workspace_receiver, download_username, download_password) and stop_with_stop_file:
            stop_training = True
            break
        pass

    return stop_training


def collect_client_info(client_info_dict, workspace_path_server, info_type, file_ext, moderator_download_folder_url,
                        download_username, download_password, fl_round=None, net_arch=None):
    """
    Collect workspace paths or dataset size from clients

    :param client_info_dict: dict, k: client name, v: dict with all info for specific client
    :param workspace_path_server: str, absolute path to server workspace
    :param info_type: str, which info to expect. Currently, supports 'dataset_size' or 'workspace_path'
    :param file_ext: str, file extension
    :param moderator_download_folder_url: str, URL where to download files from moderator
    :param download_username: str, remote username
    :param download_password: str, password corresponding to remote username
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
        wait_for_file(client_info_path, moderator_download_folder_url, download_username, download_password)

        # Define action based on file type
        suffix = pathlib.Path(client_info_path).suffix
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


def send_client_info_to_moderator(client_n, client_ws_path, client_name, url_upload, upload_username, upload_password):
    """
    Send client information  to moderator

    :client_n: str, client dataset size
    :client_ws_path: str, path to client workspace
    :client_name: str, client name
    """
    for tag, info in zip(['dataset_size', 'ws_path'], [client_n, client_ws_path]):
        print(f'==> Send {tag} to server...')
        info_txt_path = os.path.join(client_ws_path, f'{client_name}_{tag}.txt')
        with open(info_txt_path, 'w') as file:
            file.write(str(info))

        upload_file(url_upload, info_txt_path, upload_username, upload_password)


def send_test_df_to_moderator(test_df_for_server, client_name, workspace_path_client, url_upload, upload_username, upload_password):
    """
    Send test dataframe to server

    :client_name: str, client name
    :workspace_path_client: str, path to client workspace
    :moderator_username: str, moderator username
    :moderator_password: str, moderator password
    :moderator_ip_address: str, moderator ip address
    :workspace_path_moderator: str, moderator workspace path
    """
    # Send results to server
    print('==> Sending test results to server...')
    test_df_path = os.path.join(workspace_path_client, f'{client_name}_test_results.csv')
    test_df_for_server.to_csv(test_df_path, index=False)
    upload_file(url_upload, test_df_path, upload_username, upload_password)


def clean_up_workspace(workspace_dir_path, who):
    """
    Clean up workspace by removing, moving and copying files and dirs

    :param workspace_dir_path: str, path to workspace directory
    :param who: str, "server" or "client". Defines whether client or server workspace.
    """
    # Remove __pycache__
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
            # File that indicates moderator to clean workspace
            elif file == 'moderator_clean_ws':
                os.system(f'rm {src_file_path}')
        break


def experiment_folder_in_path(path):
    """
    Check whether the directory has an experiment folder format

    :param path: str, path
    :return: bool
    """

    return sum([bool(re.fullmatch('\A[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}h[0-9]{2}m[0-9]{2}s\Z', i))
               for i in path.split(os.sep)]) > 0
