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
from requests.adapters import HTTPAdapter, Retry
from requests.auth import HTTPBasicAuth
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# Add path to parent dir of this Python file: https://stackoverflow.com/questions/3430372/
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from deep_learning.model import get_weights
import shutil


def file_present_in_moderator_ws(url, username, password):
    # Send credentials in "params" attribute instead of "auth", as it did not work outside local environment
    # "params" parsed with ".args" at server side
    response = requests.get(url, params={'username': username, 'password': password})
    return response.text != 'This file is not yet present.'


def download_file(url: str, save_path: str, username: str, password: str, encrypted_aes_key=None, iv=None,
                  private_rsa_key=None, stop_with_stop_file=False, poll_interval: float = 1.0):
    """
    Adapted from: https://realpython.com/python-download-file-from-url/
    Additional source:
    - https://stackoverflow.com/questions/60798728/put-a-time-limit-on-a-request (to avoid hanging when no response from server)
    - ChatGPT
    Action: downloads.
    returns: Boolean (downloaded?)
    """
    session = create_session()

    while True:  # Infinite loop
        response = session.get(
            url,
            auth=HTTPBasicAuth(username, password),
            stream=True,
            timeout=30,
        )

        if response.status_code == 404:
            if stop_with_stop_file:
                stop_response = session.get(
                    url.replace(os.path.basename(url), 'stop_training.txt'),
                    auth=HTTPBasicAuth(username, password),
                    stream=True,
                    timeout=30,
                )
                if stop_response.status_code == 200:
                    return 'stop file present'
            time.sleep(poll_interval)
            continue

        if response.status_code == 200:

            tmp_save_path = save_path + ".tmp"

            # Step 1: download encrypted or plain file
            with open(tmp_save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            # Step 2: decrypt if needed
            if encrypted_aes_key is None and iv is None and private_rsa_key is None:

                # no decryption needed → just rename
                os.replace(tmp_save_path, save_path)

            elif encrypted_aes_key is not None and iv is not None and private_rsa_key is not None:

                # read encrypted
                with open(tmp_save_path, "rb") as f:
                    encrypted_bytes = f.read()

                decrypted_bytes = decrypt_message(
                    encrypted_bytes,
                    encrypted_aes_key,
                    iv,
                    private_rsa_key
                )

                # overwrite temp file with decrypted
                with open(tmp_save_path, "wb") as f:
                    f.write(decrypted_bytes)

                # atomic replace
                os.replace(tmp_save_path, save_path)

            else:
                os.remove(tmp_save_path)
                raise ValueError("Decryption parameters incomplete")

            return 'download successful'

        response.raise_for_status()


def upload_file(url_upload: str, local_path: str, username: str, password: str, aes_key=None, iv=None, timeout: int = 300):
    """
    Sources:
    - https://stackoverflow.com/questions/68477/send-file-using-post-from-a-python-script
    - https://proxiesapi.com/articles/a-beginner-s-guide-to-uploading-files-with-python-requests
    - ChatGPT
    """
    with open(local_path, 'rb') as f:
        file_bytes = f.read()
    if aes_key is not None and iv is not None:
        file_bytes = aes_encrypt(aes_key, file_bytes, iv)

    session = create_session()

    response = session.post(
        os.path.join(url_upload, os.path.basename(local_path)),
        files={"file": file_bytes},
        auth=HTTPBasicAuth(username, password),
        timeout=timeout
    )

    response.raise_for_status()
    return True


def create_session():
    # Sources:
    # - https://stackoverflow.com/questions/15778466/using-python-requests-sessions-cookies-and-post
    # - https://github.com/psf/requests/blob/main/docs/user/advanced.rst#example-automatic-retries
    # - https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
    # - ChatGPT

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def wait_for_file(file_path: str, download_url_base: str, username: str, password: str, aes_key=None, iv=None,
                  private_rsa_key=None, stop_with_stop_file: bool = False, poll_interval: float = 1.0,
                  timeout: float = None) -> bool:
    """
    Wait until a file becomes available on the moderator server and download it.
    Supports optional stop file detection and encryption parameters.
    Sources: ChatGPT

    Parameters
    ----------
    file_path : str. Local full path where the file should be saved.
    download_url_base : str. Base URL of moderator download folder.
    username : str. HTTP basic auth username.
    password : str. HTTP basic auth password.
    aes_key : bytes, optional. AES key for decryption.
    iv : bytes, optional. AES IV for decryption.
    private_rsa_key : RSA key, optional. RSA private key for AES key decryption.
    stop_with_stop_file : bool, default False. If True, abort when "stop_training.txt" is detected.
    poll_interval : float, default 1.0. Seconds between retry attempts.
    timeout : float, optional. Maximum seconds to wait. None = wait forever.

    Returns
    -------
    bool
        True  → stop file detected (caller should stop training)
        False → requested file successfully downloaded
    """

    filename = os.path.basename(file_path)
    file_url = os.path.join(download_url_base, filename)

    start_time = time.time()

    while True:

        # Try to download target file
        download_response = download_file(
            url=file_url,
            save_path=file_path,
            username=username,
            password=password,
            encrypted_aes_key=aes_key,
            iv=iv,
            private_rsa_key=private_rsa_key,
            stop_with_stop_file=stop_with_stop_file
        )

        if download_response == 'download successful':
            return False  # File downloaded successfully
        elif download_response == 'stop file present':
            return True  # Stop file detected

        # Check timeout
        if timeout is not None and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Timeout waiting for file: {filename}")

        time.sleep(poll_interval)


def collect_client_info(client_info_dict, workspace_path_server, info_type, file_ext, moderator_download_folder_url,
                        download_username, download_password, server_private_rsa_key=None, fl_round=None, net_arch=None):
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

        if info_type in ['aes_key', 'iv', 'public_rsa_key']:
            wait_for_file(client_info_path, moderator_download_folder_url, download_username, download_password)
        else:
            wait_for_file(client_info_path, moderator_download_folder_url, download_username, download_password,
                          client_info_dict[client_name]['aes_key'], client_info_dict[client_name]['iv'], server_private_rsa_key)

        # Define action based on file type
        suffix = pathlib.Path(client_info_path).suffix
        if suffix == '.txt':
            with open(client_info_path, 'rb') as file:
                info = file.read()
                if info_type == 'dataset_size':
                    info = int(info)
                elif info_type == 'public_rsa_key':
                    info = receive_public_key(info)
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


def send_client_n_to_moderator(client_n, client_ws_path, client_name, url_upload, upload_username, upload_password, aes_key, iv):
    """
    Send client dataset size to moderator

    :client_n: str, client dataset size
    :client_ws_path: str, path to client workspace
    :client_name: str, client name
    """

    print('==> Send dataset size to server...')
    n_txt_path = os.path.join(client_ws_path, f'{client_name}_dataset_size.txt')
    with open(n_txt_path, 'w') as file:
        file.write(str(client_n))

    upload_file(url_upload, n_txt_path, upload_username, upload_password, aes_key, iv)


def send_test_df_to_moderator(test_df_for_server, client_name, workspace_path_client, url_upload, upload_username, upload_password, aes_key, iv):
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
    upload_file(url_upload, test_df_path, upload_username, upload_password, aes_key, iv)

 
def clean_up_workspace(workspace_dir_path, who):
    """
    Clean up workspace by removing, moving and copying files and dirs

    :param workspace_dir_path: str, path to workspace directory
    :param who: str, "server" or "client". Defines whether client or server workspace.
    """
    # Remove __pycache__
    pycache_path = os.path.join(workspace_dir_path, '__pycache__')
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

    # Create unique experiment folder (date and time)
    date_time_folder_name = f'{str(dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))}'
    date_time_folder_path = os.path.join(workspace_dir_path, date_time_folder_name)
    if not os.path.exists(date_time_folder_path):
        os.mkdir(date_time_folder_path)

    # Create subdirectories per category
    subdirs = ['state_dicts', 'results', 'data', 'settings', 'log']
    if who == 'server':
        subdirs.remove('data')
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
                shutil.move(src_file_path, dest_file_path)
            # Settings files
            elif (any(file.endswith(ext) for ext in ['.json', 'dataset_size.txt', '.py',
                                                    'stop_training.txt', 'aes_key.txt', 'iv.txt', 'public_rsa_key.txt',
                                                    'public_rsa_key_server.txt'])
                  or (any(file.startswith(prefix) for prefix in ['server_aes_key_for', 'server_iv_for']))):
                dest_file_path = os.path.join(date_time_folder_path, 'settings', file)
                if file == f'FL_settings_{who}.json':
                    shutil.copyfile(src_file_path, dest_file_path)
                elif file in ['architecture.py', 'FL_plan.json'] and who == 'server':
                    shutil.copyfile(src_file_path, dest_file_path)
                else:
                    shutil.move(src_file_path, dest_file_path)
            # Data files
            elif file.endswith('tsv'):
                dest_file_path = os.path.join(date_time_folder_path, 'data', file)
                shutil.move(src_file_path, dest_file_path)
            # Log files
            elif file in ['FL_duration.txt', 'aggregation_samples.txt'] or file.endswith('.log'):
                dest_file_path = os.path.join(date_time_folder_path, 'log', file)
                shutil.move(src_file_path, dest_file_path)
            # State dicts
            elif file.endswith('.pt'):
                dest_file_path = os.path.join(date_time_folder_path, 'state_dicts', file)
                shutil.move(src_file_path, dest_file_path)
            # File that indicates moderator to clean workspace
            elif file == 'moderator_clean_ws.txt':
                os.remove(src_file_path)
        break


def experiment_folder_in_path(path):
    """
    Check whether the directory has an experiment folder format

    :param path: str, path
    :return: bool
    """

    return sum([bool(re.fullmatch('\A[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}h[0-9]{2}m[0-9]{2}s\Z', i))
               for i in path.split(os.sep)]) > 0


# Cryptography
# Functions based on the work of Manh Doan Quang
# Note: AES necessary on top of RSA to encrypt larger files:
# ==> https://stackoverflow.com/questions/65856980/python-rsa-message-encryption-plaintext-is-too-long
# Keys
def get_rsa_key_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    # public_key_pem = public_key_pem.decode('utf-8')
    return public_key_pem, private_key


def generate_aes_key():
    return os.urandom(32)  # AES-256 key


def receive_public_key(public_key_pem):
    # In original function by Manh: decode done before file transmission
    return serialization.load_pem_public_key(public_key_pem.decode('utf-8').encode('utf-8'))


# Encryption
def rsa_encrypt(receiver_public_rsa_key, message):
    # receiver_public_rsa_key = receiver_public_rsa_key.public_key()
    encrypted_message = receiver_public_rsa_key.encrypt(
        message,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )
    return encrypted_message


def aes_encrypt(aes_key, plaintext, iv):
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    padded_plaintext = plaintext + b" " * (16 - len(plaintext) % 16)
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return ciphertext


# Decryption
def decrypt_message(encrypted_message, encrypted_aes_key, iv, private_rsa_key):
    encrypted_aes_key = bytes(encrypted_aes_key)
    aes_key = rsa_decrypt(private_rsa_key, encrypted_aes_key)

    # Decrypt the model update using AES
    iv = bytes(iv)
    ciphertext = bytes(encrypted_message)
    decrypted_message = aes_decrypt(aes_key, iv, ciphertext)

    # # Deserialize the binary data to reconstruct the model weights
    # model_state_dict = pickle.loads(decrypted_update_str)

    return decrypted_message


def rsa_decrypt(private_key, encrypted_message):
    return private_key.decrypt(
        encrypted_message,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )


def aes_decrypt(aes_key, iv, ciphertext):
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
    return decrypted_data.strip()
