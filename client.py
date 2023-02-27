"""
Script for clients of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9
"""

import os
import json

import numpy as np
import torch
import argparse
import paramiko
import pandas as pd
import torch.nn as nn
from scp import SCPClient
from monai.networks.nets import DenseNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DL_utils.data import get_data_loader, split_data
from DL_utils.model import get_weights
from train import train


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
        file.write(f'The following file was succesfully transferred: {file_path}')
    scp.put(txt_file_path, txt_file_path)


def wait_for_file(file_path):
    """ This function waits for a file path to exist

    :param file_path: str, path to file
    """
    while not os.path.exists(file_path):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Client',
        description='Client for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    args = parser.parse_args()
    settings_path = args.settings_path

    print('\n\n==============================\nStarting federated learning :)\n==============================\n\n')

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path = settings_dict.get('workspace_path')            # Path to workspace for server and clients
    client_ip_address = settings_dict.get('client_ip_address')      # Client ip address
    server_ip_address = settings_dict.get('server_ip_address')      # Server ip address
    server_username = settings_dict.get('server_username')          # Server username
    server_password = settings_dict.get('server_password')          # Server password
    colname_id = settings_dict.get('colname_id')                    # Column name of the BIDS id column
    colname_img_path = settings_dict.get('colname_img_path')        # Column name of the image paths
    colname_label = settings_dict.get('colname_label')              # Column name of the label column
    subject_ids = settings_dict.get('subject_ids')                  # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')            # Path to BIDS root

    # Load dataframe and preprocess
    df_path = os.path.join(bids_root_path, 'participants.tsv')
    df = pd.read_csv(df_path, sep='\t')

    if colname_img_path is None:
        colname_img_path = 'img_path'
        df[colname_img_path] = df[colname_id].apply(
            lambda x: os.path.join(bids_root_path, 'derivatives', 'Wood_2022', str(x), 'anat', f'{x}_T1w.nii.gz'))
    colnames_dict = {'id': colname_id, 'img_path': colname_img_path, 'label': colname_label}

    if subject_ids is not None:
        df = df[df[colname_id].isin(subject_ids)].reset_index(drop=True)

    # Create workspace folder
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    # Wait for FL plan
    print('==> Waiting for FL plan...')
    FL_plan_path = os.path.join(workspace_path, 'FL_plan.json')
    wait_for_file(FL_plan_path.replace('.json', '_transfer_completed.txt'))

    # Extract FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                    # Number of FL rounds
    n_epochs = int(FL_plan_dict.get('n_epochs'))                    # Number of epochs per FL round
    transfer_learning = FL_plan_dict.get('transfer_learning')       # Only update FC layer?')
    lr = float(FL_plan_dict.get('lr'))                              # Learning rate
    patience = int(FL_plan_dict.get('patience'))                    # N epochs without loss reduction before reducing lr
    lr_reduce_factor = float(FL_plan_dict.get('lr_reduce_factor'))  # Factor by which to reduce LR on Plateau
    batch_size = int(FL_plan_dict.get('batch_size'))                # Batch size for the data loaders
    train_fraction = float(FL_plan_dict.get('train_fraction'))      # Fraction of data for training
    val_fraction = float(FL_plan_dict.get('val_fraction'))          # Fraction of data for validation
    test_fraction = float(FL_plan_dict.get('test_fraction'))        # Fraction of data for testing
    n_splits = int(FL_plan_dict.get('n_splits'))                    # Number of data splits per fl round

    # Send dataset size to server
    print('==> Send dataset size to server...')
    dataset_size_txt_path = os.path.join(workspace_path, f'{client_ip_address}_dataset_size.txt')
    with open(dataset_size_txt_path, 'w') as file:
        file.write(str(df.shape[0]))

    send_file(server_ip_address, server_username, server_password, dataset_size_txt_path)

    # General deep learning settings
    criterion = nn.L1Loss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Start federated learning
    for fl_round in range(n_rounds):
        print(f'\n*****************\nRound {fl_round}\n*****************\n')
        # Wait for global model to arrive
        print('==> Waiting for global model to arrive...')
        if fl_round == 0:
            global_model_path = os.path.join(workspace_path, f'initial_model.pt')
            global_txt_path = os.path.join(workspace_path, f'initial_model_transfer_completed.txt')
        else:
            global_model_path = os.path.join(workspace_path, f'global_model_round_{fl_round-1}.pt')
            global_txt_path = os.path.join(workspace_path, f'global_model_round_{fl_round-1}_transfer_completed.txt')

        wait_for_file(global_txt_path)

        # Create a state dict folder in the workspace for this training round
        print('==> Make preparations to start training...')
        state_dict_folder_path = os.path.join(workspace_path, f'state_dicts_{client_ip_address}_round_{fl_round}')
        if not os.path.exists(state_dict_folder_path):
            os.mkdir(state_dict_folder_path)

        # Load global network
        net_architecture = DenseNet(3, 1, 1)
        global_net = get_weights(net_architecture, global_model_path)
        if transfer_learning:
            # Freeze all weights in the network
            for param in global_net.parameters():
                param.requires_grad = False
            # Unfreeze weights of the fully connected layer
            for param in global_net.class_layers.out.parameters():
                param.requires_grad = True

        # Deep learning settings per FL round
        optimizer = torch.optim.Adam(global_net.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

        # Initiate variables
        best_model_path_across_splits = None
        local_model_path = None
        best_val_loss = np.inf
        random_states = range(n_splits*fl_round, n_splits*fl_round + n_splits)  # Assure random state is never repeated
        for split_i, random_state in enumerate(random_states):
            print(f'==> Split {split_i}, random state {random_state}...')
            # Split data
            train_df, val_df, test_df = split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction,
                                                   random_state=random_state)
            train_loader = get_data_loader(train_df, 'train', colnames_dict, batch_size=batch_size)
            val_loader = get_data_loader(val_df, 'validation', colnames_dict, batch_size=batch_size)
            test_loader = get_data_loader(val_df, 'test', colnames_dict, batch_size=batch_size)

            # Train
            print('==> Start training...')
            best_model_path, train_loss_list, val_loss_list = train(n_epochs, device, train_loader, val_loader,
                                                                    optimizer, global_net, criterion, scheduler,
                                                                    state_dict_folder_path)

            print('==> Send training results to server...')
            train_df = pd.DataFrame({'fl_round': [fl_round]*n_epochs,
                                     'train_loss': train_loss_list,
                                     'val_loss': val_loss_list})
            train_df_path = os.path.join(
                workspace_path, f'train_results_{client_ip_address}_round_{fl_round}_random_state_{random_state}.csv'
            )
            train_df.to_csv(train_df_path, index=False)
            send_file(server_ip_address, server_username, server_password, train_df_path)

            # Get best validation loss across all splits
            if min(val_loss_list) < best_val_loss:
                best_model_path_across_splits = best_model_path

        # Copy the best model in the state dict folder to the workspace folder
        local_model_path = os.path.join(workspace_path, f'model_{client_ip_address}_round_{fl_round}.pt')
        os.system(f'cp {best_model_path_across_splits} {local_model_path}')

        # Send to server
        print('==> Send best local model to server ...')
        send_file(server_ip_address, server_username, server_password, local_model_path)
