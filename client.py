"""
Script for clients of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9
"""

import os
import copy
import json
import numpy as np
import torch
import argparse
import paramiko
import pandas as pd
import torch.nn as nn
from scp import SCPClient
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import OrderedDict
from monai.networks.nets import DenseNet
from sklearn.metrics import mean_absolute_error
from DL_utils.data import get_data_loader, split_data
from DL_utils.model import get_weights
from DL_utils.evaluation import evaluate
from train import train

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
        file.write(f'The following file was succesfully transferred: {file_path}')
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


def prepare_for_transfer_learning(net, method, print_trainable_params=False):
    """
    Prepare torch neural network for transfer learning
    ==> freeze all weights except those in the fully connected layer
    :param net: Torch net
    :param method: str, method of transfer learning. Choose from:
        ['no_tl', 'freeze_up_to_trans_1', 'freeze_up_to_trans_2', 'freeze_up_to_trans_3', 'freeze_up_to_norm_5']
    :param print_trainable_params: bool
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

    elif method == 'no_tl':
        pass
    else:
        raise ValueError('Transfer learning method not recognised')

    # Print number of trainable parameters
    if print_trainable_params:
        print('Number of trainable parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    return net


def freeze(parameters):
    for param in parameters:
        param.requires_grad = False


def copy_net(net):
    """ Copy torch network

    Source: https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
    """
    net_copy = copy.deepcopy(net)
    return net_copy


def loss_to_contribution(loss_list):
    """
    Convert loss list into normalised weights.
    Higher loss = lower contribution.
    """
    contribution_weights = [1 / val for val in loss_list]
    contribution_weights_normalised = [val / (sum(contribution_weights)) for val in contribution_weights]
    return contribution_weights_normalised


def get_weighted_average_model(net_architecture, path_error_dict):
    """
    Get weighted average of multiple models.
    The contribution of a model is based on its loss (higher loss = lower contribution)
    """
    # Convert loss to normalised contribution (so that sum of the contributions is 1)
    path_contribution_dict = dict(zip(path_error_dict.keys(), loss_to_contribution(path_error_dict.values())))

    state_dict_avg = OrderedDict()
    for i, (path, contribution) in enumerate(path_contribution_dict.items()):
        # Load network weights
        net = get_weights(copy_net(net_architecture), path)
        net.to(torch.device('cpu'))
        state_dict = net.state_dict()

        for key in state_dict.keys():
            state_dict_contribution = state_dict[key] * contribution
            if i == 0:
                state_dict_avg[key] = state_dict_contribution
            else:
                state_dict_avg[key] += state_dict_contribution

    weighted_avg_net = DenseNet(3, 1, 1)
    weighted_avg_net.load_state_dict(state_dict_avg)
    return weighted_avg_net


def get_criterion(criterion_txt):
    if criterion_txt == 'l1loss':
        return nn.L1Loss(reduction='sum')
    else:
        raise ValueError(f'Cannot find criterion for {criterion_txt}')


def get_optimizer(optimizer_txt, net, lr):
    if optimizer_txt == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError(f'Cannot find optimizer for {optimizer_txt}')


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
    client_name = settings_dict.get('client_name')                  # Client name
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
    batch_size = int(FL_plan_dict.get('batch_size'))                # Batch size for the data loaders
    train_fraction = float(FL_plan_dict.get('train_fraction'))      # Fraction of data for training
    val_fraction = float(FL_plan_dict.get('val_fraction'))          # Fraction of data for validation
    test_fraction = float(FL_plan_dict.get('test_fraction'))        # Fraction of data for testing
    n_splits = int(FL_plan_dict.get('n_splits'))                    # Number of data splits per fl round
    lr = float(FL_plan_dict.get('lr'))                              # Learning rate
    lr_reduce_factor = float(FL_plan_dict.get('lr_reduce_factor'))  # Factor by which to reduce LR on Plateau
    patience_lr_reduction = int(FL_plan_dict.get('pat_lr_red'))     # N fl rounds stagnating val loss before reducing lr
    criterion_txt = FL_plan_dict.get('criterion')                   # Criterion in txt format, lowercase (e.g. l1loss)
    optimizer_txt = FL_plan_dict.get('optimizer')                   # Optimizer in txt format, lowercase (e.g. adam)
    tl_method = FL_plan_dict.get('tl_method')                       # Get transfer learning method

    # Send dataset size to server
    print('==> Send dataset size to server...')
    dataset_size = df.shape[0]
    dataset_size_txt_path = os.path.join(workspace_path, f'{client_name}_dataset_size.txt')
    with open(dataset_size_txt_path, 'w') as file:
        file.write(str(dataset_size))

    send_file(server_ip_address, server_username, server_password, dataset_size_txt_path)

    # General deep learning settings
    criterion = get_criterion(criterion_txt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_architecture = DenseNet(3, 1, 1)
    test_loader = None

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf       # Reference validation loss
    counter_lr_red = 0          # Counter for lr reduction

    # Start federated learning
    for fl_round in range(1, n_rounds + 1):  # Start counting from 1
        print(f'\n*****************\nRound {fl_round}\n*****************\n')

        # Wait for global model to arrive
        print('==> Waiting for global model to arrive...')
        if fl_round == 1:
            global_model_path = os.path.join(workspace_path, f'initial_model.pt')
            global_txt_path = os.path.join(workspace_path, f'initial_model_transfer_completed.txt')
        else:
            global_model_path = os.path.join(workspace_path, f'global_model_round_{fl_round-1}.pt')
            global_txt_path = os.path.join(workspace_path, f'global_model_round_{fl_round-1}_transfer_completed.txt')

        stop_training = wait_for_file(global_txt_path, stop_with_stop_file=True)
        if stop_training:
            break

        # Create a state dict folder in the workspace for this training round
        print('==> Make preparations to start training...')
        state_dict_folder_path = os.path.join(workspace_path, f'state_dicts_{client_name}_round_{fl_round}')
        if not os.path.exists(state_dict_folder_path):
            os.mkdir(state_dict_folder_path)

        # Load global network and prepare for transfer learning
        global_net = get_weights(net_architecture, global_model_path)
        global_net = prepare_for_transfer_learning(global_net, tl_method, print_trainable_params=True)

        # Deep learning settings per FL round
        optimizer = get_optimizer(optimizer_txt, global_net, lr)

        # Initiate variables
        path_error_dict = {}
        mean_val_loss = 0
        random_states = range(n_splits*fl_round, n_splits*fl_round + n_splits)  # Assure random state is never repeated
        train_results_df = pd.DataFrame()
        for split_i, random_state in enumerate(random_states):
            print(f'==> Split {split_i}, random state {random_state}...')
            # Split data
            # Note: Fix train_test_random_state to assure test data is always the same
            train_df, val_df, test_df = split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction,
                                                   train_test_random_state=42, train_val_random_state=random_state)
            train_loader, n_train = get_data_loader(train_df, 'train', colnames_dict, batch_size, return_n=True)
            val_loader, n_val = get_data_loader(val_df, 'validation', colnames_dict, batch_size, return_n=True)
            test_loader, n_test = get_data_loader(test_df, 'test', colnames_dict, batch_size, return_n=True)

            # Train
            print('==> Start training...')
            best_model_path, train_loss_list, val_loss_list = train(1, device, train_loader, val_loader, optimizer,
                                                                    global_net, criterion, None, state_dict_folder_path)

            train_results_df_i = pd.DataFrame({'random_state': [random_state],
                                               'fl_round': [fl_round],
                                               'train_loss': train_loss_list,
                                               'val_loss': val_loss_list,
                                               'n_train': n_train,
                                               'n_val': n_val,
                                               'n_test': n_test})
            train_results_df = pd.concat([train_results_df, train_results_df_i], axis=0)

            # Update mean val loss
            mean_val_loss += val_loss_list[0]/n_splits

            # Update dict
            path_error_dict.update({best_model_path: val_loss_list[0]})

        print('==> Send training results to server...')
        train_results_df_path = os.path.join(workspace_path, f'train_results_{client_name}_round_{fl_round}.csv')
        train_results_df.to_csv(train_results_df_path, index=False)
        send_file(server_ip_address, server_username, server_password, train_results_df_path)

        # Get local model
        local_model = get_weighted_average_model(net_architecture, path_error_dict)
        local_model_path = os.path.join(workspace_path, f'model_{client_name}_round_{fl_round}.pt')
        torch.save(local_model.state_dict(), local_model_path)

        # Send to server
        print('==> Send model with weighted average fc to server ...')
        send_file(server_ip_address, server_username, server_password, local_model_path)

        # Perform actions based on min validation loss across splits and epochs
        print('==> Validation loss tracking...')
        if mean_val_loss < val_loss_ref:    # Improvement
            val_loss_ref = mean_val_loss
            counter_lr_red = 0
        else:                               # No improvement
            counter_lr_red += 1
            if counter_lr_red == patience_lr_reduction:
                lr *= lr_reduce_factor
                FL_plan_dict['lr'] = lr
                counter_lr_red = 0
        print(f'     ==> lr reduction counter: {counter_lr_red}')

    # Test final model
    print('==> Waiting for final model...')
    final_model_path = os.path.join(workspace_path, 'final_model.pt')
    wait_for_file(final_model_path.replace('final_model.pt', 'final_model_transfer_completed.txt'))
    print('==> Testing final model...')
    global_net = get_weights(net_architecture, final_model_path)
    test_loss, true_labels_test, pred_labels_test = evaluate(global_net, test_loader, criterion, device, 'test')
    test_mae = mean_absolute_error(true_labels_test, pred_labels_test)
    r_true_pred, p_true_pred = stats.pearsonr(true_labels_test, pred_labels_test)

    # Save predictions and ground truth to workspace
    true_pred_test_df = pd.DataFrame({'true': true_labels_test, 'pred': pred_labels_test})
    true_pred_test_df.to_csv(os.path.join(workspace_path, f'true_pred_test_{client_name}.csv'))

    # Create scatterplot
    fig, ax = plt.subplots()
    ax.scatter(x=true_labels_test, y=pred_labels_test)
    ax.set_title(f'Test performance {client_name}:\nMAE: {test_mae:.2f}, r: {r_true_pred:.2f} (p: {p_true_pred:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    plt.savefig(os.path.join(workspace_path, f'scatterplot_true_pred_test_{client_name}.png'))

    # Send results to server
    print('==> Sending test results to server...')
    stat_sw_true, p_sw_true = stats.shapiro(true_pred_test_df['true'])
    stat_sw_pred, p_sw_pred = stats.shapiro(true_pred_test_df['pred'])
    test_df = pd.DataFrame({
        'test_loss': [test_loss],
        'test_mae': [test_mae],
        'r_true_pred': [r_true_pred],
        'p_true_pred': [p_true_pred],
        'true_mean': [true_pred_test_df['true'].describe()['mean']],
        'true_sd': [true_pred_test_df['true'].describe()['std']],
        'true_skewness': [stats.skew(true_pred_test_df['true'])],
        'true_kurtosis': [stats.kurtosis(true_pred_test_df['true'])],
        'true_shapiro-wilk_stat': [stat_sw_true],
        'true_shapiro-wilk_p': [p_sw_true],
        'pred_mean': [true_pred_test_df['pred'].describe()['mean']],
        'pred_sd': [true_pred_test_df['pred'].describe()['std']],
        'pred_skewness': [stats.skew(true_pred_test_df['pred'])],
        'pred_kurtosis': [stats.kurtosis(true_pred_test_df['pred'])],
        'pred_shapiro-wilk_stat': [stat_sw_pred],
        'pred_shapiro-wilk_p': [p_sw_pred]
    })
    test_df_path = os.path.join(workspace_path, f'test_results_{client_name}.csv')
    test_df.to_csv(test_df_path, index=False)
    send_file(server_ip_address, server_username, server_password, test_df_path)
