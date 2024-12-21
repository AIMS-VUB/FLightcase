"""
Script for clients of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9

Info:
- Run client scripts first, then server script
"""

import os
import json
import numpy as np
import torch
import argparse
import paramiko
import pandas as pd
import torch.nn as nn
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils.deep_learning.data import get_data_loader, split_data
from utils.deep_learning.model import get_weights, get_weighted_average_model, import_net_architecture, copy_net
from utils.deep_learning.evaluation import evaluate
from utils.communication import wait_for_file, send_file, clean_up_workspace
from train import train

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


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
    workspace_path_client = settings_dict.get('workspace_path_client')  # Path to client workspace
    client_name = settings_dict.get('client_name')                      # Client name
    server_ip_address = settings_dict.get('server_ip_address')          # Server ip address
    server_username = settings_dict.get('server_username')              # Server username
    server_password = settings_dict.get('server_password')              # Server password
    workspace_path_server = settings_dict.get('workspace_path_server')  # Path to server workspace
    colname_id = settings_dict.get('colname_id')                        # Column name of the BIDS id column
    colname_img_path = settings_dict.get('colname_img_path')            # Column name of the image paths
    colname_label = settings_dict.get('colname_label')                  # Column name of the label column
    subject_ids = settings_dict.get('subject_ids')                      # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')                # Path to BIDS root

    # Load dataframe and preprocess
    df_path = os.path.join(bids_root_path, 'participants.tsv')
    df = pd.read_csv(df_path, sep='\t')

    if colname_img_path is None:
        colname_img_path = 'img_path'
        # For now, only use the first session
        ses = 'ses-01'
        df[colname_img_path] = df[colname_id].apply(
            lambda x: os.path.join(bids_root_path, 'derivatives', 'Wood_2022', str(x), 'anat', ses,
                                   f'{x}_{ses}_T1w.nii.gz'))

    colnames_dict = {'id': colname_id, 'img_path': colname_img_path, 'label': colname_label}

    if subject_ids is not None:
        df = df[df[colname_id].isin(subject_ids)].reset_index(drop=True)

    # Create workspace folder
    if not os.path.exists(workspace_path_client):
        os.makedirs(workspace_path_client)

    # Save filtered clinical dataframe to workspace path as reference
    df.to_csv(os.path.join(workspace_path_client, 'participants.tsv'), sep='\t')

    # Send dataset size to server
    print('==> Send dataset size to server...')
    dataset_size = df.shape[0]
    dataset_size_txt_path = os.path.join(workspace_path_client, f'{client_name}_dataset_size.txt')
    with open(dataset_size_txt_path, 'w') as file:
        file.write(str(dataset_size))

    send_file(server_ip_address, server_username, server_password, dataset_size_txt_path, workspace_path_client,
              workspace_path_server)

    # Send client workspace path to server
    print('==> Send workspace path to server...')
    ws_path_txt_path = os.path.join(workspace_path_client, f'{client_name}_ws_path.txt')
    with open(ws_path_txt_path, 'w') as file:
        file.write(workspace_path_client)

    send_file(server_ip_address, server_username, server_password, ws_path_txt_path, workspace_path_client,
              workspace_path_server)

    # Wait for FL plan
    print('==> Waiting for FL plan...')
    FL_plan_path = os.path.join(workspace_path_client, 'FL_plan.json')
    wait_for_file(FL_plan_path.replace('.json', '_transfer_completed.txt'))

    # Wait for network architecture
    print('==> Waiting for network architecture...')
    architecture_path = os.path.join(workspace_path_client, 'architecture.py')
    wait_for_file(architecture_path.replace('.py', '_transfer_completed.txt'))

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

    # General deep learning settings
    criterion = get_criterion(criterion_txt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_architecture = import_net_architecture(architecture_path)
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
            global_model_path = os.path.join(workspace_path_client, f'initial_model.pt')
            global_txt_path = os.path.join(workspace_path_client, f'initial_model_transfer_completed.txt')
        else:
            # Load model from previous round as starting point (hence fl_round - 1)
            global_model_path = os.path.join(workspace_path_client, f'global_model_round_{fl_round-1}.pt')
            global_txt_path = os.path.join(workspace_path_client,
                                           f'global_model_round_{fl_round-1}_transfer_completed.txt')

        stop_training = wait_for_file(global_txt_path, stop_with_stop_file=True)
        if stop_training:
            break

        # Load global network
        global_net = get_weights(net_architecture, global_model_path)

        # Deep learning settings per FL round
        optimizer = get_optimizer(optimizer_txt, global_net, lr)

        # Initiate variables
        model_error_dict = {}
        mean_val_loss = 0
        best_model = None
        random_states = range(n_splits*fl_round, n_splits*fl_round + n_splits)  # Assure random state is never repeated
        train_results_df = pd.DataFrame()
        for split_i, random_state in enumerate(random_states):
            print(f'==> Split {split_i}, random state {random_state}...')
            # Split data
            # Note: Fix train_test_random_state to assure test data is always the same
            train_df, val_df, test_df = split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction,
                                                   train_test_random_state=42, train_val_random_state=random_state)
            if fl_round == 1 and split_i == 0:  # fl_round starts from 1
                train_overall_df = pd.concat([train_df, val_df], ignore_index=True)
                train_overall_df.to_csv(os.path.join(workspace_path_client, 'train_overall_df.csv'), index=False)
                test_df.to_csv(os.path.join(workspace_path_client, 'test_df.csv'), index=False)
            train_loader, n_train = get_data_loader(train_df, 'train', colnames_dict, batch_size, return_n=True)
            val_loader, n_val = get_data_loader(val_df, 'validation', colnames_dict, batch_size, return_n=True)
            test_loader, n_test = get_data_loader(test_df, 'test', colnames_dict, batch_size, return_n=True)

            # Train
            print('==> Start training...')
            best_model, train_loss_list, val_loss_list = train(1, device, train_loader, val_loader, optimizer,
                                                               global_net, criterion, None, False,
                                                               None)

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
            model_error_dict.update({best_model: val_loss_list[0]})

        print('==> Send training results to server...')
        train_results_df_path = os.path.join(workspace_path_client, f'{client_name}_round_{fl_round}_train_results.csv')
        train_results_df.to_csv(train_results_df_path, index=False)
        send_file(server_ip_address, server_username, server_password, train_results_df_path, workspace_path_client,
                  workspace_path_server)

        # Get local model
        if n_splits > 1:
            local_model = get_weighted_average_model(model_error_dict)
        else:
            local_model = copy_net(best_model)
        local_model_path = os.path.join(workspace_path_client, f'{client_name}_round_{fl_round}_model.pt')
        torch.save(local_model.state_dict(), local_model_path)

        # Send to server
        print('==> Send model to server ...')
        send_file(server_ip_address, server_username, server_password, local_model_path, workspace_path_client,
                  workspace_path_server)

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
    final_model_path = os.path.join(workspace_path_client, 'final_model.pt')
    wait_for_file(final_model_path.replace('final_model.pt', 'final_model_transfer_completed.txt'))
    print('==> Testing final model...')
    global_net = get_weights(net_architecture, final_model_path)
    test_loss, true_labels_test, pred_labels_test = evaluate(global_net, test_loader, criterion, device, 'test')
    test_mae = mean_absolute_error(true_labels_test, pred_labels_test)
    r_true_pred, p_true_pred = stats.pearsonr(true_labels_test, pred_labels_test)

    # Save predictions and ground truth to workspace
    true_pred_test_df = pd.DataFrame({'true': true_labels_test, 'pred': pred_labels_test})
    true_pred_test_df.to_csv(os.path.join(workspace_path_client, f'true_pred_test_{client_name}.csv'))

    # Create scatterplot
    fig, ax = plt.subplots()
    ax.scatter(x=true_labels_test, y=pred_labels_test)
    ax.set_title(f'Test performance {client_name}:\nMAE: {test_mae:.2f}, r: {r_true_pred:.2f} (p: {p_true_pred:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    plt.savefig(os.path.join(workspace_path_client, f'scatterplot_true_pred_test_{client_name}.png'))

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
    test_df_path = os.path.join(workspace_path_client, f'{client_name}_test_results.csv')
    test_df.to_csv(test_df_path, index=False)
    send_file(server_ip_address, server_username, server_password, test_df_path, workspace_path_client,
              workspace_path_server)

    # Clean up workspace
    print('Cleaning up workspace...')
    clean_up_workspace(workspace_path_client, server_or_client='client')
