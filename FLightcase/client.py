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
from FLightcase.utils.deep_learning.data import get_data_loader, split_data, prepare_participants_df
from FLightcase.utils.deep_learning.model import get_weights, get_weighted_average_model, import_net_architecture, copy_net
from FLightcase.utils.deep_learning.evaluation import evaluate
from FLightcase.utils.deep_learning.general import get_device
from FLightcase.utils.communication import (wait_for_file, send_file, clean_up_workspace, send_client_info_to_server,
                                 send_test_df_to_server)
from FLightcase.utils.results import create_test_true_pred_df, create_test_scatterplot, create_test_df_for_server
from FLightcase.utils.deep_learning.train import train, get_criterion, get_optimizer

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


def client(settings_path):
    """
    Run the client

    :param settings_path: str, path to client settings JSON
    """
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
    derivative_name = settings_dict.get('derivative_name')              # Name of derivative subfolder, else None
    modalities_dict = settings_dict.get('modalities_to_include')        # Modalities (e.g. {'anat': ['T1w', 'FLAIR']})
    colnames_dict = settings_dict.get('colnames_dict')                  # Colnames dict
    subject_sessions = settings_dict.get('subject_sessions')            # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')                # Path to BIDS root
    batch_size = int(settings_dict.get('batch_size'))                   # Batch size
    device = get_device(settings_dict.get('device'))                    # Device for DL process (cpu, cuda, cuda:0, ...)

    # Create workspace folder
    if not os.path.exists(workspace_path_client):
        os.makedirs(workspace_path_client)

    # Preprocess participants dataframe + save to workspace path as reference
    df, colnames_dict = prepare_participants_df(bids_root_path, colnames_dict, subject_sessions,
                                                modalities_dict, derivative_name)
    df.to_csv(os.path.join(workspace_path_client, 'participants.tsv'), sep='\t')

    # Send dataset size and client workspace path to server
    send_client_info_to_server(df.shape[0], workspace_path_client, client_name, server_ip_address, server_username,
                               server_password, workspace_path_server)

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
    train_fraction = float(FL_plan_dict.get('train_fraction'))      # Fraction of data for training
    val_fraction = float(FL_plan_dict.get('val_fraction'))          # Fraction of data for validation
    test_fraction = float(FL_plan_dict.get('test_fraction'))        # Fraction of data for testing
    n_splits = int(FL_plan_dict.get('n_splits'))                    # Number of data splits per fl round
    lr = float(FL_plan_dict.get('lr'))                              # Learning rate
    lr_reduce_factor = float(FL_plan_dict.get('lr_reduce_factor'))  # Factor by which to reduce LR on Plateau
    patience_lr_reduction = int(FL_plan_dict.get('pat_lr_red'))     # N fl rounds stagnating val loss before reducing lr
    criterion_txt = FL_plan_dict.get('criterion')                   # Criterion in txt format, lowercase (e.g. l1loss)
    optimizer_txt = FL_plan_dict.get('optimizer')                   # Optimizer in txt format, lowercase (e.g. adam)
    n_epochs_per_round = FL_plan_dict.get('n_epochs_per_round')     # Number of epochs per FL round

    # General deep learning settings
    criterion = get_criterion(criterion_txt)
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
                train_overall_df.to_csv(os.path.join(workspace_path_client, 'train_overall_df.tsv'), sep='\t',
                                        index=False)
                test_df.to_csv(os.path.join(workspace_path_client, 'test_df.tsv'), sep='\t', index=False)
            train_loader, n_train = get_data_loader(train_df, 'train', colnames_dict, batch_size, return_n=True)
            val_loader, n_val = get_data_loader(val_df, 'validation', colnames_dict, batch_size, return_n=True)
            test_loader, n_test = get_data_loader(test_df, 'test', colnames_dict, batch_size, return_n=True)

            # Train
            print('==> Start training...')
            best_model, best_val_loss, train_loss_list, val_loss_list = train(
                n_epochs_per_round, device, train_loader, val_loader, optimizer, global_net, criterion, None,
                False, None
            )

            train_results_df_i = pd.DataFrame({'random_state': [random_state]*len(train_loss_list),
                                               'fl_round': [fl_round]*len(train_loss_list),
                                               'train_loss': train_loss_list,
                                               'val_loss': val_loss_list,
                                               'n_train': [n_train]*len(train_loss_list),
                                               'n_val': [n_val]*len(train_loss_list),
                                               'n_test': [n_test]*len(train_loss_list)})
            train_results_df = pd.concat([train_results_df, train_results_df_i], axis=0)

            # Update mean val loss
            mean_val_loss += best_val_loss/n_splits

            # Update dict
            model_error_dict.update({best_model: best_val_loss})

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

    # Test result analysis
    true_pred_test_df = create_test_true_pred_df(true_labels_test, pred_labels_test, workspace_path_client, save=True)
    create_test_scatterplot(true_pred_test_df, client_name, workspace_path_client)
    test_df_for_server = create_test_df_for_server(true_pred_test_df, test_loss)
    send_test_df_to_server(test_df_for_server, client_name, workspace_path_client, server_username,
                           server_password, server_ip_address, workspace_path_server)

    # Clean up workspace
    print('Cleaning up workspace...')
    clean_up_workspace(workspace_path_client, who='client')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Client',
        description='Client for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    args = parser.parse_args()
    client(args.settings_path)
