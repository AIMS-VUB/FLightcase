"""
Client-specific training

FL plan inspiration: https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9
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
from monai.networks.nets import DenseNet
from DL_utils.data import get_data_loader, split_data
from DL_utils.model import get_weights
from DL_utils.evaluation import evaluate
from train import train
from client import send_file, get_net_weighted_average_fc, wait_for_file, prepare_for_transfer_learning

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


def client(settings_path, clients_to_test):
    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path = settings_dict.get('workspace_path')        # Path to workspace for server and clients
    client_name = settings_dict.get('client_name')              # Client name
    server_ip_address = settings_dict.get('server_ip_address')  # Server ip address
    server_username = settings_dict.get('server_username')      # Server username
    server_password = settings_dict.get('server_password')      # Server password
    colname_id = settings_dict.get('colname_id')                # Column name of the BIDS id column
    colname_img_path = settings_dict.get('colname_img_path')    # Column name of the image paths
    colname_label = settings_dict.get('colname_label')          # Column name of the label column
    subject_ids = settings_dict.get('subject_ids')              # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')        # Path to BIDS root

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

    # Create subfolder in FL workspace and copy FL plan and initial model to this subfolder
    workspace_path_client_specific = os.path.join(workspace_path, 'client-specific')
    if not os.path.exists(workspace_path_client_specific):
        os.mkdir(workspace_path_client_specific)
    for file in ['initial_model.pt', 'FL_plan.json']:
        os.system(f'cp {os.path.join(workspace_path, file)} {os.path.join(workspace_path_client_specific, file)}')

    # Extract FL plan
    FL_plan_path = os.path.join(workspace_path_client_specific, 'FL_plan.json')
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                    # Number of rounds
    batch_size = int(FL_plan_dict.get('batch_size'))                # Batch size for the data loaders
    train_fraction = float(FL_plan_dict.get('train_fraction'))      # Fraction of data for training
    val_fraction = float(FL_plan_dict.get('val_fraction'))          # Fraction of data for validation
    test_fraction = float(FL_plan_dict.get('test_fraction'))        # Fraction of data for testing
    n_splits = int(FL_plan_dict.get('n_splits'))                    # Number of data splits per round
    lr = float(FL_plan_dict.get('lr'))                              # Learning rate
    lr_reduce_factor = float(FL_plan_dict.get('lr_reduce_factor'))  # Factor by which to reduce LR on Plateau
    patience_lr_reduction = int(FL_plan_dict.get('pat_lr_red'))     # N rounds stagnating val loss before reducing lr
    patience_stop = int(FL_plan_dict.get('pat_stop'))               # N rounds stagnating val loss before stopping

    # Send dataset size to server
    print('==> Send dataset size to server...')
    dataset_size = df.shape[0]
    dataset_size_txt_path = os.path.join(workspace_path_client_specific, f'{client_name}_dataset_size.txt')
    with open(dataset_size_txt_path, 'w') as file:
        file.write(str(dataset_size))

    send_file(server_ip_address, server_username, server_password, dataset_size_txt_path)

    # General deep learning settings
    criterion = nn.L1Loss(reduction='sum')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_architecture = DenseNet(3, 1, 1)
    test_loader = None

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf   # Reference validation loss
    counter_lr_red = 0      # Counter for lr reduction
    counter_stop = 0        # Counter for stopping training

    # Initialise model paths
    best_model_path = None
    model_path = os.path.join(workspace_path_client_specific, f'initial_model.pt')

    # Start client-specific learning
    for round_i in range(1, n_rounds + 1):  # Start counting from 1
        print(f'\n*****************\nRound {round_i}\n*****************\n')

        # Create a state dict folder in the workspace for this training round
        print('==> Make preparations to start training...')
        state_dict_folder_path = os.path.join(workspace_path_client_specific,
                                              f'state_dicts_{client_name}_round_{round_i}')
        if not os.path.exists(state_dict_folder_path):
            os.mkdir(state_dict_folder_path)

        # Load global network and prepare for transfer learning
        global_net = get_weights(net_architecture, model_path)
        global_net = prepare_for_transfer_learning(global_net)

        # Deep learning settings per FL round
        optimizer = torch.optim.Adam(global_net.parameters(), lr=lr)

        # Initiate variables
        path_error_dict = {}
        mean_val_loss = 0
        random_states = range(n_splits * round_i,
                              n_splits * round_i + n_splits)  # Assure random state is never repeated
        train_results_df = pd.DataFrame()
        for split_i, random_state in enumerate(random_states):
            print(f'==> Split {split_i}, random state {random_state}...')
            # Split data
            # Note: Fix train_test_random_state to assure test data is always the same
            train_df, val_df, test_df = split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction,
                                                   train_test_random_state=42, train_val_random_state=random_state)
            train_loader, n_train = get_data_loader(train_df, 'train', colnames_dict, batch_size, return_n=True)
            val_loader, n_val = get_data_loader(val_df, 'validation', colnames_dict, batch_size, return_n=True)
            test_loader, n_test = get_data_loader(val_df, 'test', colnames_dict, batch_size, return_n=True)

            # Train
            print('==> Start training...')
            split_model_path, train_loss_list, val_loss_list = train(1, device, train_loader, val_loader, optimizer,
                                                                     global_net, criterion, None, state_dict_folder_path)

            train_results_df_i = pd.DataFrame({'random_state': [random_state],
                                               'round': [round_i],
                                               'train_loss': train_loss_list,
                                               'val_loss': val_loss_list,
                                               'n_train': n_train,
                                               'n_val': n_val,
                                               'n_test': n_test})
            train_results_df = pd.concat([train_results_df, train_results_df_i], axis=0)

            # Update mean val loss
            mean_val_loss += val_loss_list[0] / n_splits

            # Update dict
            path_error_dict.update({split_model_path: val_loss_list[0]})

        print('==> Send training results to server...')
        train_results_df_path = os.path.join(workspace_path_client_specific,
                                             f'train_results_{client_name}_round_{round_i}.csv')
        train_results_df.to_csv(train_results_df_path, index=False)
        send_file(server_ip_address, server_username, server_password, train_results_df_path)

        # Get weighted average fc model
        model = get_net_weighted_average_fc(net_architecture, path_error_dict)
        model_path = os.path.join(workspace_path_client_specific, f'model_{client_name}_round_{round_i}.pt')
        torch.save(model.state_dict(), model_path)

        # Perform actions based on min validation loss across splits and epochs
        print('==> Validation loss tracking...')
        if mean_val_loss < val_loss_ref:  # Improvement
            val_loss_ref = mean_val_loss
            best_model_path = model_path
            counter_lr_red = 0
            counter_stop = 0
        else:  # No improvement
            counter_lr_red += 1
            counter_stop += 1
            if counter_lr_red == patience_lr_reduction:
                lr *= lr_reduce_factor
                FL_plan_dict['lr'] = lr
                counter_lr_red = 0
            if counter_stop == patience_stop:
                break
        print(f'     ==> val loss ref: {val_loss_ref} || mean val loss: {mean_val_loss}')
        print(f'     ==> lr reduction counter: {counter_lr_red}')
        print(f'     ==> stop counter: {counter_stop}')

    # Copy final model and send to server
    final_model_path = os.path.join(workspace_path_client_specific, f"final_client-specific_model_{client_name}.pt")
    os.system(f'cp {best_model_path} {final_model_path}')
    send_file(server_ip_address, server_username, server_password, final_model_path)

    # Send best model path in txt file
    final_model_txt_path = os.path.join(workspace_path_client_specific, f'final_client-specific_model_{client_name}.txt')
    with open(final_model_txt_path, 'w') as file:
        file.write(f'Final model {client_name}: {best_model_path}')
    send_file(server_ip_address, server_username, server_password, final_model_txt_path)

    test_df = pd.DataFrame()
    for client_to_test in clients_to_test:
        client_model_path = os.path.join(workspace_path_client_specific,
                                         f'final_client-specific_model_{client_to_test}.pt')
        # Only wait for model if expecting model from another client
        if client_to_test != client_name:
            print(f'==> Waiting for {client_to_test} model...')
            wait_for_file(client_model_path.replace('.pt', '_transfer_completed.txt'))
        print(f'==> Testing final model ({client_to_test})...')
        client_model = get_weights(net_architecture, client_model_path)
        test_loss, true_labels_test, pred_labels_test = evaluate(client_model, test_loader, criterion, device, 'test')
        test_mae = mean_absolute_error(true_labels_test, pred_labels_test)
        r_true_pred, p_true_pred = stats.pearsonr(true_labels_test, pred_labels_test)

        # Save predictions and ground truth to client-specific workspace
        true_pred_test_df = pd.DataFrame({'true': true_labels_test, 'pred': pred_labels_test})
        true_pred_test_df.to_csv(os.path.join(workspace_path_client_specific, f'true_pred_test_model_{client_to_test}_data_{client_name}.csv'))

        # Create scatterplot
        fig, ax = plt.subplots()
        ax.scatter(x=true_labels_test, y=pred_labels_test)
        ax.set_title(
            f'Test performance {client_name}:\nMAE: {test_mae:.2f}, r: {r_true_pred:.2f} (p: {p_true_pred:.3f})')
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        plt.savefig(os.path.join(workspace_path_client_specific,
                                 f'scatterplot_true_pred_test_model_{client_to_test}_data_{client_name}.png'))

        # Fill dataframe with results for server
        stat_sw_true, p_sw_true = stats.shapiro(true_pred_test_df['true'])
        stat_sw_pred, p_sw_pred = stats.shapiro(true_pred_test_df['pred'])
        row = pd.DataFrame({
            'client': [client_to_test],
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
        test_df = pd.concat([test_df, row], axis=0)

    # Send results to server
    print('==> Sending test results to server...')
    test_df_path = os.path.join(workspace_path_client_specific, f'test_results_{client_name}.csv')
    test_df.to_csv(test_df_path, index=False)
    send_file(server_ip_address, server_username, server_password, test_df_path)


def server(settings_path):
    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path = settings_dict.get('workspace_path')                    # Path to workspace for server and clients
    client_credentials_dict = settings_dict.get('client_credentials')       # Client credentials dict

    # Create subfolder in FL workspace and copy FL plan and initial model to this subfolder
    workspace_path_client_specific = os.path.join(workspace_path, 'client-specific')
    if not os.path.exists(workspace_path_client_specific):
        os.mkdir(workspace_path_client_specific)
    for file in ['initial_model.pt', 'FL_plan.json']:
        os.system(f'cp {os.path.join(workspace_path, file)} {os.path.join(workspace_path_client_specific, file)}')

    for client_name in client_credentials_dict.keys():
        # Wait for client-specific model
        print(f'==> Wait for client-specific model: {client_name}')
        client_model_path = os.path.join(workspace_path_client_specific,
                                         f'final_client-specific_model_{client_name}.pt')
        wait_for_file(client_model_path.replace('.pt', '_transfer_completed.txt'))

        # Send to other clients
        other_client_names = list(client_credentials_dict.keys())
        other_client_names.remove(client_name)

        for other_client_name in other_client_names:
            print(f'    ==> Send model client {client_name} to {other_client_name}')
            other_client_ip_address = client_credentials_dict.get(other_client_name).get('ip_address')
            other_client_username = client_credentials_dict.get(other_client_name).get('username')
            other_client_password = client_credentials_dict.get(other_client_name).get('password')
            send_file(other_client_ip_address, other_client_username, other_client_password, client_model_path)

    # Get overall test MAE per client model
    print('==> Get overall test MAE per client model...')
    # Get dataset sizes
    client_dataset_size_dict = {}
    for client_name in client_credentials_dict.keys():
        # Get client size
        with open(os.path.join(workspace_path_client_specific, f'{client_name}_dataset_size.txt'), 'r') as txt_file:
            n_client = int(txt_file.read())
        client_dataset_size_dict.update({client_name: n_client})

    # Create dictionary with overall test MAE per client model and save to dataframe
    overall_test_mae_dict = {}
    for client_name in client_credentials_dict.keys():
        overall_test_mae = 0
        print(f'    ==> Wait for test results {client_name}...')
        test_results_txt_path = os.path.join(workspace_path_client_specific, f'test_results_{client_name}_transfer_completed.txt')
        wait_for_file(test_results_txt_path)
        client_test_df = pd.read_csv(test_results_txt_path.replace('_transfer_completed.txt', '.csv'))
        for i, row in client_test_df.iterrows():
            overall_test_mae += (client_dataset_size_dict.get(row['client']) * row['test_mae']) / sum(client_dataset_size_dict.values())
        overall_test_mae_dict.update({f'{client_name}_model': overall_test_mae})
    overall_test_mae_df = pd.DataFrame(overall_test_mae_dict, index=['overall_test_mae'])
    overall_test_mae_df.to_csv(os.path.join(workspace_path_client_specific, 'overall_test_mae_results.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    parser.add_argument('--client_or_server', type=str, help='Choose either "client" or "server"')
    parser.add_argument('--clients_to_test', nargs='+', required=False,
                        help='Client argument, name of centers to expect a model from (list all client centers)')
    args = parser.parse_args()

    print('\n\n====================================\n'
          'Starting client-specific learning :)\n'
          '====================================\n\n')

    if args.client_or_server == 'server':
        server(args.settings_path)
    elif args.client_or_server == 'client':
        client(args.settings_path, args.clients_to_test)
