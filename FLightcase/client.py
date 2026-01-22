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
from FLightcase.utils.communication import (wait_for_file, upload_file, clean_up_workspace, send_client_n_to_moderator,
                                            send_test_df_to_moderator, get_rsa_key_pair, generate_aes_key, rsa_encrypt,
                                            receive_public_key)
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
    print(settings_dict)
    workspace_path_client = settings_dict.get('workspace_path_client')  # Path to client workspace
    client_name = settings_dict.get('client_name')                      # Client name
    username_dl_ul = settings_dict.get('username_dl_ul')                # Username for download and upload
    password_dl_ul = settings_dict.get('password_dl_ul')                # Password for download and upload
    moderator_url_dl = settings_dict.get('moderator_url_dl')            # URL where to download from moderator
    moderator_url_ul = settings_dict.get('moderator_url_ul')            # URL where to upload to moderator
    derivative_name = settings_dict.get('derivative_name')              # Name of derivative subfolder, else None
    colnames_dict = settings_dict.get('colnames_dict')                  # Colnames dict
    subject_sessions = settings_dict.get('subject_sessions')            # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')                # Path to BIDS root
    batch_size = int(settings_dict.get('batch_size'))                   # Batch size
    preferred_device = settings_dict.get('preferred_device')            # Preferred device

    # Create workspace folder
    if not os.path.exists(workspace_path_client):
        os.makedirs(workspace_path_client)

    # Create RSA keys and send public to moderator
    print('Creating RSA keys and sending public to moderator...')
    public_rsa_key_pem_client, private_rsa_key = get_rsa_key_pair()
    public_rsa_key_client_path = os.path.join(workspace_path_client, f'{client_name}_public_rsa_key.txt')
    with open(public_rsa_key_client_path, 'wb') as f:
        f.write(public_rsa_key_pem_client)
    upload_file(moderator_url_ul, public_rsa_key_client_path, username_dl_ul, password_dl_ul)

    # Create IV for AES and send to moderator
    print('Creating IV and sending to moderator...')
    iv = os.urandom(16)
    iv_path_client = os.path.join(workspace_path_client, f'{client_name}_iv.txt')
    with open(iv_path_client, 'wb') as f:
        f.write(iv)
    upload_file(moderator_url_ul, iv_path_client, username_dl_ul, password_dl_ul)

    # Wait for public RSA key from server
    print('Waiting for public RSA key from server...')
    public_rsa_key_server_path = os.path.join(workspace_path_client, f'public_rsa_key_server.txt')
    wait_for_file(public_rsa_key_server_path, moderator_url_dl, username_dl_ul, password_dl_ul)
    with open(public_rsa_key_server_path, 'rb') as f:
        public_rsa_key_server = f.read()
        public_rsa_key_server = receive_public_key(public_rsa_key_server)

    # Create encrypted AES key and send to moderator
    print('Creating encrypted AES key and sending to moderator...')
    aes_key = generate_aes_key()
    aes_key_encrypted_client = rsa_encrypt(public_rsa_key_server, aes_key)
    aes_key_encrypted_client_path = os.path.join(workspace_path_client, f'{client_name}_aes_key.txt')
    with open(aes_key_encrypted_client_path, 'wb') as f:
        f.write(aes_key_encrypted_client)
    upload_file(moderator_url_ul, aes_key_encrypted_client_path, username_dl_ul, password_dl_ul)

    # Wait for IV and AES key from server
    print('Waiting for IV and AES key from server...')
    iv_path = os.path.join(workspace_path_client, f'server_iv_for_{client_name}.txt')
    wait_for_file(iv_path, moderator_url_dl, username_dl_ul, password_dl_ul)
    with open(iv_path, 'rb') as f:
        iv_server = f.read()
    aes_key_path = os.path.join(workspace_path_client, f'server_aes_key_for_{client_name}.txt')
    wait_for_file(aes_key_path, moderator_url_dl, username_dl_ul, password_dl_ul)
    with open(aes_key_path, 'rb') as f:
        aes_key_encrypted_server = f.read()

    # Wait for FL plan
    print('==> Waiting for FL plan...')
    FL_plan_path = os.path.join(workspace_path_client, 'FL_plan.json')
    wait_for_file(FL_plan_path, moderator_url_dl, username_dl_ul, password_dl_ul)

    # Wait for network architecture
    print('==> Waiting for network architecture...')
    architecture_path = os.path.join(workspace_path_client, 'architecture.py')
    wait_for_file(architecture_path, moderator_url_dl, username_dl_ul, password_dl_ul)

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
    modalities_dict = FL_plan_dict.get('modalities_to_include')     # Modalities (e.g. {'anat': ['T1w', 'FLAIR']})

    # Preprocess participants dataframe + save to workspace path as reference
    df, colnames_dict = prepare_participants_df(bids_root_path, colnames_dict, subject_sessions,
                                                modalities_dict, derivative_name)
    df.to_csv(os.path.join(workspace_path_client, 'participants.tsv'), sep='\t')

    # Send dataset size to server
    send_client_n_to_moderator(df.shape[0], workspace_path_client, client_name, moderator_url_ul, username_dl_ul, password_dl_ul, aes_key, iv)

    # General deep learning settings
    if preferred_device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(preferred_device)
    else:
        device = torch.device('cpu')
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
            global_model_path = os.path.join(workspace_path_client, f'initial_model_for_{client_name}.pt')
        else:
            # Load model from previous round as starting point (hence fl_round - 1)
            global_model_path = os.path.join(workspace_path_client, f'global_model_round_{fl_round-1}_for_{client_name}.pt')

        stop_training = wait_for_file(global_model_path, moderator_url_dl, username_dl_ul, password_dl_ul, aes_key_encrypted_server, iv_server, private_rsa_key, stop_with_stop_file=True)
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
        upload_file(moderator_url_ul, train_results_df_path, username_dl_ul, password_dl_ul, aes_key, iv)

        # Get local model
        if n_splits > 1:
            local_model = get_weighted_average_model(model_error_dict)
        else:
            local_model = copy_net(best_model)
        local_model_path = os.path.join(workspace_path_client, f'{client_name}_round_{fl_round}_model.pt')
        torch.save(local_model.state_dict(), local_model_path)

        # Send to server
        print('==> Send model to server ...')
        upload_file(moderator_url_ul, local_model_path, username_dl_ul, password_dl_ul, aes_key, iv)

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
    final_model_path = os.path.join(workspace_path_client, f'final_model_for_{client_name}.pt')
    wait_for_file(final_model_path, moderator_url_dl, username_dl_ul, password_dl_ul, aes_key_encrypted_server, iv_server, private_rsa_key)
    print('==> Testing final model...')
    global_net = get_weights(net_architecture, final_model_path)
    test_loss, true_labels_test, pred_labels_test = evaluate(global_net, test_loader, criterion, device, 'test')

    # Test result analysis
    true_pred_test_df = create_test_true_pred_df(true_labels_test, pred_labels_test, workspace_path_client, save=True)
    create_test_scatterplot(true_pred_test_df, client_name, workspace_path_client)
    test_df_for_server = create_test_df_for_server(true_pred_test_df, test_loss)
    send_test_df_to_moderator(test_df_for_server, client_name, workspace_path_client, moderator_url_ul, username_dl_ul, password_dl_ul, aes_key, iv)

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
