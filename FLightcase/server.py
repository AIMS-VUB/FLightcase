"""
Script for server of the federated learning network

Inspiration federated learning workflow:
- FederatedAveraging algorithm in https://arxiv.org/pdf/1602.05629.pdf
- FL plan in https://iopscience.iop.org/article/10.1088/1361-6560/ac97d9

Info:
- Run client scripts first, then server script
"""

import os
import json
import torch
import argparse
import paramiko
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from FLightcase.utils.deep_learning.model import (get_weights, weighted_avg_local_models, get_n_random_pairs_from_dict,
                                       get_model_param_info, import_net_architecture, copy_net)
from FLightcase.utils.communication import (clean_up_workspace, upload_file, collect_client_info, get_rsa_key_pair,
                                            generate_aes_key, rsa_encrypt)
from FLightcase.utils.tracking import print_FL_plan, create_overall_loss_df, fl_duration_print_save
from FLightcase.utils.results import update_avg_val_loss, calculate_overall_test_mae

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: torch.load with weights_only=True in future

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


def server(settings_path):
    """
    Run the server

    :param settings_path: str, path to server settings JSON
    """

    print('\n\n==============================\nStarting federated learning :)\n==============================\n\n')

    # FL start time
    fl_start_time = dt.datetime.now()

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path_server = settings_dict.get('workspace_path_server')          # Path to server workspace
    initial_state_dict_path = settings_dict.get('initial_state_dict_path')      # Path to initial state dict
    client_info_dict = settings_dict.get('client_credentials')                  # Initialise client info dict
    moderator_url_dl = settings_dict.get('moderator_url_dl')                    # URL where to download from moderator
    moderator_url_ul = settings_dict.get('moderator_url_ul')                    # URL where to upload to moderator
    username_dl_ul = settings_dict.get('username_dl_ul')                        # Username for download and upload
    password_dl_ul = settings_dict.get('password_dl_ul')                        # Password for download and upload

    client_names = client_info_dict.keys()
    FL_plan_path = os.path.join(workspace_path_server, 'FL_plan.json')
    architecture_path = os.path.join(workspace_path_server, 'architecture.py')

    # Share public RSA key with client
    print('Sending public RSA key to clients...')
    public_rsa_key_pem_server, private_rsa_key = get_rsa_key_pair()
    public_rsa_key_server_path = os.path.join(workspace_path_server, f'public_rsa_key_server.txt')
    with open(public_rsa_key_server_path, 'wb') as f:
        f.write(public_rsa_key_pem_server)
    upload_file(moderator_url_ul, public_rsa_key_server_path, username_dl_ul, password_dl_ul)

    # Wait for all clients to share their decryption tools, workspace path and dataset size
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'public_rsa_key', '.txt', moderator_url_dl, username_dl_ul, password_dl_ul)
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'aes_key', '.txt', moderator_url_dl, username_dl_ul, password_dl_ul)
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'iv', '.txt', moderator_url_dl, username_dl_ul, password_dl_ul)
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'dataset_size', '.txt', moderator_url_dl, username_dl_ul, password_dl_ul, private_rsa_key)
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'ws_path', '.txt', moderator_url_dl, username_dl_ul, password_dl_ul, private_rsa_key)
    n_sum_clients = sum([dct['dataset_size'] for dct in client_info_dict.values()])

    # Create IV for AES and send to moderator
    print('Creating IV and sending to moderator...')
    for client_name in client_names:
        iv_server = os.urandom(16)
        iv_path = os.path.join(workspace_path_server, f'server_iv_for_{client_name}.txt')
        with open(iv_path, 'wb') as f:
            f.write(iv_server)
        upload_file(moderator_url_ul, iv_path, username_dl_ul, password_dl_ul)
        client_info_dict[client_name]['iv_server'] = iv_server

    # Create encrypted AES key and send to moderator
    print('Creating encrypted AES keys and sending to moderator...')
    for client_name in client_names:
        aes_key = generate_aes_key()
        aes_key_encrypted_client = rsa_encrypt(client_info_dict[client_name]['public_rsa_key'], aes_key)
        aes_key_encrypted_client_path = os.path.join(workspace_path_server, f'server_aes_key_for_{client_name}.txt')
        with open(aes_key_encrypted_client_path, 'wb') as f:
            f.write(aes_key_encrypted_client)
        client_info_dict[client_name]['aes_key_server'] = aes_key
        upload_file(moderator_url_ul, aes_key_encrypted_client_path, username_dl_ul, password_dl_ul)

    # Send to all clients: FL plan and network architecture
    # Note: Currently without encryption
    upload_file(moderator_url_ul, FL_plan_path, username_dl_ul, password_dl_ul)
    upload_file(moderator_url_ul, architecture_path, username_dl_ul, password_dl_ul)

    # Extract and print FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                    # Number of FL rounds
    n_clients_set = FL_plan_dict.get('n_clients_set')               # Number of clients in set for averaging
    patience_stop = int(FL_plan_dict.get('pat_stop'))               # N fl rounds stagnating val loss before stopping
    print_FL_plan(FL_plan_dict)

    # Load initial network and save
    net_architecture = import_net_architecture(architecture_path)
    if initial_state_dict_path is not None:
        global_net = get_weights(net_architecture, initial_state_dict_path)
    else:
        global_net = copy_net(net_architecture)
    model_path = os.path.join(workspace_path_server, 'initial_model.pt')
    torch.save(global_net.state_dict(), model_path)

    # Model information: get, print and save
    get_model_param_info(global_net, os.path.join(workspace_path_server, 'parameters_info.txt'))

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf       # Reference validation loss
    counter_stop = 0            # Counter for FL stop
    best_model_path = None      # Best model path with lowest avg validation loss
    avg_val_loss_clients = []   # Average validation loss across clients

    # Initialise string to log client models in aggregation per round
    aggregation_samples_log = ''

    # Start federated learning
    for fl_round in range(1, n_rounds + 1):  # Start counting from 1
        # Add round key to fill in client_info_dict
        [client_info_dict[cl].update({f'round_{fl_round}': {}}) for cl in client_names]

        print(f'\n*****************\nRound {fl_round}\n*****************\n')
        round_start_time = dt.datetime.now()

        # Send global model to all clients
        for client_name in client_names:
            # Copy the model path to share encrypted version with clients
            model_path_for_client = f'{model_path.removesuffix(".pt")}_for_{client_name}.pt'
            os.system(f'cp {model_path} {model_path_for_client}')
            upload_file(moderator_url_ul, model_path_for_client, username_dl_ul, password_dl_ul, client_info_dict[client_name]['aes_key_server'], client_info_dict[client_name]['iv_server'])
        print('==> Model shared with all clients. Waiting for updated client models...')
        client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'model', '.pt', moderator_url_dl, username_dl_ul, password_dl_ul, private_rsa_key, fl_round, net_architecture)

        # Create new global model by combining local models
        print('==> Combining local model weights and saving...')
        if n_clients_set is not None:
            client_info_dict_sample = get_n_random_pairs_from_dict(client_info_dict, n_clients_set, fl_round)
            log_txt = (f'    ==> Clients in sample (round = {fl_round}, random seed = {fl_round}):'
                       f'{list(client_info_dict_sample.keys())}')
        else:
            client_info_dict_sample = client_info_dict
            log_txt = f'    ==> All clients in sample (round = {fl_round})'
        print(log_txt)
        aggregation_samples_log += f'{log_txt}\n'
        new_global_state_dict = weighted_avg_local_models(client_info_dict_sample, fl_round)
        model_path = os.path.join(workspace_path_server, f'global_model_round_{fl_round}.pt')  # Overwrite model_path
        torch.save(new_global_state_dict, model_path)

        # Calculate average validation loss
        val_loss_avg = 0
        print('==> Average validation loss tracking...')
        client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'train_results',
                                               '.csv', moderator_url_dl, username_dl_ul, password_dl_ul, private_rsa_key, fl_round)
        val_loss_avg = update_avg_val_loss(client_info_dict_sample, val_loss_avg, fl_round)
        print(f'     ==> val loss ref: {val_loss_ref} || val loss avg: {val_loss_avg}')
        avg_val_loss_clients.append(val_loss_avg)

        # Perform actions based on average validation loss
        if val_loss_avg < val_loss_ref:         # Improvement
            val_loss_ref = val_loss_avg
            counter_stop = 0
            best_model_path = model_path
        else:                                   # No improvement
            counter_stop += 1
            if counter_stop == patience_stop:
                stop_txt_file_path = os.path.join(workspace_path_server, 'stop_training.txt')
                with open(stop_txt_file_path, 'w') as txt_file:
                    txt_file.write('This file causes early FL stopping')
                # Note: Send without encryption
                upload_file(moderator_url_ul, stop_txt_file_path, username_dl_ul, password_dl_ul)
                break
        print(f'     ==> lr stop counter: {counter_stop}')

        # Time tracking
        round_stop_time = dt.datetime.now()
        round_duration = round_stop_time - round_start_time
        ETA = (round_stop_time + round_duration * (n_rounds - fl_round - 1)).strftime('%Y/%m/%d, %H:%M:%S')
        print(f'Round time: {str(round_duration)} || ETA: {ETA}')

    # Combine all train/val loss results across clients and rounds in one dataframe
    overall_loss_df = create_overall_loss_df(workspace_path_server)
    overall_loss_df.to_csv(os.path.join(workspace_path_server, 'overall_loss_df.csv'))

    # Create dataframe with average validation loss across clients
    avg_val_loss_df = pd.DataFrame({'avg_val_loss_clients': avg_val_loss_clients,
                                    'fl_round': range(1, fl_round+1)})
    avg_val_loss_df.to_csv(os.path.join(workspace_path_server, 'avg_val_loss_clients.csv'), index = False)

    # Copy final model path and send to clients
    final_model_path = os.path.join(workspace_path_server, "final_model.pt")
    os.system(f'cp {best_model_path} {final_model_path}')
    print(f'==> Sending final model ({os.path.basename(best_model_path)}) to all clients...')
    with open(os.path.join(workspace_path_server, 'final_model.txt'), 'w') as txt_file:
        txt_file.write(best_model_path)
    for client_name in client_names:
        # Copy the model path to share encrypted version with clients
        final_model_path_for_client = os.path.join(workspace_path_server, f'final_model_for_{client_name}.pt')
        os.system(f'cp {final_model_path} {final_model_path_for_client}')
        upload_file(moderator_url_ul, final_model_path_for_client, username_dl_ul, password_dl_ul,
                    client_info_dict[client_name]['aes_key_server'], client_info_dict[client_name]['iv_server'])

    # Calculate overall test MAE
    print('==> Calculate overall test MAE...')
    client_info_dict = collect_client_info(client_info_dict, workspace_path_server, 'test_results', '.csv', moderator_url_dl, username_dl_ul, password_dl_ul, private_rsa_key)
    calculate_overall_test_mae(client_info_dict, workspace_path_server, save=True)

    # Save client sample log
    with open(os.path.join(workspace_path_server, 'aggregation_samples.txt'), 'w') as samples_log:
        samples_log.write(aggregation_samples_log)

    # Print and save total FL duration
    fl_stop_time = dt.datetime.now()
    fl_duration_print_save(fl_start_time, fl_stop_time, workspace_path_server)

    # Send final message to moderator that allowed to clean up workspace entirely
    moderator_clean_ws_file = os.path.join(workspace_path_server, 'moderator_clean_ws.txt')
    with open(moderator_clean_ws_file, 'w') as samples_log:
        samples_log.write('Prompt to clean workspace')
    # No encryption
    upload_file(moderator_url_ul, moderator_clean_ws_file, username_dl_ul, password_dl_ul)

    # Clean up workspace
    print('Cleaning up workspace...')
    clean_up_workspace(workspace_path_server, who='server')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Server',
        description='Server for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    args = parser.parse_args()

    # Run server
    server(args.settings_path)
