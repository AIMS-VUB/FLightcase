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
from utils.deep_learning.model import (get_weights, weighted_avg_local_models, get_n_random_pairs_from_dict,
                                       get_model_param_info, import_net_architecture)
from utils.communication import wait_for_file, send_file

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: torch.load with weights_only=True in future

# Suppress printing of paramiko info
# Source: https://stackoverflow.com/questions/340341/suppressing-output-of-paramiko-sshclient-class
logger = paramiko.util.logging.getLogger()
logger.setLevel(paramiko.util.logging.WARN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Server',
        description='Server for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    args = parser.parse_args()
    settings_path = args.settings_path

    print('\n\n==============================\nStarting federated learning :)\n==============================\n\n')

    # FL start time
    fl_start_time = dt.datetime.now()

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path_server = settings_dict.get('workspace_path_server')          # Path to server workspace
    initial_state_dict_path = settings_dict.get('initial_state_dict_path')      # Path to initial state dict
    client_credentials_dict = settings_dict.get('client_credentials')           # Client credentials dict
    FL_plan_path = os.path.join(workspace_path_server, 'FL_plan.json')
    architecture_path = os.path.join(workspace_path_server, 'architecture.py')

    # Wait for all clients to share their dataset size
    print('==> Collecting all client dataset sizes...')
    client_dataset_size_dict = {}
    for client_name in client_credentials_dict.keys():
        client_dataset_txt_path = os.path.join(workspace_path_server, f'{client_name}_dataset_size.txt')
        wait_for_file(client_dataset_txt_path.replace('.txt', '_transfer_completed.txt'))
        with open(client_dataset_txt_path, 'r') as file:
            n_client = int(file.read())
            client_dataset_size_dict.update({client_name: n_client})
            print(f'     ==> {client_name}: n = {n_client}')

    # Wait for all clients to share their workspace_path
    print('==> Collecting all client workspace paths...')
    client_workspace_path_dict = {}
    for client_name in client_credentials_dict.keys():
        client_ws_path_txt_path = os.path.join(workspace_path_server, f'{client_name}_ws_path.txt')
        wait_for_file(client_ws_path_txt_path.replace('.txt', '_transfer_completed.txt'))
        with open(client_ws_path_txt_path, 'r') as file:
            client_ws_path = file.read()
            client_workspace_path_dict.update({client_name: client_ws_path})

    # Send FL plan to all clients
    print(f'==> Sending FL plan to all clients...')
    for client_name, credentials in client_credentials_dict.items():
        print(f'    ==> Sending to {client_name} ...')
        client_ip_address = credentials.get('ip_address')
        send_file(client_ip_address, credentials.get('username'), credentials.get('password'),
                  FL_plan_path, workspace_path_server, client_workspace_path_dict.get(client_name))

    # Send network architecture to all clients
    print(f'==> Sending architecture to all clients...')
    for client_name, credentials in client_credentials_dict.items():
        print(f'    ==> Sending to {client_name} ...')
        client_ip_address = credentials.get('ip_address')
        send_file(client_ip_address, credentials.get('username'), credentials.get('password'),
                  architecture_path, workspace_path_server, client_workspace_path_dict.get(client_name))

    # Extract FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    n_rounds = int(FL_plan_dict.get('n_rounds'))                    # Number of FL rounds
    n_clients_set = FL_plan_dict.get('n_clients_set')               # Number of clients in set for averaging
    patience_stop = int(FL_plan_dict.get('pat_stop'))               # N fl rounds stagnating val loss before stopping
    print('\n========\nFL plan:\n========\n')
    for k, v in FL_plan_dict.items():
        print(f'- {k}: {v}')
    print('\n')

    # Load initial network and save
    net_architecture = import_net_architecture(architecture_path)
    global_net = get_weights(net_architecture, initial_state_dict_path)
    model_path = os.path.join(workspace_path_server, 'initial_model.pt')
    torch.save(global_net.state_dict(), model_path)

    # Model information: get, print and save
    get_model_param_info(global_net, os.path.join(workspace_path_server, 'parameters_info.txt'))

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf       # Reference validation loss
    counter_stop = 0            # Counter for FL stop
    best_model_path = None      # Best model path with lowest avg validation loss
    avg_val_loss_clients = []   # Average validation loss across clients

    # Start federated learning
    for fl_round in range(1, n_rounds + 1):  # Start counting from 1
        print(f'\n*****************\nRound {fl_round}\n*****************\n')
        round_start_time = dt.datetime.now()

        # Send global model to all clients
        for client_name, credentials in client_credentials_dict.items():
            print(f'==> Sending global model to {client_name}...')
            client_ip_address = credentials.get('ip_address')
            send_file(client_ip_address, credentials.get('username'), credentials.get('password'), model_path,
                      workspace_path_server, client_workspace_path_dict.get(client_name))
        print('==> Model shared with all clients. Waiting for updated client models...')
        txt_file_paths = [os.path.join(workspace_path_server, f'model_{client_name}_round_{fl_round}_transfer_completed.txt')
                          for client_name in client_credentials_dict.keys()]
        for txt_file_path in txt_file_paths:
            wait_for_file(txt_file_path)

        # Create new global model by combining local models
        print('==> Combining local model weights and saving...')
        local_model_paths_dict = {client_name: os.path.join(workspace_path_server, f'model_{client_name}_round_{fl_round}.pt')
                                  for client_name in client_credentials_dict.keys()}
        local_state_dicts_dict = {k: torch.load(v, map_location='cpu') for k, v in local_model_paths_dict.items()}
        if n_clients_set is not None:
            local_state_dicts_dict = get_n_random_pairs_from_dict(local_state_dicts_dict, n_clients_set, fl_round)
            print(f'    ==> Clients in sample (random seed = {fl_round}): {list(local_state_dicts_dict.keys())}')
        new_global_state_dict = weighted_avg_local_models(local_state_dicts_dict,
                                                          {k: client_dataset_size_dict.get(k)
                                                           for k in local_state_dicts_dict.keys()})
        model_path = os.path.join(workspace_path_server, f'global_model_round_{fl_round}.pt')  # Overwrite model_path
        torch.save(new_global_state_dict, model_path)

        # Calculate average validation loss
        val_loss_avg = 0
        print('==> Average validation loss tracking...')
        for client_name in client_credentials_dict.keys():
            wait_for_file(os.path.join(
                workspace_path_server, f'train_results_{client_name}_round_{fl_round}_transfer_completed.txt'
            ))
            filename = f'train_results_{client_name}_round_{fl_round}.csv'
            train_results_client_df = pd.read_csv(os.path.join(workspace_path_server, filename))
            val_loss_avg += train_results_client_df['val_loss'].mean() / len(client_credentials_dict)
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
                for client_name, credentials in client_credentials_dict.items():
                    print(f'==> Sending stop txt file to {client_name}...')
                    client_ip_address = credentials.get('ip_address')
                    send_file(client_ip_address, credentials.get('username'), credentials.get('password'),
                              stop_txt_file_path, workspace_path_server, client_workspace_path_dict.get(client_name))
                break
        print(f'     ==> lr stop counter: {counter_stop}')

        # Time tracking
        round_stop_time = dt.datetime.now()
        round_duration = round_stop_time - round_start_time
        ETA = (round_stop_time + round_duration * (n_rounds - fl_round - 1)).strftime('%Y/%m/%d, %H:%M:%S')
        print(f'Round time: {round_duration / 60} min || ETA: {ETA}')

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
    for client_name, credentials in client_credentials_dict.items():
        print(f'     ==> Sending to {client_name}')
        client_ip_address = credentials.get('ip_address')
        send_file(client_ip_address, credentials.get('username'), credentials.get('password'), final_model_path,
                  workspace_path_server, client_workspace_path_dict.get(client_name))

    # Calculate overall test MAE
    print('==> Calculate overall test MAE...')
    test_mae_overall = 0
    for client_name, n_client in client_dataset_size_dict.items():
        print(f'    ==> Wait for test results {client_name}...')
        test_results_txt_path = os.path.join(workspace_path_server, f'test_results_{client_name}_transfer_completed.txt')
        wait_for_file(test_results_txt_path)
        test_df_client = pd.read_csv(test_results_txt_path.replace('_transfer_completed.txt', '.csv'))
        test_mae_client = test_df_client['test_mae'].iloc[0]
        test_mae_overall += test_mae_client * n_client / sum(client_dataset_size_dict.values())
    with open(os.path.join(workspace_path_server, 'overall_test_mae.txt'), 'w') as txt_file:
        txt_file.write(f'Overall test MAE: {test_mae_overall}')

    # Print total FL duration
    fl_stop_time = dt.datetime.now()
    fl_duration = fl_stop_time - fl_start_time
    print(f'Total federated learning duration: {fl_duration/3600} hrs')
