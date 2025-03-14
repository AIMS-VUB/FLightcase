"""
This is a script for centralised training of a model
"""

import os
import sys
import json
import torch
import pathlib
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# Add path to great-great-grandparent dir of this Python file: https://stackoverflow.com/questions/3430372/
path_to_toolbox_manuscript_folder = str(pathlib.Path(__file__).parent.resolve())
path_to_root_FLightcase_folder = os.sep + os.sep.join(path_to_toolbox_manuscript_folder.split(os.sep)[1:-3]) + os.sep  # Start from 1 as splits on first sep
sys.path.append(path_to_root_FLightcase_folder)
from FLightcase.utils.deep_learning.data import get_data_loader, prepare_participants_df
from FLightcase.utils.deep_learning.model import get_weights, import_net_architecture, copy_net, get_weighted_average_model
from FLightcase.utils.deep_learning.evaluation import evaluate
from FLightcase.utils.deep_learning.general import get_device
from FLightcase.utils.communication import clean_up_workspace
from FLightcase.utils.results import create_test_true_pred_df, create_test_scatterplot
from FLightcase.utils.deep_learning.train import train, get_criterion, get_optimizer
from FLightcase.utils.tracking import fl_duration_print_save


def train_centralised(settings_path):
    """
    Run the client

    :param settings_path: str, path to client settings JSON
    """

    print('\n\n==============================\nStarting centralised training :)\n==============================\n\n')

    # Start time
    start_time = dt.datetime.now()

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    workspace_path = settings_dict.get('workspace_path')                # Path to workspace
    initial_sd_path = settings_dict.get('initial_state_dict_path')      # Path to initial state dict
    derivative_name = settings_dict.get('derivative_name')              # Name of derivative subfolder, else None
    modalities_dict = settings_dict.get('modalities_to_include')        # Modalities (e.g. {'anat': ['T1w', 'FLAIR']})
    colnames_dict = settings_dict.get('colnames_dict')                  # Colnames dict
    subject_sessions = settings_dict.get('subject_sessions')            # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')                # Path to BIDS root
    batch_size = int(settings_dict.get('batch_size'))                   # Batch size
    device = get_device(settings_dict.get('device'))                    # Device for DL process (cpu, cuda, cuda:0, ...)
    train_fraction = float(settings_dict.get('train_fraction'))         # Fraction of data for training
    val_fraction = float(settings_dict.get('val_fraction'))             # Fraction of data for validation
    test_fraction = float(settings_dict.get('test_fraction'))           # Fraction of data for testing
    lr = float(settings_dict.get('lr'))                                 # Learning rate
    lr_reduce_factor = float(settings_dict.get('lr_reduce_factor'))     # Factor by which to reduce LR on Plateau
    patience_lr_reduction = int(settings_dict.get('pat_lr_red'))        # N epochs stagnating val loss before reducing lr
    criterion_txt = settings_dict.get('criterion')                      # Criterion in txt format, lowercase (e.g. l1loss)
    optimizer_txt = settings_dict.get('optimizer')                      # Optimizer in txt format, lowercase (e.g. adam)
    n_epochs = settings_dict.get('n_epochs')                            # Number of epochs
    patience_stop = int(settings_dict.get('pat_stop'))                  # N epochs stagnating val loss before stopping
    n_splits = int(settings_dict.get('n_splits'))                       # N Train/Val splits
    test_subjects = settings_dict.get('test_subjects')                  # Get test subjects

    # Preprocess participants dataframe + save to workspace path as reference
    df, colnames_dict = prepare_participants_df(bids_root_path, colnames_dict, subject_sessions,
                                                modalities_dict, derivative_name)
    df.to_csv(os.path.join(workspace_path, 'participants.tsv'), sep='\t')

    # Split data with same test set as federated experiment
    test_df = df[df[colnames_dict['id']].isin(test_subjects)]
    test_loader, n_test = get_data_loader(test_df, 'test', colnames_dict, batch_size, return_n=True)
    train_overall_df = df[~df[colnames_dict['id']].isin(test_subjects)]
    train_val_ids = train_overall_df[colnames_dict['id']]

    # General deep learning settings
    criterion = get_criterion(criterion_txt)
    architecture_path = os.path.join(workspace_path, 'architecture.py')
    net_architecture = import_net_architecture(architecture_path)
    if initial_sd_path is not None:
        net = get_weights(net_architecture, initial_sd_path)
    else:
        net = copy_net(net_architecture)

    # Initialize variables related to validation loss tracking
    val_loss_ref = np.inf       # Reference validation loss
    counter_lr_red = 0          # Counter for lr reduction
    counter_stop = 0            # Counter stop
    best_net = None             # Best net initialisation

    for epoch in range(n_epochs + 1):  # Start counting from 1
        print(f'Epoch {epoch}/{n_epochs}...')

        # Deep learning settings per epoch
        optimizer = get_optimizer(optimizer_txt, net, lr)

        # Initiate variables
        model_error_dict = {}
        mean_val_loss = 0
        split_model = None
        random_states = range(n_splits * epoch, n_splits * epoch + n_splits)  # Assure random state is never repeated
        train_results_df = pd.DataFrame()
        for split_i, random_state in enumerate(random_states):
            print(f'==> Split {split_i}/{n_splits} (split random state = {random_state})')
            # Split data
            train_ids, val_ids = train_test_split(train_val_ids, random_state=random_state,
                                                  test_size=val_fraction / (train_fraction + val_fraction))
            train_df = df[df[colnames_dict.get('id')].isin(train_ids)][list(colnames_dict.values())]
            val_df = df[df[colnames_dict.get('id')].isin(val_ids)][list(colnames_dict.values())]

            # Save dataframes
            if epoch == 1 and split_i == 0:  # fl_round starts from 1
                train_overall_df.to_csv(os.path.join(workspace_path, 'train_overall_df.tsv'), sep='\t',
                                        index=False)
                test_df.to_csv(os.path.join(workspace_path, 'test_df.tsv'), sep='\t', index=False)

            # Create data loaders
            train_loader, n_train = get_data_loader(train_df, 'train', colnames_dict, batch_size, return_n=True)
            val_loader, n_val = get_data_loader(val_df, 'validation', colnames_dict, batch_size, return_n=True)

            # Train
            print('==> Start training...')
            split_model, best_val_loss, train_loss_list, val_loss_list = train(
                1, device, train_loader, val_loader, optimizer, net, criterion, None,
                False, None
            )

            train_results_df_i = pd.DataFrame({'random_state': [random_state] * len(train_loss_list),
                                               'epoch': [epoch] * len(train_loss_list),
                                               'train_loss': train_loss_list,
                                               'val_loss': val_loss_list,
                                               'n_train': [n_train] * len(train_loss_list),
                                               'n_val': [n_val] * len(train_loss_list),
                                               'n_test': [n_test] * len(train_loss_list)})
            train_results_df = pd.concat([train_results_df, train_results_df_i], axis=0)

            # Update mean val loss
            mean_val_loss += best_val_loss / n_splits

            # Update dict
            model_error_dict.update({split_model: best_val_loss})

        train_results_df_path = os.path.join(workspace_path, f'centralised_epoch_{epoch}_train_results.csv')
        train_results_df.to_csv(train_results_df_path, index=False)

        # Get average model across splits
        if n_splits > 1:
            net = get_weighted_average_model(model_error_dict)
        else:
            net = copy_net(split_model)
        epoch_model_path = os.path.join(workspace_path, f'centralised_epoch_{epoch}_model.pt')
        torch.save(net.state_dict(), epoch_model_path)

        # Perform actions based on min validation loss across splits and epochs
        print('==> Validation loss tracking...')
        if mean_val_loss < val_loss_ref:    # Improvement
            val_loss_ref = mean_val_loss
            counter_lr_red = 0
            counter_stop = 0
            best_net = copy_net(net)
        else:                               # No improvement
            counter_lr_red += 1
            counter_stop += 1
            if counter_lr_red == patience_lr_reduction:
                lr *= lr_reduce_factor
                counter_lr_red = 0
            if counter_stop == patience_stop:
                print(f'Stopping early at epoch {epoch}...')
                break
        print(f'     ==> lr reduction counter: {counter_lr_red}')

    # Test
    print('Testing...')
    test_loss, true_labels_test, pred_labels_test = evaluate(best_net, test_loader, criterion, device, 'test')
    torch.save(best_net.state_dict(), os.path.join(workspace_path, 'final_model.pt'))

    # Test result analysis
    true_pred_test_df = create_test_true_pred_df(true_labels_test, pred_labels_test, workspace_path, save=True)
    create_test_scatterplot(true_pred_test_df, 'centralised', workspace_path)
    with open(os.path.join(workspace_path, 'test_results.csv'), 'w') as f:
        f.write(f'MAE: {mean_absolute_error(true_labels_test, pred_labels_test)}\n'
                f'Pearsonr: {pearsonr(true_labels_test, pred_labels_test)}')

    # Clean up workspace
    print('Cleaning up workspace...')
    clean_up_workspace(workspace_path, who='client')

    # Print and save total FL duration
    stop_time = dt.datetime.now()
    fl_duration_print_save(start_time, stop_time, workspace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Centralised training',
        description='Training a model centralised'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    args = parser.parse_args()
    train_centralised(args.settings_path)
