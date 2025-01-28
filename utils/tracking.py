"""
Functions related to tracking
"""

import os
import pandas as pd
from communication import experiment_folder_in_path


def print_FL_plan(FL_plan_dict):
    """ Print FL plan

    :param FL_plan_dict: dict, FL plan
    """
    print('\n========\nFL plan:\n========\n')
    for k, v in FL_plan_dict.items():
        print(f'- {k}: {v}')
    print('\n')


def create_overall_loss_df(workspace_path_server):
    """
    This function combines all dataframes (across rounds and clients) with train/validation loss in one dataframe
    :param workspace_path_server: str, path to the workspace folder of the server
    """

    df = pd.DataFrame()
    for root, dirs, files in os.walk(workspace_path_server):
        for file in files:
            # Skip files in other experiment folders
            if file.endswith('train_results.csv') and not experiment_folder_in_path(root):
                client = file.split('_')[0]
                sub_df = pd.read_csv(os.path.join(root, file))
                sub_df['client'] = [client] * len(sub_df)
                df = pd.concat([df, sub_df])

    # Sort dataframe
    df = df.sort_values(by=['fl_round', 'client']).reset_index(drop=True)

    return df


def fl_duration_print_save(start_time, stop_time, workspace_path_server):
    """
    This function prints and saves the FL duration
    :param start_time: datetime object
    :param stop_time: datetime object
    :param workspace_path_server: str, absolute path to workspace server
    """

    fl_duration = stop_time - start_time
    with open(os.path.join(workspace_path_server, 'FL_duration.txt'), 'w') as duration_file:
        duration_file.write(str(fl_duration))
    print(f'Total federated learning duration: {str(fl_duration)}')
