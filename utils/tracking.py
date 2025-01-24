"""
Functions related to tracking
"""

import os
import pandas as pd
from communication import is_experiment_folder


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
            if file.endswith('train_results.csv') and sum([is_experiment_folder(x) for x in root.split(os.sep)]) == 0:
                client = file.split('_')[0]
                sub_df = pd.read_csv(os.path.join(root, file))
                sub_df['client'] = [client] * len(sub_df)
                df = pd.concat([df, sub_df])

    # Sort dataframe
    df = df.sort_values(by=['fl_round', 'client']).reset_index(drop=True)

    return df
