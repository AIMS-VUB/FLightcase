"""
This script performs the same train/val/test split as in the client.py script.
Test always remains the same. Train and val differs per iteration, so they are concatenated into an "overall_train_df"
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from DL_utils.data import split_data


def mean_diff(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Client',
        description='Client for federated learning'
    )
    parser.add_argument('--settings_path', type=str, help='Path to the settings JSON')
    parser.add_argument('--FL_plan_path', type=str, help='Path to the FL plan JSON')
    parser.add_argument('--output_dir_path', type=str, help='Path to output directory')
    parser.add_argument('--colnames_describe_method', nargs='*', help='List the column names to assess with the '
                                                                      '.describe() method')
    parser.add_argument('--colnames_value_counts_method', nargs='*', help='List the column names to assess with the '
                                                                          '.value_counts() method')
    args = parser.parse_args()
    settings_path = args.settings_path
    FL_plan_path = args.FL_plan_path
    output_dir_path = args.output_dir_path

    # Extract settings
    with open(settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    colname_id = settings_dict.get('colname_id')                    # Column name of the BIDS id column
    colname_img_path = settings_dict.get('colname_img_path')        # Column name of the image paths
    colname_label = settings_dict.get('colname_label')              # Column name of the label column
    subject_ids = settings_dict.get('subject_ids')                  # Which subject ids to take into account?
    bids_root_path = settings_dict.get('bids_root_path')            # Path to BIDS root

    # Extract FL plan
    with open(FL_plan_path, 'r') as json_file:
        FL_plan_dict = json.load(json_file)
    train_fraction = float(FL_plan_dict.get('train_fraction'))      # Fraction of data for training
    val_fraction = float(FL_plan_dict.get('val_fraction'))          # Fraction of data for validation
    test_fraction = float(FL_plan_dict.get('test_fraction'))        # Fraction of data for testing

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

    random_state = 42  # Note: this is iteration-dependent during federated learning
    # split_data only retains columns in "colnames_dict". Obtain dfs in 2 steps
    train_df, val_df, test_df = split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction,
                                           train_test_random_state=42, train_val_random_state=random_state)
    train_overall_df = pd.concat([train_df, val_df], ignore_index=True)

    # Extract ids and extract those ids from original dataframe to get all data
    train_overall_df = df[df[colname_id].isin(train_overall_df[colname_id])]
    test_df = df[df[colname_id].isin(test_df[colname_id])]
    train_overall_df.to_csv(os.path.join(output_dir_path, 'train_overall_df.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir_path, 'test_df.csv'), index=False)

    txt = ''
    # Add dataset description
    for df_to_describe, dataset_type in zip([train_overall_df, test_df], ['Train', 'Test']):
        txt += f'***********\n{dataset_type}\n***********\n\n'

        # Return .txt file with values
        txt += f'==================\n' \
               f'.describe() method\n' \
               f'==================\n\n'
        for col in args.colnames_describe_method:
            txt += f'{df_to_describe[col].describe()}\n\n'

        txt += f'======================\n' \
               f'.value_counts() method\n' \
               f'======================\n\n'
        for col in args.colnames_value_counts_method:
            txt += f'{df_to_describe[col].value_counts()}\n\n'

    # Add test comparisons
    txt += '==============================================\n' \
           'Comparisons between train and test            \n' \
           'Note: continuous (chi-squared separate script)\n' \
           '==============================================\n\n'
    for col in args.colnames_describe_method:
        res = stats.permutation_test((train_overall_df[col], test_df[col]), statistic=mean_diff, vectorized=True,
                                     n_resamples=100000, alternative='two-sided')
        txt += f'{col}: Statistic = {res.statistic}, p = {res.pvalue}\n'

    with open(os.path.join(output_dir_path, 'train_test_comparison.txt'), 'w') as txt_file:
        txt_file.write(txt)
