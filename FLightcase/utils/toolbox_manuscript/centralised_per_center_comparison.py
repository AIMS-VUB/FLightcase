"""
Investigate the difference between centers in the centralised experiment
"""

import os
import argparse
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_participants_df")
    parser.add_argument("--path_true_pred_df")
    parser.add_argument("--output_dir_path")
    parser.add_argument("--colname_dataset_name")
    parser.add_argument("--colname_id")
    args = parser.parse_args()

    # Read dataframes
    df_participants = pd.read_csv(args.path_participants_df, sep='\t')
    df_true_pred = pd.read_csv(args.path_true_pred_df)
    df_true_pred['BAD'] = df_true_pred['pred'] - df_true_pred['true']

    # Search for
    df_true_pred_enriched = pd.merge(left=df_true_pred,
                                     right=df_participants[[args.colname_dataset_name, args.colname_id]],
                                     how='left',
                                     on=args.colname_id)

    txt = ''
    for ds_name in df_true_pred_enriched[args.colname_dataset_name].unique():
        df_ds = df_true_pred_enriched[df_true_pred_enriched[args.colname_dataset_name] == ds_name]

        # Get statistics
        r_ba_age, p_ba_age = stats.pearsonr(df_ds['true'], df_ds['pred'])
        r_bad_ba, p_bad_ba = stats.pearsonr(df_ds['BAD'], df_ds['pred'])
        r_bad_age, p_bad_age = stats.pearsonr(df_ds['BAD'], df_ds['true'])
        mae = mean_absolute_error(df_ds['true'], df_ds['pred'])

        txt += f'Centralised analysis for test dataframe: {ds_name}\n\n'
        txt += df_ds['BAD'].describe().to_string()
        txt += f'\n\nPearson r between age and brain age" {r_ba_age} ({p_ba_age})\n\n'
        txt += f'Pearson r between BAD and brain age" {r_bad_ba} ({p_bad_ba})\n\n'
        txt += f'Pearson r between BAD and true age" {r_bad_age} ({p_bad_age})\n\n'
        txt += f'MAE = {mae}\n\n'

    with open(os.path.join(args.output_dir_path, "per_center_comparison.txt"), 'w') as fp:
        fp.write(txt)

