"""
Calculate brain age difference (BAD)
"""

import os
import argparse
import pandas as pd
import scipy.stats as stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_true_pred_df")
    parser.add_argument("--output_dir_path")
    args = parser.parse_args()

    # Read dataframe
    df = pd.read_csv(args.path_true_pred_df)
    df['BAD'] = df['pred'] - df['true']

    # Get pearson correlations with BAD
    r_bad_ba, p_bad_ba = stats.pearsonr(df['BAD'], df['pred'])
    r_bad_age, p_bad_age = stats.pearsonr(df['BAD'], df['true'])

    txt = f'BAD analysis for dataframe: {args.path_true_pred_df}\n\n'
    txt += df['BAD'].describe().to_string()
    txt += f'\n\nPearson r between BAD and brain age" {r_bad_ba} ({p_bad_ba})\n\n'
    txt += f'Pearson r between BAD and true age" {r_bad_age} ({p_bad_age})'

    with open(os.path.join(args.output_dir_path, "BAD_analysis.txt"), 'w') as fp:
        fp.write(txt)
