"""
This script creates a prettified scatter plot of true vs predicted on a test dataset
"""

import os
import argparse
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def pearson_p_to_text(p):
    if p < .001:
        return 'p < .001'
    else:
        return f'p = {round(p, 3)}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_name", help='Name of the client')
    parser.add_argument("--results_dir_path", help='Path to results directory of the client')
    parser.add_argument("--xlabel", default='True', help="Label of the x axis")
    parser.add_argument("--ylabel", default='Predicted', help="Label of the y axis")
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join(args.results_dir_path, 'true_pred_test.csv'))
    fig, ax = plt.subplots()
    ax.scatter(x=test_df['true'], y=test_df['pred'], color='black')
    x_y_lim_list = list(ax.get_xlim()) + list(ax.get_ylim())
    min_x_y = min(x_y_lim_list)
    max_x_y = max(x_y_lim_list)

    # Plot x = y line
    ax.plot((min_x_y, max_x_y), (min_x_y, max_x_y), color='grey', linestyle='--')
    ax.set_xlabel(args.xlabel, fontsize=14)
    ax.set_ylabel(args.ylabel, fontsize=14)
    r_pearson, p_pearson = stats.pearsonr(test_df['true'], test_df['pred'])
    mae = mean_absolute_error(test_df['true'], test_df['pred'])
    ax.set_title(f'Test performance {args.client_name}:\n'
                 f'MAE: {mae:.2f}, r: {r_pearson:.2f} ({pearson_p_to_text(p_pearson)})')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(args.results_dir_path, f'{args.client_name}_true_pred_test_scatterplot_prettified.png'))
