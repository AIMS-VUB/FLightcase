"""
This script visualises the loss across fl rounds
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def create_loss_figure(workspace_dir_path, fig_width, clients, server_FL_settings):
    # Initiate figure and variables
    fig, axes = plt.subplots(nrows=len(clients), ncols=1, figsize=(fig_width, fig_width * 0.4 * len(clients)))

    for i, client in enumerate(clients):
        # Load training results
        regexp = f'\Atrain_results_{client}_round_[0-9]+.csv\Z'
        csv_files = list(filter(lambda x: re.fullmatch(regexp, x), os.listdir(workspace_dir_path)))

        all_rounds_df = pd.DataFrame()
        for csv_file in csv_files:
            csv_path = os.path.join(workspace_dir_path, csv_file)
            round_df = pd.read_csv(csv_path)[['fl_round', 'train_loss', 'val_loss']]
            round_summary = pd.DataFrame(
                {
                    'fl_round': [round_df['fl_round'][0]],
                    'train_loss_mean': [round_df['train_loss'].mean()],
                    'train_loss_95_CI_lower': [round_df['train_loss'].mean() - round_df['train_loss'].std()*1.96],
                    'train_loss_95_CI_upper': [round_df['train_loss'].mean() + round_df['train_loss'].std() * 1.96],
                    'val_loss_mean': [round_df['val_loss'].mean()],
                    'val_loss_95_CI_lower': [round_df['val_loss'].mean() - round_df['val_loss'].std()*1.96],
                    'val_loss_95_CI_upper': [round_df['val_loss'].mean() + round_df['val_loss'].std() * 1.96]
                }
            )
            all_rounds_df = pd.concat([all_rounds_df, round_summary], axis=0)
        all_rounds_df = all_rounds_df.sort_values('fl_round')

        # Plot
        # Source: https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
        axes[i].plot(all_rounds_df['fl_round'], all_rounds_df['train_loss_mean'], color='blue', label='train')
        axes[i].fill_between(all_rounds_df['fl_round'], all_rounds_df['train_loss_95_CI_lower'],
                             all_rounds_df['train_loss_95_CI_upper'], color='blue', alpha=.06)
        axes[i].plot(all_rounds_df['fl_round'], all_rounds_df['val_loss_mean'], color='red', label='val')
        axes[i].fill_between(all_rounds_df['fl_round'], all_rounds_df['val_loss_95_CI_lower'],
                             all_rounds_df['val_loss_95_CI_upper'], color='red', alpha=.06)
        axes[i].legend()

        # Figure aesthetics
        axes[i].set_title(server_FL_settings
                          .get('client_credentials')
                          .get(client)
                          .get('name'), loc='right', color='#AEAEAE')
        if i == len(clients) - 1:
            axes[i].set_xlabel('FL rounds', fontsize=16)
        axes[i].set_ylabel('L1 Loss', fontsize=16)

    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace_dir_path', type=str, help='Path to the workspace directory')
    parser.add_argument('--figure_filename', type=str, help='Filename of the figure')
    parser.add_argument('--figure_width', type=int, default=10, help='Figure width')
    parser.add_argument('--clients', nargs='+', help='Client ip addresses')
    parser.add_argument('--server_settings_path', type=str, help='Path to server settings JSON')
    args = parser.parse_args()

    # Create figure
    with open(args.server_settings_path, 'r') as json_file:
        server_FL_settings = json.load(json_file)
    fig, axes = create_loss_figure(args.workspace_dir_path, args.figure_width, args.clients, server_FL_settings)

    # Save figure to workspace dir
    fig.savefig(os.path.join(args.workspace_dir_path, args.figure_filename), dpi=300)
