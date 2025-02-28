"""
This script visualises the loss across fl rounds
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_loss_figure(results_dir_path, fig_width):

    # Collect clients
    clients = []
    best_model_round = None
    for file in os.listdir(results_dir_path):
        if file.endswith("test_results.csv"):
            client = file.replace('_test_results.csv', '')
            clients.append(client)
        elif file == 'final_model.txt':
            with open(os.path.join(results_dir_path, file), 'r') as file:
                best_model_round = int(file.read().split('_')[-1].split('.')[0])
    clients = sorted(clients)


    # Initiate figure and variables
    fig, axes = plt.subplots(nrows=len(clients) + 1, ncols=1, figsize=(fig_width, fig_width * 0.7 * len(clients)))

    client_dfs = []
    for i, client in enumerate(clients):
        # Load training results
        regexp = f'\A{client}_round_[0-9]+_train_results.csv\Z'
        csv_files = list(filter(lambda x: re.fullmatch(regexp, x), os.listdir(results_dir_path)))

        all_rounds_df = pd.DataFrame()
        for csv_file in csv_files:
            csv_path = os.path.join(results_dir_path, csv_file)
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
        client_dfs.append(all_rounds_df)

        # Plot
        # Source: https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
        axes[i].plot(all_rounds_df['fl_round'], all_rounds_df['train_loss_mean'], color='blue', label='train')
        axes[i].fill_between(all_rounds_df['fl_round'], all_rounds_df['train_loss_95_CI_lower'],
                             all_rounds_df['train_loss_95_CI_upper'], color='blue', alpha=.06)
        axes[i].plot(all_rounds_df['fl_round'], all_rounds_df['val_loss_mean'], color='red', label='validation')
        axes[i].fill_between(all_rounds_df['fl_round'], all_rounds_df['val_loss_95_CI_lower'],
                             all_rounds_df['val_loss_95_CI_upper'], color='red', alpha=.06)
        axes[i].legend(fontsize=14)

        # Figure aesthetics
        test_results_client = pd.read_csv(os.path.join(results_dir_path, f'{client}_test_results.csv'))
        test_loss_client = test_results_client['test_loss'].iloc[0]
        title_center = f'test MAE = {test_loss_client:.2f}'
        title_right = f'Client: {client}'
        xticks = np.arange(all_rounds_df['fl_round'].min(), all_rounds_df['fl_round'].max()+1, 1)
        xticks = list(filter(lambda x: x % 5 == 0, xticks))  # Filter xticks to intervals of 5
        axes[i] = fig_aesthetics(ax=axes[i], xticks=xticks, best_model_round=best_model_round,
                                 title_center=title_center, title_right=title_right)

    # Last subplot
    last_ax = len(clients)
    avg_val_loss_clients = pd.read_csv(os.path.join(results_dir_path, 'avg_val_loss_clients.csv'))
    axes[last_ax].plot(avg_val_loss_clients['fl_round'], avg_val_loss_clients['avg_val_loss_clients'],
                       color='black', label='avg validation\nacross clients')
    axes[last_ax].legend(fontsize=14, loc='upper right')
    with open(os.path.join(results_dir_path, 'overall_test_mae.txt'), 'r') as txt_file:
        title_center = f'overall test MAE: {float(txt_file.read().strip("Overall test MAE: ")):.2f}'
    title_right = 'Server'
    xticks = np.arange(avg_val_loss_clients['fl_round'].min(), avg_val_loss_clients['fl_round'].max() + 1, 1)
    xticks = list(filter(lambda x: x%5==0, xticks))  # Filter xticks to intervals of 5
    xlabel = 'FL rounds'
    axes[last_ax] = fig_aesthetics(ax=axes[last_ax], xticks=xticks, best_model_round=best_model_round,
                                   title_center = title_center, title_right=title_right, xlabel=xlabel)

    return fig, axes


def fig_aesthetics(ax, xticks, best_model_round, title_center=None, title_right=None, xlabel=None):
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('MAE', fontsize=20)
    ax.axvline(x=best_model_round, color='#808080')
    y_text = ax.get_ylim()[0] + 0.67 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    # y_text = np.log10(ax.get_ylim()[0] + 0.67 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.text(x=best_model_round + 0.25, y=y_text, s='final model', color='#808080', fontsize=14, rotation=90)
    if title_center is not None:
        ax.set_title(title_center, fontsize=18)
    if title_right is not None:
        ax.set_title(title_right, loc='right', fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    # ax.set_yscale('log')

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir_path', type=str, help='Path to the results directory')
    parser.add_argument('--figure_filename', default='loss_figure.png', type=str, help='Filename of the figure')
    parser.add_argument('--figure_width', type=int, default=10, help='Figure width')
    args = parser.parse_args()

    # Create figure
    fig, axes = create_loss_figure(args.results_dir_path, args.figure_width)

    # Save figure to results dir
    fig.savefig(os.path.join(args.results_dir_path, args.figure_filename), dpi=300)
