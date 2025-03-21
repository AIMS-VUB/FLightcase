"""
Centralised version of the "create_loss_figure.py" script
"""


import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    parser.add_argument('--figure_width', type=int, default=15, help='Figure width')
    parser.add_argument('--best_epoch', type=int, help='Best epoch')
    args = parser.parse_args()

    # Initiate figure and variables
    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_width/2))

    # Load training results
    regexp = f'\Acentralised_epoch_[0-9]+_train_results.csv\Z'
    csv_files = list(filter(lambda x: re.fullmatch(regexp, x), os.listdir(args.results_dir_path)))

    epoch_df = pd.DataFrame()
    for csv_file in csv_files:
        csv_path = os.path.join(args.results_dir_path, csv_file)
        round_df = pd.read_csv(csv_path)[['epoch', 'train_loss', 'val_loss']]
        round_summary = pd.DataFrame(
            {
                'epoch': [round_df['epoch'][0]],
                'train_loss_mean': [round_df['train_loss'].mean()],
                'train_loss_95_CI_lower': [round_df['train_loss'].mean() - round_df['train_loss'].std()*1.96],
                'train_loss_95_CI_upper': [round_df['train_loss'].mean() + round_df['train_loss'].std() * 1.96],
                'val_loss_mean': [round_df['val_loss'].mean()],
                'val_loss_95_CI_lower': [round_df['val_loss'].mean() - round_df['val_loss'].std()*1.96],
                'val_loss_95_CI_upper': [round_df['val_loss'].mean() + round_df['val_loss'].std() * 1.96]
            }
        )
        epoch_df = pd.concat([epoch_df, round_summary], axis=0)
    epoch_df = epoch_df.sort_values('epoch')

    # Plot
    # Source: https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
    ax.plot(epoch_df['epoch'], epoch_df['train_loss_mean'], color='blue', label='train')
    ax.fill_between(epoch_df['epoch'], epoch_df['train_loss_95_CI_lower'],
                         epoch_df['train_loss_95_CI_upper'], color='blue', alpha=.06)
    ax.plot(epoch_df['epoch'], epoch_df['val_loss_mean'], color='red', label='validation')
    ax.fill_between(epoch_df['epoch'], epoch_df['val_loss_95_CI_lower'],
                         epoch_df['val_loss_95_CI_upper'], color='red', alpha=.06)
    ax.legend(fontsize=14)

    # Figure aesthetics
    test_results_df = pd.read_csv(os.path.join(args.results_dir_path, f'test_results.csv'))
    test_mae = round(float(test_results_df.columns[0].replace('MAE: ', '')), 2)
    title_center = f'test MAE = {test_mae:.2f}'
    title_right = f'Centralised'
    xticks = np.arange(epoch_df['epoch'].min(), epoch_df['epoch'].max()+1, 1)
    xticks = list(filter(lambda x: x % 5 == 0, xticks))  # Filter xticks to intervals of 5
    ax = fig_aesthetics(ax=ax, xticks=xticks, best_model_round=args.best_epoch,
                        title_center=title_center, title_right=title_right, xlabel='Epoch')

    # Save figure to results dir
    fig.savefig(os.path.join(args.results_dir_path, args.figure_filename), dpi=300)
