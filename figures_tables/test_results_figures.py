"""
Figures displaying model performance on test datasets
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def create_starplot(values_matrix, categories, plot_names, n_tick_decimals=0, y_tick_list=None):
    """
    Code acknowledgement: Prof. Dr. ir. Guy Nagels
    """
    # Calculate angle for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    # Repeat the first angle to close the circle
    angles += angles[:1]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), subplot_kw=dict(polar=True))

    # Define colors for each subplot
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    # Determine the global min and max values for consistent color scales across subplots
    all_values = np.concatenate(values_matrix)
    vmin = all_values.min()
    vmax = all_values.max()

    for ax, values, name, color in zip(axs.flat, values_matrix, plot_names, colors):
        # Repeat the first value to close the circle
        values_circular = values + values[:1]

        # Draw the outline of our data
        ax.fill(angles, values_circular, color=color, alpha=0.25)

        # Draw the line around the data
        ax.plot(angles, values_circular, color=color, linewidth=2)

        # Set the model name as the title
        ax.set_title(name, size=15, color='navy', y=1.1)

        # Define the range of MAE values and generate gridlines and labels
        if y_tick_list is not None:
            value_range = y_tick_list
        else:
            value_range = np.linspace(vmin, vmax, 5)

        ax.set_yticks(value_range)
        if n_tick_decimals == 0:
            ax.set_yticklabels([f'{x:.0f}' for x in value_range], color='grey')
        elif n_tick_decimals == 1:
            ax.set_yticklabels([f'{x:.1f}' for x in value_range], color='grey')
        ax.yaxis.set_tick_params(labelsize=8)

        # Set x-ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

    # Adjust layout to prevent label overlap and to ensure that the subplot titles and labels fit well
    plt.tight_layout(pad=4.0)

    return fig, axs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir_path', type=str)
    args = parser.parse_args()

    matplotlib.use('TkAgg')  # !IMPORTANT
    # root_dir_path = '/home/sdenisse/Documents/FL_POC/review_CIBM'
    root_dir_path = args.root_dir_path

    # Read dataset sizes
    dataset_size_dict = {}
    for client in ['Brussels', 'Greifswald', 'Prague']:
        txt_path = os.path.join(root_dir_path, 'freeze_up_to_norm_5', 'FL_workspace', f'{client}_dataset_size.txt')
        with open(txt_path, 'r') as txt_file:
            ds_size = int(txt_file.read())
            dataset_size_dict.update({client: ds_size})

    ####################
    # Federated learning
    ####################

    # Initialise dataframe
    df_fl = pd.DataFrame()

    models = ['freeze_up_to_norm_5', 'freeze_up_to_trans_3', 'freeze_up_to_trans_2', 'freeze_up_to_trans_1', 'no_freeze', 'Glorot_initialisation']
    xlabels = ['TL_model_1', 'TL_model_2', 'TL_model_3', 'TL_model_4', 'TL_model_5', 'Glorot_model']

    for model, xlabel in zip(models, xlabels):
        # Load dataframes
        df_Brussels = pd.read_csv(os.path.join(root_dir_path, model, "FL_workspace", "test_results_Brussels.csv"))
        df_Greifswald = pd.read_csv(os.path.join(root_dir_path, model, "FL_workspace", "test_results_Greifswald.csv"))
        df_Prague = pd.read_csv(os.path.join(root_dir_path, model, "FL_workspace", "test_results_Prague.csv"))

        with open(os.path.join(root_dir_path, model, 'FL_workspace', 'overall_test_mae.txt'), 'r') as txt_file:
            overall_test_mae = float(txt_file.read().split('Overall test MAE: ')[1])

        overall_test_r = (df_Brussels['r_true_pred'] * dataset_size_dict.get('Brussels') +
                          df_Greifswald['r_true_pred'] * dataset_size_dict.get('Greifswald') +
                          df_Prague['r_true_pred'] * dataset_size_dict.get('Prague')) / sum(dataset_size_dict.values())

        row = {'model': model,
               'xlabel': xlabel,
               'overall_test_mae': overall_test_mae,
               'overall_Pearson_r': overall_test_r}
        for client, client_df in zip(['Brussels', 'Greifswald', 'Prague'], [df_Brussels, df_Greifswald, df_Prague]):
            for colname in ['test_mae', 'r_true_pred', 'p_true_pred',
                            'true_mean', 'true_sd', 'true_skewness', 'true_kurtosis', 'true_shapiro-wilk_stat', 'true_shapiro-wilk_p',
                            'pred_mean', 'pred_sd', 'pred_skewness', 'pred_kurtosis', 'pred_shapiro-wilk_stat', 'pred_shapiro-wilk_p'
                            ]:
                row.update({f'{client}_{colname}': client_df[colname]})
        df_fl = pd.concat([df_fl, pd.DataFrame(row)])

    df_fl.reset_index(drop=True)
    df_fl.to_csv(os.path.join(root_dir_path, 'FL_test_results.csv'))

    # Test MAE figure
    # Starplot
    test_mae_matrix = []
    # CAVE: columns and clients must be in same order
    columns = ['Brussels_test_mae', 'Greifswald_test_mae', 'Prague_test_mae', 'overall_test_mae']
    clients = ['Brussels', 'Greifswald', 'Prague', 'overall']
    for model in models:
        test_mae_matrix.append(df_fl[df_fl['model'] == model][columns].iloc[0].to_list())
    # y_tick_list = np.arange(0, 24, 3)
    y_tick_list = np.arange(0, 2.4, 0.3)  # For z-normalised SDMT scores
    # n_tick_decimals = 0
    n_tick_decimals = 1
    fig, ax = create_starplot(test_mae_matrix, clients, xlabels, n_tick_decimals=n_tick_decimals, y_tick_list=y_tick_list)
    plt.savefig(os.path.join(root_dir_path, 'figure_test_mae_across_models.png'), dpi=1000)

    # Pearson r figure
    # Starplot
    test_r_matrix = []
    # CAVE: columns and clients must be in same order
    columns = ['Brussels_r_true_pred', 'Greifswald_r_true_pred', 'Prague_r_true_pred', 'overall_Pearson_r']
    clients = ['Brussels', 'Greifswald', 'Prague', 'overall']
    for model in models:
        test_r_matrix.append(df_fl[df_fl['model'] == model][columns].iloc[0].to_list())
    fig, ax = create_starplot(test_r_matrix, clients, xlabels, n_tick_decimals=1, y_tick_list=np.arange(0, 1, 0.2))
    plt.savefig(os.path.join(root_dir_path, 'figure_test_pearson_r_across_models.png'), dpi=1000)

    #################
    # Client-specific
    #################

    # Initialise dataframe
    df_client_specific = pd.DataFrame()
    # Originally: freeze_up_to_norm_5 was fully_connected
    for tl_model, xlabel in zip(['freeze_up_to_norm_5', 'freeze_up_to_trans_3', 'freeze_up_to_trans_2', 'freeze_up_to_trans_1', 'no_freeze', 'Glorot_initialisation'],
                                ['TL_model_1', 'TL_model_2', 'TL_model_3', 'TL_model_4', 'TL_model_5', 'Glorot_model']):

        for client_model in ['Brussels', 'Greifswald', 'Prague']:
            client_specific_overall_test_r = 0
            df_overall_test_mae = pd.read_csv(os.path.join(root_dir_path, tl_model, "FL_workspace", "client-specific",
                                                           "overall_test_mae_results.csv"))
            row = {'tl_model': tl_model,
                   'client_model': client_model,
                   'xlabel': xlabel,
                   'overall_test_mae': [df_overall_test_mae[f'{client_model}_model'].iloc[0]]}

            for client_dataset in ['Brussels', 'Greifswald', 'Prague']:
                # Load dataframe
                df_client_data = pd.read_csv(os.path.join(root_dir_path, tl_model, "FL_workspace", "client-specific", f"test_results_{client_dataset}.csv"))
                # CHECK 'client' replaced with 'client_model' after git commit of weekend 28/04/2024
                client_specific_overall_test_r += (df_client_data[df_client_data['client_model'] == client_model]['r_true_pred'].iloc[0] * dataset_size_dict.get(client_dataset)) / sum(dataset_size_dict.values())
                for colname in ['test_mae', 'r_true_pred', 'p_true_pred',
                                'true_mean', 'true_sd', 'true_skewness', 'true_kurtosis', 'true_shapiro-wilk_stat', 'true_shapiro-wilk_p',
                                'pred_mean', 'pred_sd', 'pred_skewness', 'pred_kurtosis', 'pred_shapiro-wilk_stat', 'pred_shapiro-wilk_p'
                                ]:
                    row.update({f'{client_dataset}_{colname}': [df_client_data[df_client_data['client_model'] == client_model][colname].iloc[0]]})

            row.update({'overall_Pearson_r': [client_specific_overall_test_r]})
            df_client_specific = pd.concat([df_client_specific, pd.DataFrame(row)])

    df_client_specific.reset_index(drop=True)
    df_client_specific.to_csv(os.path.join(root_dir_path, 'client-specific_test_results.csv'))

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    for row_nr, client_model in enumerate(['Brussels', 'Greifswald', 'Prague']):
        df_to_plot = df_client_specific[df_client_specific['client_model'] == client_model]

        # Test MAE
        test_mae_matrix = []
        # CAVE: columns and clients must be in same order
        columns = ['Brussels_test_mae', 'Greifswald_test_mae', 'Prague_test_mae', 'overall_test_mae']
        clients = ['Brussels', 'Greifswald', 'Prague', 'overall']
        for model in models:
            test_mae_matrix.append(df_to_plot[df_to_plot['tl_model'] == model][columns].iloc[0].to_list())
        # y_tick_list = np.arange(0, 24, 3)
        y_tick_list = np.arange(0, 2.4, 0.3)  # For z-normalised SDMT scores
        # n_tick_decimals = 0
        n_tick_decimals = 1
        fig, ax = create_starplot(test_mae_matrix, clients, xlabels, n_tick_decimals=n_tick_decimals, y_tick_list=y_tick_list)
        plt.savefig(os.path.join(root_dir_path, f'figure_test_mae_{client_model}.png'), dpi=1000)

        # Pearson r
        test_r_matrix = []
        # CAVE: columns and clients must be in same order
        columns = ['Brussels_r_true_pred', 'Greifswald_r_true_pred', 'Prague_r_true_pred', 'overall_Pearson_r']
        clients = ['Brussels', 'Greifswald', 'Prague', 'overall']

        for model in models:
            test_r_matrix.append(df_to_plot[df_to_plot['tl_model'] == model][columns].iloc[0].to_list())
        fig, ax = create_starplot(test_r_matrix, clients, xlabels, n_tick_decimals=1, y_tick_list=np.arange(0, 1, 0.2))
        plt.savefig(os.path.join(root_dir_path, f'figure_test_pearson_r_{client_model}.png'), dpi=1000)
