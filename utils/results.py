"""
Functions related to results calculation and integration
"""

import os
import sys
import pathlib
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
# Add path to parent dir of this Python file: https://stackoverflow.com/questions/3430372/
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from communication import send_file


def update_avg_val_loss(client_info_dict, val_loss_avg, fl_round):
    for client_name in client_info_dict.keys():
        train_results_client_df = client_info_dict[client_name][f'round_{fl_round}']['train_results']
        val_loss_avg += train_results_client_df['val_loss'].mean() / len(client_info_dict)
    return val_loss_avg


def calculate_overall_test_mae(client_info_dict, workspace_path_server, save=True):
    test_mae_overall = 0
    n_sum_clients = sum([dct['dataset_size'] for dct in client_info_dict.values()])
    for client_name in client_info_dict.keys():
        n_client = client_info_dict[client_name]['dataset_size']
        test_df_client = client_info_dict[client_name]['test_results']
        test_mae_client = test_df_client['test_mae'].iloc[0]
        test_mae_overall += test_mae_client * n_client / n_sum_clients
    if save:
        with open(os.path.join(workspace_path_server, 'overall_test_mae.txt'), 'w') as txt_file:
            txt_file.write(f'Overall test MAE: {test_mae_overall}')


def create_test_scatterplot(true_pred_test_df, client_name, workspace_path_client):
    # Create scatterplot (for client only)
    test_mae = mean_absolute_error(true_pred_test_df['true'], true_pred_test_df['pred'])
    r_true_pred, p_true_pred = stats.pearsonr(true_pred_test_df['true'], true_pred_test_df['pred'])

    fig, ax = plt.subplots()
    ax.scatter(x=true_pred_test_df['true'], y=true_pred_test_df['pred'])
    ax.set_title(
        f'Test performance {client_name}:\nMAE: {test_mae:.2f}, r: {r_true_pred:.2f} (p: {p_true_pred:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    plt.savefig(os.path.join(workspace_path_client, f'scatterplot_true_pred_test_{client_name}.png'))


def create_test_result_df(true_labels_test, pred_labels_test, workspace_path_client, save=True):
    # Save predictions and ground truth to workspace
    true_pred_test_df = pd.DataFrame({'true': true_labels_test, 'pred': pred_labels_test})
    if save:
        true_pred_test_df.to_csv(os.path.join(workspace_path_client, f'true_pred_test.csv'))

    return true_pred_test_df


def send_test_df_to_server(true_pred_test_df, test_loss, client_name, workspace_path_client, server_username,
                           server_password, server_ip_address, workspace_path_server):
    test_mae = mean_absolute_error(true_pred_test_df['true'], true_pred_test_df['pred'])
    r_true_pred, p_true_pred = stats.pearsonr(true_pred_test_df['true'], true_pred_test_df['pred'])

    # Send results to server
    print('==> Sending test results to server...')
    stat_sw_true, p_sw_true = stats.shapiro(true_pred_test_df['true'])
    stat_sw_pred, p_sw_pred = stats.shapiro(true_pred_test_df['pred'])
    test_df = pd.DataFrame({
        'test_loss': [test_loss],
        'test_mae': [test_mae],
        'r_true_pred': [r_true_pred],
        'p_true_pred': [p_true_pred],
        'true_mean': [true_pred_test_df['true'].describe()['mean']],
        'true_sd': [true_pred_test_df['true'].describe()['std']],
        'true_skewness': [stats.skew(true_pred_test_df['true'])],
        'true_kurtosis': [stats.kurtosis(true_pred_test_df['true'])],
        'true_shapiro-wilk_stat': [stat_sw_true],
        'true_shapiro-wilk_p': [p_sw_true],
        'pred_mean': [true_pred_test_df['pred'].describe()['mean']],
        'pred_sd': [true_pred_test_df['pred'].describe()['std']],
        'pred_skewness': [stats.skew(true_pred_test_df['pred'])],
        'pred_kurtosis': [stats.kurtosis(true_pred_test_df['pred'])],
        'pred_shapiro-wilk_stat': [stat_sw_pred],
        'pred_shapiro-wilk_p': [p_sw_pred]
    })
    test_df_path = os.path.join(workspace_path_client, f'{client_name}_test_results.csv')
    test_df.to_csv(test_df_path, index=False)
    send_file(server_ip_address, server_username, server_password, test_df_path, workspace_path_client,
              workspace_path_server)
