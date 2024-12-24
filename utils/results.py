"""
Functions related to results calculation and integration
"""

import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def update_avg_val_loss(client_info_dict, val_loss_avg, fl_round):
    """ Update the average validation loss

    :param client_info_dict: dict, key: client name, value: client info
    :param val_loss_avg: float, average validation loss
    :param fl_round: int, the federated learning round
    :return: float, updated validation loss
    """
    for client_name in client_info_dict.keys():
        train_results_client_df = client_info_dict[client_name][f'round_{fl_round}']['train_results']
        val_loss_avg += train_results_client_df['val_loss'].mean() / len(client_info_dict)
    return val_loss_avg


def calculate_overall_test_mae(client_info_dict, workspace_path_server, save=True):
    """ Calculate overall test MAE

    :param client_info_dict: dict, key: client name, value: client info
    :param workspace_path_server: str, path to server workspace
    :param save: bool, save the overall MAE to a txt file?
    :return: float
    """
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
    return test_mae_overall


def create_test_scatterplot(true_pred_test_df, client_name, workspace_path_client):
    """ Create scatterplot of test data

    :param true_pred_test_df: pd DataFrame, true and predicted test results
    :param client_name: str, client name
    :param workspace_path_client: str, path to client workspace
    """
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


def create_test_true_pred_df(true_labels_test, pred_labels_test, workspace_path_client, save=True):
    """ Create dataframe containing true and predicted test results

    :param true_labels_test: list, true labels
    :param pred_labels_test: list, predicted labels
    :param workspace_path_client: str, path to client workspace
    :param save: bool, save dataframe?
    """
    # Save predictions and ground truth to workspace
    true_pred_test_df = pd.DataFrame({'true': true_labels_test, 'pred': pred_labels_test})
    if save:
        true_pred_test_df.to_csv(os.path.join(workspace_path_client, f'true_pred_test.csv'))

    return true_pred_test_df


def create_test_df_for_server(true_pred_test_df, test_loss):
    """ Creates test dataframe with aggregate results for server

    :param true_pred_test_df: pd DataFrame, true and predicted test results
    :param test_loss: float, test loss
    :return: pd DataFrame
    """
    test_mae = mean_absolute_error(true_pred_test_df['true'], true_pred_test_df['pred'])
    r_true_pred, p_true_pred = stats.pearsonr(true_pred_test_df['true'], true_pred_test_df['pred'])
    stat_sw_true, p_sw_true = stats.shapiro(true_pred_test_df['true'])
    stat_sw_pred, p_sw_pred = stats.shapiro(true_pred_test_df['pred'])
    test_df_for_server = pd.DataFrame({
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
    return test_df_for_server
