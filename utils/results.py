"""
Functions related to results calculation and integration
"""

import os


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
