"""
Functions related to results calculation and integration
"""

import os


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
