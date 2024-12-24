"""
Functions related to tracking
"""


def print_FL_plan(FL_plan_dict):
    """ Print FL plan

    :param FL_plan_dict: dict, FL plan
    """
    print('\n========\nFL plan:\n========\n')
    for k, v in FL_plan_dict.items():
        print(f'- {k}: {v}')
    print('\n')
