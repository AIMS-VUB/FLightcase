
import os
import glob
import json
import pathlib
import readline


# Allow tab completion with input()
# Adapted from: https://stackoverflow.com/questions/6656819/filepath-autocompletion-using-users-input
def complete(text, state):
    return (glob.glob(text+'*/')+[None])[state]


readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(dict_to_write, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict_to_write, f, indent=4)


def copy_template(workspace_path, file_name):

    parent_dir_path = str(pathlib.Path(__file__).parent.resolve())
    file_path_src = os.path.join(parent_dir_path, '..', 'templates', file_name)
    file_path_dst = os.path.join(workspace_path, file_name)
    os.system(f'cp {file_path_src} {file_path_dst}')
    print(f'Copied template {file_name} to {file_path_dst}.')


def fill_or_copy(workspace_path, file_name):
    fill_in = False
    while fill_in not in ['yes', 'no']:
        fill_in = input(f'\nFill in {file_name} (no: copies template to workspace)? (yes|no): ').lower()
    if fill_in == 'yes':
        fill_in_template(workspace_path, file_name)
    else:
        copy_template(workspace_path, file_name)


def fill_in_template(workspace_path, file_name):

    # Define paths and load FL plan template
    parent_dir_path = str(pathlib.Path(__file__).parent.resolve())
    file_path_src = os.path.join(parent_dir_path, '..', 'templates', file_name)
    tmpl_dict = load_json(file_path_src)
    file_path_dst = os.path.join(workspace_path, file_name)

    # Iterate over template keys and fill
    completed_dict = {}
    for k, v in tmpl_dict.items():
        if k == "client_credentials":
            new_v = define_client_credentials()
        elif k == "modalities_to_include":
            new_v = define_modalities_dict()
        elif k == "colnames_dict":
            new_v = define_colnames_dict()
        elif k == "subject_sessions":
            new_v = define_subject_sessions()
        elif type(v) == str:
            new_v = input(f'> Specify "{k}" (hit "enter" for default: "{v}"): ')
            if new_v == '':
                new_v = v
                print(f'Choosing default value "{v}"')
        elif type(v) == int:
            new_v = input(f'> Specify "{k}" (integer, hit "enter" for default: {v}): ')
            if new_v == '':
                new_v = v
            else:
                new_v = int(new_v)
        elif type(v) == float:
            new_v = input(f'> Specify "{k}" (int/float (. (dot) decimal point), hit "enter" for default: {v}): ')
            if new_v == '':
                new_v = v
            else:
                new_v = float(new_v)
        elif v is None:
            new_v = input(f'> Specify "{k}" (hit "enter" for default: null): ')
            if new_v == '':
                new_v = v
            else:
                new_v = str(new_v)
        else:
            raise ValueError(f'Key "{k}" not recognized.')
        completed_dict[k] = new_v

    # Write dict to workspace
    write_json(completed_dict, file_path_dst)


def define_colnames_dict():
    print('Specify column name for:')
    dict_to_fill = {}
    for colname_type, default in zip(['id', 'session', 'label'], ['subject_id_BIDS', 'session_BIDS', 'label']):
        colname = input(f'> {colname_type} column name (hit "enter" for default {default}): ')
        if colname == '':
            colname = default
            print(f'Choosing default: {default}')
        dict_to_fill[colname_type] = colname
    return dict_to_fill


def define_modalities_dict():
    dict_to_fill = {}
    data_types = input(f'Data types to include (seperated by space: e.g. "anat func"): ').split(' ')
    for data_type in data_types:
        msg = f'> File suffices to include for {data_type} (seperated by space: e.g. T1w.nii.gz FLAIR.nii.gz): '
        suffices = input(msg).split(' ')
        dict_to_fill[data_type] = suffices
    return dict_to_fill


def define_client_credentials():
    n_clients = int(input(f'How many clients are in the federation (int)? '))
    dict_to_fill = {}
    for i in range(n_clients):
        client_name = input(f'> Enter name for client {i+1} (str): ')
        dict_to_fill[client_name] = {}
        for info_type in ['ip_address', 'username', 'password']:
            info = input(f'    >> Specify {info_type} for {client_name}: ')
            dict_to_fill[client_name][info_type] = info
    return dict_to_fill


def define_subject_sessions():
    specify = False
    while specify not in ['yes', 'no']:
        specify = input('Specify subjects and sessions (yes|no)? ')
    if specify == 'yes':
        dict_to_fill = {}
        completed = False
        while not completed:
            add_subject = False
            while add_subject not in ['yes', 'no']:
                add_subject = input(f'Add subject (yes|no): ')
            if add_subject == 'yes':
                subject = input('> Enter subject name: ')
                sessions = input(f'    >> Enter sessions for subject {subject} (separated by space): ').split(' ')
                dict_to_fill[subject] = sessions
            else:
                return dict_to_fill
    else:
        return None
