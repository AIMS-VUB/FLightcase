"""
Utilities related to loading data
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import torch
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, df, colnames_dict):
        """
        :param df: pd DataFrame.
        :param colnames_dict: dict, k: data type, v: column name
        """

        self.df = df
        self.colname_label = colnames_dict['label']
        path_colnames = []
        for k, v in colnames_dict.items():
            if k not in ['label', 'id', 'session']:
                path_colnames.append(v)
        self.path_colnames = path_colnames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row
        row = self.df.iloc[idx]

        # Prepare input data. Collect in list.
        neuro_data_list = []
        for path_colname in self.path_colnames:
            neuro_data_path = row[path_colname]
            neuro_data_prepped = prep_neuro_data(neuro_data_path, path_colname)
            neuro_data_list.append(neuro_data_prepped)

        # Prepare label
        label = row[self.colname_label]
        label = torch.tensor(label, dtype=torch.float)

        return neuro_data_list, label


def prep_neuro_data(neuro_data_path, path_colname):
    """
    This function prepares neuro data
    :param neuro_data_path: str, path to neuro data
    :param path_colname: str, column name of column containing paths for the neuro data of interest
    :return: neuro data prepared as input for network
    """
    niftis_3d_suffices = ['T1w', 'FLAIR', 'T2w', 'PDw', 'T2starw', 'angio', 'inplaneT1', 'inplaneT2']
    if any(path_colname.endswith(suffix) for suffix in niftis_3d_suffices):
        return img_path_to_tensor(neuro_data_path)
    else:
        raise ValueError(f'No handler specified for {path_colname} data')


def prepare_participants_df(bids_root_path, colnames_dict, subject_sessions, modalities_dict, derivative_name):
    """ Prepare participants dataframe for machine learning

    :param bids_root_path: str, path to BIDS root directory
    :param colnames_dict: dict, necessary keys: "id", "img_path" and "label"
    :param subject_sessions: dict, key: subject_id (str), val: sessions (list) to take into account
    :param modalities_dict: dict, k: modality (str, e.g. "anat"), val: filename tags (list, e.g. ['T1w.nii.gz'])
    :param derivative_name: str, name of the subfolder in the derivatives folder if applicable
    :return: pd DataFrame
    """

    # Load participants dataframe and preprocess
    df_path = os.path.join(bids_root_path, 'participants.tsv')
    df = pd.read_csv(df_path, sep='\t')

    # Extract subject ids and sessions if specified
    df = extract_subject_sessions(df, colnames_dict, subject_sessions)

    # Derive paths to neuroimaging data and check whether the files exist
    df, path_cols = add_neuro_data_paths(df, modalities_dict, bids_root_path, colnames_dict, derivative_name)
    df = filter_on_data_availability(df, path_cols)

    # Update colname_dict (k = v)
    for path_col in path_cols:
        colnames_dict[path_col] = path_col

    return df, colnames_dict


def extract_subject_sessions(df, colnames_dict, subject_sessions):
    """
    Extract subjects and sessions from data frame if specified

    :param df: pd dataframe
    :param colnames_dict: dict, key: column name type (str, e.g. "id"), val: column name (str, e.g. "subject_id_BIDS")
    :param subject_sessions: dict, key: subject_id (str), val: sessions (list)
    :return: pd DataFrame
    """
    sub_df = pd.DataFrame()
    if subject_sessions is not None:
        for subject, sessions in subject_sessions.items():
            df_subses = df[(df[colnames_dict['id'] == subject]) & (df[colnames_dict['session'].isin(sessions)])]
            sub_df = pd.concat([sub_df, df_subses])
    else:
        sub_df = df
    return sub_df.reset_index(drop=True)


def add_neuro_data_paths(df, modalities_dict, bids_root_path, colnames_dict, derivative_name):
    """
    This function adds columns with the paths to the neuro data

    :param df: pd dataframe
    :param modalities_dict: dict, key: modality (str, e.g. "anat"), val: sub-modality suffix (str, e.g. "T1w.nii.gz")
    :param bids_root_path: str, path to BIDS root directory
    :param colnames_dict: dict, key: column name type (str, e.g. "id"), val: column name (str, e.g. "subject_id_BIDS")
    :param derivative_name: str, name of the subfolder in the derivatives folder if applicable
    :return: [1] pd DataFrame enriched with neuro data paths, [2] newly added column names
    """
    new_cols = []
    for data_type, filename_tags in modalities_dict.items():
        for filename_tag in filename_tags:
            new_col = f'path_{filename_tag.split(".")[0]}'
            new_cols.append(new_col)
            id_col = colnames_dict['id']
            path_list = []
            for i, row in df.iterrows():
                path_items_dict = {
                    'bids_root_path': bids_root_path,
                    'sub_id': row[id_col],
                    'data_type': data_type,
                    'filename_tag': filename_tag,
                    'derivative_name': derivative_name
                }
                if 'session' in colnames_dict.keys():
                    path_items_dict['session'] = row[colnames_dict['session']]
                else:
                    path_items_dict['session'] = None
                path_list.append(create_path(path_items_dict))
            df[new_col] = path_list
    return df, new_cols


def create_path(path_items_dict):
    """
    Creates path to the neuro data

    :param path_items_dict: dict, contains all info to create the path
    :return: str, path
    """

    bids_root_path = path_items_dict['bids_root_path']
    sub_id = path_items_dict['sub_id']
    session = path_items_dict['session']
    data_type = path_items_dict['data_type']
    filename_tag = path_items_dict['filename_tag']
    derivative_name = path_items_dict['derivative_name']

    if session is not None:
        filename = f'{sub_id}_{session}_{filename_tag}'
        if derivative_name is not None:
            path = os.path.join(bids_root_path, 'derivatives', derivative_name, sub_id, session, data_type, filename)
        else:
            path = os.path.join(bids_root_path, sub_id, data_type, session, filename)
    else:
        filename = f'{sub_id}_{filename_tag}'
        if derivative_name is not None:
            path = os.path.join(bids_root_path, 'derivatives', derivative_name, sub_id, data_type, filename)
        else:
            path = os.path.join(bids_root_path, sub_id, data_type, filename)
    return path


def filter_on_data_availability(df, path_cols):
    """
    This function filters out rows with neuro data paths that do not exist in the BIDS dataset

    :param df: pd DataFrame
    :param path_cols: list, column names of columns containing neuro data paths
    :return: pd DataFrame
    """
    sub_df = pd.DataFrame()
    for i, row in df.iterrows():
        keep_row = True
        for path_col in path_cols:
            if not os.path.exists(row[path_col]):
                keep_row = False
        if keep_row:
            sub_df = pd.concat([sub_df, row.to_frame().T])
    return sub_df.reset_index(drop=True)


def split_data(df, colnames_dict, train_fraction, val_fraction, test_fraction, output_root_path=None,
               train_test_random_state=42, train_val_random_state=42):
    """ Create train, validation and test dataframes

    :param df: pd DataFrame
    :param colnames_dict: dict, necessary keys: "id", "img_path" and "label"
    :param train_fraction: float, fraction of the dataset for training
    :param val_fraction: float, fraction of the dataset for validation
    :param test_fraction: float, fraction of the dataset for testing
    :param output_root_path: str, path to output root directory
    :param train_test_random_state: int, random state for split of total data in train and test data
    :param train_val_random_state: int, random state for split of train data in train and val data
    :return: dict, paths to train, validation and test dataframes
    """

    # Check if fractions total to 1
    if train_fraction + val_fraction + test_fraction != 1:
        raise ValueError('Please make sure that the sum of all fractions equals 1')

    # Splits defined on subject level. If heterogeneity in n sessions per subject, actual fractions might deviate
    subject_ids = df[colnames_dict.get('id')].unique()

    # Split in train, test and validation
    train_ids, test_ids = train_test_split(subject_ids, random_state=train_test_random_state,
                                           test_size=test_fraction)
    train_ids, val_ids = train_test_split(train_ids, random_state=train_val_random_state,
                                          test_size=val_fraction/(train_fraction + val_fraction))

    # Extract dataframes
    # Note: if multiple sessions per subject, all sessions are collected in the same dataframe
    train_df = df[df[colnames_dict.get('id')].isin(train_ids)][list(colnames_dict.values())]
    val_df = df[df[colnames_dict.get('id')].isin(val_ids)][list(colnames_dict.values())]
    test_df = df[df[colnames_dict.get('id')].isin(test_ids)][list(colnames_dict.values())]

    # Write to tsv files
    if output_root_path is not None:
        tsv_dir_path = os.path.join(output_root_path, 'train_val_test_dfs')
        if not os.path.exists(tsv_dir_path):
            os.mkdir(tsv_dir_path)
        train_df.to_csv(os.path.join(tsv_dir_path, 'train.tsv'), sep='\t', index=False)
        val_df.to_csv(os.path.join(tsv_dir_path, 'validation.tsv'), sep='\t', index=False)
        test_df.to_csv(os.path.join(tsv_dir_path, 'test.tsv'), sep='\t', index=False)

    return train_df, val_df, test_df


def get_data_loader(df, purpose, colnames_dict, batch_size, return_n=False, print_n=False):
    """ Get data loader from dataframe

    :param df: pd DataFrame
    :param purpose: str, choose from ['train', 'validation', 'test']
    :param colnames_dict: dict, necessary keys: "id", "img_path" and "label"
    :param batch_size: int, batch size
    :param return_n: bool, return the number of subjects as an additional output?
    :param print_n: bool, print the number of subjects?
    :return: Torch DataLoader (train, validation and test), [optional] int, number of subjects
    """

    dataset = DataSet(df, colnames_dict=colnames_dict)
    loader = DataLoader(dataset, batch_size=batch_size)
    n = len(dataset)

    if print_n:
        print(f'Number of {purpose} scans: {n}')

    if return_n:
        return loader, n
    else:
        return loader


def img_path_to_tensor(image_path):
    """ Prepare image for input to net

    :param image_path: str, path to image
    :return: tensor, prepared image
    """

    img = nib.load(image_path).get_fdata()                                      # Load preprocessed NIFTI
    img_tensor = torch.Tensor(img)                                              # Convert to torch Tensor
    img_tensor = torch.unsqueeze(img_tensor, dim=0)                             # Add dimension
    img_tensor = (img_tensor - torch.mean(img_tensor))/torch.std(img_tensor)    # Normalise tensor

    return img_tensor
