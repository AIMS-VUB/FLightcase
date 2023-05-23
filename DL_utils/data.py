"""
Utilities related to loading data
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import torch
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, df, colnames_dict):
        """
        :param df: pd DataFrame. Necessary columns: "subject_id", "nifti_path" and the label_colname
        :param colnames_dict: dict, necessary keys: "id", "img_path" and "label"
        """

        self.df = df
        self.img_path_colname = colnames_dict.get('img_path')
        self.label_colname = colnames_dict.get('label')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get subject info
        subject_info = self.df.iloc[idx]
        nifti_path = subject_info[self.img_path_colname]

        # Prepare image
        img_tensor = img_to_tensor(nifti_path)

        # Load other data
        label = subject_info[self.label_colname]
        label = torch.tensor(label, dtype=torch.float)

        return img_tensor, label


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

    subject_ids = df[colnames_dict.get('id')]

    # Split in train, test and validation
    train_ids, test_ids = train_test_split(subject_ids, random_state=train_test_random_state,
                                           test_size=test_fraction)
    train_ids, val_ids = train_test_split(train_ids, random_state=train_val_random_state,
                                          test_size=val_fraction/(train_fraction + val_fraction))

    # Extract dataframes
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


def img_to_tensor(image_path):
    """ Prepare preprocessed image for input to net

    :param image_path: str, path to T1 NIfTI
    :return: tensor, prepared image
    """

    img = nib.load(image_path).get_fdata()                                      # Load preprocessed NIFTI
    img_tensor = torch.Tensor(img)                                              # Convert to torch Tensor
    img_tensor = torch.unsqueeze(img_tensor, dim=0)                             # Add dimension
    img_tensor = (img_tensor - torch.mean(img_tensor))/torch.std(img_tensor)    # Normalise tensor

    return img_tensor
