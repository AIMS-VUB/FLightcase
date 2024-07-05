"""
Preprocess T1w and FLAIR brain images
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import sys
import pathlib
import argparse
import numpy as np
import nibabel as nib
from matplotlib.image import imsave
from bids_validator import BIDSValidator
from submodules.Wood_2022.pre_process import preprocess


def dataset_is_bids(path_to_root):
    """ Check if the dataset is BIDS-compliant

    :param path_to_root: str, path to the root directory of the dataset
    :return: bool
    """

    validator = BIDSValidator()
    dataset_bids_bool = True
    for root, dirs, files in os.walk(path_to_root):
        for file in files:
            file_path = os.path.join(root, file)
            if not validator.is_bids(file_path.replace(path_to_root, os.sep)):
                print(file_path.replace(path_to_root, os.sep))
                print(f'File not bids: {file_path}')
                dataset_bids_bool = False
    return dataset_bids_bool


def preprocess_BIDS_dataset(input_root_path, output_root_path, skull_strip, use_gpu, Wood_2022_path):
    """ Preprocess T1w and FLAIR images of a BIDS dataset

    :param input_root_path: str, absolute path to BIDS root directory
    :param output_root_path: str, absolute path to output root directory
    :param skull_strip: bool, perform skull-stripping?
    :param use_gpu: bool, use GPU?
    """

    for root, dirs, files in os.walk(input_root_path):
        for file in files:
            # Do not process images in "derivatives" sub-folder of the BIDS tree
            if (file.endswith('_T1w.nii.gz') or file.endswith('_FLAIR.nii.gz')) and f'{os.sep}derivatives{os.sep}' not in root:
                input_path = os.path.join(root, file)
                output_path = input_path.replace(input_root_path, output_root_path)

                # Check if output NIfTI is already present. If not, create destination folder path and preprocess.
                if not os.path.exists(output_path):
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    print(f'--> Start preprocessing {input_path}')
                    print(f'--> Output destination: {output_path}')
                    preprocess_image(input_path=input_path, output_path=output_path, use_gpu=use_gpu,
                                         skull_strip=skull_strip, Wood_2022_path=Wood_2022_path)


def preprocess_image(input_path, output_path, skull_strip, Wood_2022_path, use_gpu):
    """ Preprocess image

    :param input_path: str, absolute path to image
    :param output_path: str, absolute path to output image
    :param skull_strip: bool, perform skull-stripping?
    :param Wood_2022_path: str, absolute path to parent directory of this script
    :param use_gpu: bool, use GPU?
    """

    # Change working directory to the Wood_2022 submodule
    os.chdir(Wood_2022_path)

    # Preprocess but save separately (function saves a 4D array to a NIfTI file)
    processed_array_4D = preprocess(input_path=input_path, use_gpu=use_gpu, skull_strip=skull_strip, register=True,
                                    project_name='Wood_2022_preprocessing')
    if processed_array_4D is not None:
        processed_array_3D = processed_array_4D[0, :, :, :]
        new_image = nib.Nifti1Image(processed_array_3D, np.eye(4))
        nib.save(new_image, output_path)


def get_brain_slice_images(nifti_path, output_dir_path=None):
    """ Get PNG images of brain slices in all directions of a 3D NIfTI

    :param nifti_path: str, path to NIfTI file
    :param output_dir_path: str, path to output directory. If None, this will be the path to the NIfTI parent directory.
    """
    # Load NIfTI and check if it is 3D
    img = nib.load(nifti_path).get_fdata()
    if img.ndim == 3:
        # Get slices in all directions
        slice_1 = img[img.shape[0]//2, :, :]
        slice_2 = img[:, img.shape[1]//2, :]
        slice_3 = img[:, :, img.shape[2]//2]

        # Save images
        if output_dir_path is None:
            output_dir_path = os.path.dirname(nifti_path)
        for i, img_slice in enumerate([slice_1, slice_2, slice_3]):
            filename = os.path.basename(nifti_path).replace('.nii.gz', f'_slice_{i+1}.png')
            imsave(os.path.join(output_dir_path, filename), img_slice, cmap='gray')

    else:
        raise ValueError(f'{nifti_path} does not have exactly 3 dimensions')


if __name__ == "__main__":
    # Define command line options
    parser = argparse.ArgumentParser(
        prog = 'Preprocess T1w and FLAIR images',
        description = 'This program preprocesses T1w and FLAIR images'
    )
    parser.add_argument('--dataset_root_path', type=str, help='Absolute path to dataset root', required=True)
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU?')
    parser.add_argument('--preprocessing_name', type=str, required=True, help='Preprocessing pipeline description')
    parser.add_argument('--skull_strip', action='store_true', help='Perform skull-stripping?')

    # Parse arguments
    args = parser.parse_args()
    dataset_root_path = args.dataset_root_path
    use_gpu = args.use_gpu
    preprocessing_name = args.preprocessing_name
    skull_strip = args.skull_strip

    # Stop script if the dataset is not BIDS compliant
    if not dataset_is_bids(dataset_root_path):
        sys.exit('Please make sure the dataset is BIDS compliant.')

    # Get absolute path to Wood_2022 submodule
    # Source: https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
    script_parent_dir_path = str(pathlib.Path(__file__).parent.resolve())
    Wood_2022_path = os.path.join(script_parent_dir_path, 'submodules', 'Wood_2022')

    # Preprocess dataset
    preprocessed_output_root_path = os.path.join(dataset_root_path, 'derivatives', preprocessing_name)
    preprocess_BIDS_dataset(input_root_path=dataset_root_path,
                            output_root_path=preprocessed_output_root_path,
                            skull_strip=skull_strip,
                            use_gpu=use_gpu,
                            Wood_2022_path=Wood_2022_path)

    # Get images of brain slices
    print('Getting images of brain slices...')
    for root, dirs, files in os.walk(preprocessed_output_root_path):
        for file in files:
            if file.endswith('.nii.gz'):
                get_brain_slice_images(nifti_path=os.path.join(root, file))
