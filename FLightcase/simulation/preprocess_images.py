"""
Preprocess T1w brain images
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import pathlib
import argparse
import numpy as np
import nibabel as nib
import torch.nn
from matplotlib.image import imsave
from pre_process import preprocess


def preprocess_BIDS_dataset(input_root_path, output_root_path, skull_strip, use_gpu, BA_submodule_path,
                            output_resolution=None):
    """ Preprocess T1w images of a BIDS dataset

    :param input_root_path: str, absolute path to BIDS root directory
    :param output_root_path: str, absolute path to output root directory
    :param skull_strip: bool, perform skull-stripping?
    :param output_resolution: list/tuple, dims to resample to
    :param use_gpu: bool, use GPU?
    """

    for root, dirs, files in os.walk(input_root_path):
        for file in files:
            # Do not process images in "derivatives" sub-folder of the BIDS tree
            if (file.endswith('_FLAIR.nii.gz') or file.endswith('_T1w.nii.gz')) and f'{os.sep}derivatives{os.sep}' not in root:
                input_path = os.path.join(root, file)
                output_path = input_path.replace(input_root_path, output_root_path + os.sep)

                # Check if output NIfTI is already present. If not, create destination folder path and preprocess.
                if not os.path.exists(output_path):
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    print(f'--> Start preprocessing {input_path}')
                    print(f'--> Output destination: {output_path}')
                    preprocess_t1w_image(input_path=input_path, output_path=output_path, use_gpu=use_gpu,
                                         skull_strip=skull_strip, BA_submodule_path=BA_submodule_path,
                                         output_resolution=output_resolution)


def preprocess_t1w_image(input_path, output_path, skull_strip, BA_submodule_path, use_gpu, output_resolution=None):
    """ Preprocess T1w image

    :param input_path: str, absolute path to t1w image
    :param output_path: str, absolute path to output image
    :param skull_strip: bool, perform skull-stripping?
    :param BA_submodule_path: str, absolute path to parent directory of the BrainAge submodule
    :param output_resolution: list/tuple, dims to downsample to
    :param use_gpu: bool, use GPU?
    """

    # Change working directory to the BA_submodule submodule
    os.chdir(BA_submodule_path)

    # Preprocess but save separately (function saves a 4D array to a NIfTI file)
    processed_array_4D = preprocess(input_path=input_path, use_gpu=use_gpu, skull_strip=skull_strip, register=True,
                                    project_name='preproc_for_simulation')
    if processed_array_4D is not None:
        processed_array_3D = processed_array_4D[0, :, :, :]
        # np.asanyarray added with updating Python packages and making Wood pipeline compatible with it
        new_image = nib.Nifti1Image(np.asanyarray(processed_array_3D), np.eye(4))
        if output_resolution is not None:
            new_image = resample_3d(new_image, output_resolution)
            print(new_image.get_fdata().shape)
        nib.save(new_image, output_path)


def resample_3d(nifti_3d, output_resolution):
    """
    This function resamples a 3D NIfTI image to the desired resolution
    """
    # Convert NIfTI image to torch tensor and add N and C dimension
    new_image = torch.tensor(nifti_3d.get_fdata()).unsqueeze(dim=0).unsqueeze(dim=0)

    # Define the downscale function and apply to image
    # Source: https://discuss.pytorch.org/t/info-need-how-resize-3d-volumetric-data/150944/2
    downscale_func = torch.nn.Upsample(output_resolution)
    new_image = downscale_func(new_image)

    # Take last 3 dimensions and convert back to NIfTI
    new_image = nib.Nifti1Image(new_image.squeeze(dim=0).squeeze(dim=0).numpy(), np.eye(4))

    return new_image


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
        prog='Preprocess T1w images',
        description='This program preprocesses T1w images'
    )
    parser.add_argument('--dataset_root_path', type=str, help='Absolute path to dataset root', required=True)
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU?')
    parser.add_argument('--preprocessing_name', type=str, required=True, help='Preprocessing pipeline description')
    parser.add_argument('--skull_strip', action='store_true', help='Perform skull-stripping?')
    parser.add_argument('--output_resolution', nargs='+', default=[130, 130, 130], help='Resolution of output image')

    # Parse arguments
    args = parser.parse_args()
    dataset_root_path = args.dataset_root_path
    use_gpu = args.use_gpu
    preprocessing_name = args.preprocessing_name
    skull_strip = args.skull_strip
    output_resolution = [int(i) for i in args.output_resolution]

    # Get absolute path to BrainAge submodule
    # Source: https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
    script_parent_dir_path = str(pathlib.Path(__file__).parent.resolve())
    BA_submodule_path = os.path.join(script_parent_dir_path, 'BrainAge_submodule')

    # Preprocess dataset
    preprocessed_output_root_path = os.path.join(dataset_root_path, 'derivatives', preprocessing_name)
    preprocess_BIDS_dataset(input_root_path=dataset_root_path,
                            output_root_path=preprocessed_output_root_path,
                            skull_strip=skull_strip,
                            use_gpu=use_gpu,
                            BA_submodule_path=BA_submodule_path,
                            output_resolution=output_resolution)

    # Get images of brain slices
    print('Getting images of brain slices...')
    for root, dirs, files in os.walk(preprocessed_output_root_path):
        for file in files:
            if file.endswith('.nii.gz'):
                get_brain_slice_images(nifti_path=os.path.join(root, file))
