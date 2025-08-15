import os
import copy
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
from bias_correction import n4_bias_correction
from resampling import resample
import uuid
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import logging

# --- right after your imports ---
logging.basicConfig(
    filename='bad_affines.log',
    filemode='a',               # append
    level=logging.ERROR,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


with open('classification\config.yaml') as file:
    config = yaml.safe_load(file)

DATA_PATH = config['paths']['raw_dataset']
DESTINATION_PATH = config['paths']['dataset']

data_path = Path(DATA_PATH)
dest_path = Path(DESTINATION_PATH)

TRAIN_FOLDER = data_path / 'Training' / 'Labeled'
TRAIN_FOLDER_DEST = dest_path / 'Training'

VAL_FOLDER = data_path / 'Validation'

TEST_FOLDER = data_path / 'Testing'
TEST_FOLDER_DEST = dest_path / 'Testing'

INFO_FILE = data_path.parent / '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'

# csv columns:
CODE = 'External code'
ED = 'ED'
ES = 'ES'

#
IMG_SUFFIX = '_sa.nii.gz'
LBL_SUFFIX = '_sa_gt.nii.gz'
ED_SUFFIX = '_ed.nii.gz'
ED_GT_SUFFIX = '_ed_gt.nii.gz'
ES_SUFFIX = '_es.nii.gz'
ES_GT_SUFFIX = '_es_gt.nii.gz'
info = pd.read_csv(INFO_FILE)

def orthonormalize_affine(affine: np.ndarray) -> np.ndarray:
    """
    Force the 3×3 rotation matrix in the affine to be orthonormal via SVD.
    Preserves translation.
    """
    R = affine[:3, :3]
    U, _, VT = np.linalg.svd(R)
    R_ortho = U @ VT
    new_affine = np.eye(4, dtype=affine.dtype)
    new_affine[:3, :3] = R_ortho
    new_affine[:3, 3] = affine[:3, 3]
    return new_affine


def process_case(info, subject, images_dir, images_dir_dest):
    '''Extracts ED and ES time points from 4D MRI and applies bias
    field correction
    '''
    case = info.loc[info[CODE] == subject]
    # load this image
    case_code = case[CODE].item()
    img_path = images_dir / case_code / str(case_code + IMG_SUFFIX)
    label_path = images_dir / case_code / str(case_code + LBL_SUFFIX)
    img = nib.load(img_path)
    gt = nib.load(label_path)
    ed_id = int(case[ED].item())
    es_id = int(case[ES].item())

    # create save dir if doesn't exist
    save_path = images_dir_dest / case_code
   
    # Skip if already processed
    ed_output_file = save_path / f"{case_code}_ed.nii.gz"
    es_output_file = save_path / f"{case_code}_es.nii.gz"
    if ed_output_file.exists() and es_output_file.exists():
        print(f"[SKIP] {case_code} already processed.")
        return
    
    
    
    
    if not save_path.exists():
        save_path.mkdir(parents=True)  # also create missing parents
    # for both ED and ES
    for idx, suffix, gt_suffix in zip([ed_id, es_id],
                                      [ED_SUFFIX, ES_SUFFIX],
                                      [ED_GT_SUFFIX, ES_GT_SUFFIX]):
        # extract time point
        data = img.get_fdata()[..., idx]
        lbl = gt.get_fdata()[..., idx]
        # save the ground truth right away
        lbl = nib.Nifti1Image(lbl, gt.affine)
        new_header = lbl.header
        lbl, new_spacing = resample(lbl)
        new_header['pixdim'][1:4] = new_spacing #change spacing info
        nib.save(nib.Nifti1Image(lbl, gt.affine, new_header),
                 save_path / str(case_code + gt_suffix))
        
        #ds
        # correct bias field for data and save
        temp_file = save_path / f"temp_{uuid.uuid4().hex}.nii.gz"

        # Build NIfTI image and resample
        image = nib.Nifti1Image(data, img.affine)
        new_header = image.header
        image_resampled, new_spacing = resample(image)
        new_header['pixdim'][1:4] = new_spacing  # update header

        # Save to temp file manually
        # fixed_affine = orthonormalize_affine(img.affine)
        # nib.save(nib.Nifti1Image(image_resampled, fixed_affine, new_header), str(temp_file))
        nib.save(nib.Nifti1Image(image_resampled, img.affine, new_header), str(temp_file))

        # Run N4 bias correction and save to final path
        final_path = save_path / str(case_code + suffix)
        
        # n4_bias_correction(str(temp_file), str(final_path))

        # # Optionally delete temp file
        # os.remove(temp_file)
        try:
            n4_bias_correction(str(temp_file), str(final_path))
        except RuntimeError as e:
            # Only catch the non‐orthonormal error
            if "No orthonormal definition found" in str(e):
                print(f"[AFFINE FIX] {case_code}{suffix}: retrying with orthonormalized affine")
                # Reload via NiBabel
                tmp_nii = nib.load(str(temp_file))
                arr2 = tmp_nii.get_fdata()
                hdr2 = tmp_nii.header
                # Orthonormalize and overwrite temp
                clean_aff = orthonormalize_affine(tmp_nii.affine)
                nib.save(
                    nib.Nifti1Image(arr2, clean_aff, hdr2),
                    str(temp_file)
                )
                # Retry N4 one last time, but catch that, too:
                try:
                    n4_bias_correction(str(temp_file), str(final_path))
                except RuntimeError:
                    # Log it!
                    logging.error(
                        f"FAILED AFFINE: case={case_code} slice={suffix}"
                    )
                    print(f"[ERROR] {case_code}{suffix}: still bad affine → logged and skipping")
            else:
                # Unexpected error: re‐raise
                raise
        finally:
            try:
                os.remove(str(temp_file))
            except PermissionError:
                # File still in use by SimpleITK—skip deletion for now
                print(f"[WARN] Couldn’t delete {temp_file}, will leave it be.")
            # Clean up
            # if os.path.exists(str(temp_file)):
            #     os.remove(str(temp_file))

# get all subjects from one folder
# images_dir = TRAIN_FOLDER
# images_dir_dest = TRAIN_FOLDER_DEST
# subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
# with tqdm_joblib(
#     tqdm(desc="Preprocessing", total=len(subjects), ncols=80)
# ):
#     Parallel(n_jobs=4)(
#         delayed(process_case)(info, subject, images_dir, images_dir_dest)
#         for subject in subjects
#     )

# images_dir = VAL_FOLDER
# images_dir_dest = TRAIN_FOLDER_DEST 
# subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
# Parallel(n_jobs=4)(
#     delayed(process_case)(info, subject, images_dir, images_dir_dest)
#     for subject in tqdm(subjects)
# )



images_dir = TEST_FOLDER
images_dir_dest = TEST_FOLDER_DEST
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
with tqdm_joblib(
    tqdm(desc="Preprocessing", total=len(subjects), ncols=80)
):
    Parallel(n_jobs=4)(
        delayed(process_case)(info, subject, images_dir, images_dir_dest)
        for subject in subjects
    )

