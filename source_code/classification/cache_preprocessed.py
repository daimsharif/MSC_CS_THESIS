# cache_preprocessed.py

import os, yaml
from pathlib import Path
import torchio as tio

# 1. Load your config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

HOME      = Path.home()
RAW_ROOT  = HOME / config['paths']['dataset']     # e.g. ~/…/dataset
PROC_ROOT = HOME / config['paths']['misc'] / 'proc'  # e.g. ~/…/misc/proc
LANDMARKS = HOME / config['paths']['misc'] / config['names']['histogram_landmarks']

# 2. Build deterministic pipeline
deterministic = tio.Compose([
    tio.HistogramStandardization({'mri': LANDMARKS}),
    tio.CropOrPad(tuple(config['transforms']['crop_size']), mask_name='gt'),
    tio.RescaleIntensity((0, 1)),
])

# 3. Only cache TRAINING split
split = 'Training'
in_root  = RAW_ROOT  / split
out_root = PROC_ROOT / split
out_root.mkdir(parents=True, exist_ok=True)

for center in os.listdir(in_root):
    center_in  = in_root  / center
    center_out = out_root / center
    center_out.mkdir(exist_ok=True)

    for img in center_in.glob(f"{center}_*.nii.gz"):
        gt      = center_in / f"{img.stem}_gt.nii.gz"
        out_img = center_out / img.name
        out_gt  = center_out / gt.name
        if out_img.exists() and out_gt.exists():
            continue

        subj = tio.Subject(
            mri=tio.ScalarImage(str(img)),
            gt =tio.LabelMap   (str(gt)),
        )
        proc = deterministic(subj)
        proc.mri.save(str(out_img))
        proc.gt .save(str(out_gt))
        print(f"Cached {split}/{center}/{img.name}")
