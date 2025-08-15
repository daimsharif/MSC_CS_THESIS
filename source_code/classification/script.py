#!/usr/bin/env python3
import os
import pandas as pd

# Column names from CSV
CASE_COL = "External code"
VENDOR_COL = "VendorName"
PATHOLOGY_COL = "Pathology"

# Paths
CSV_PATH = "C:/Users/daims/Desktop/dissertation/code/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
TRAIN_DIR = "data/Training"
VENDOR_NAME = "Canon"

def get_cases_by_pathology(vendor_name="Canon"):
    df = pd.read_csv(CSV_PATH)
    df = df[df[VENDOR_COL].str.lower() == vendor_name.lower()]
    hcm_cases = df[df[PATHOLOGY_COL] == "HCM"][CASE_COL].astype(str).tolist()
    nor_cases = df[df[PATHOLOGY_COL] == "NOR"][CASE_COL].astype(str).tolist()
    return hcm_cases, nor_cases

def get_cases_in_folder():
    items = os.listdir(TRAIN_DIR)
    return [name for name in items if os.path.isdir(os.path.join(TRAIN_DIR, name)) or name.endswith(".nii.gz")]

def print_comparison(label, canonical_cases, folder_cases):
    print(f"\n📋 {label} cases in metadata: {len(canonical_cases)}")
    print(f"📁 Cases found in training folder: {len(folder_cases)}")

    present, missing = [], []

    for case in canonical_cases:
        matches = [x for x in folder_cases if case in x]
        if matches:
            present.append(case)
        else:
            missing.append(case)

    print(f"\n✅ Present {label} cases ({len(present)}):")
    # for p in present:
    #     print(f"  {p}")

    print(f"\n❌ Missing {label} cases ({len(missing)}):")
    for m in missing[:3]:
        print(f"  {m}")

def main():
    hcm_cases, nor_cases = get_cases_by_pathology()
    train_cases = get_cases_in_folder()

    print_comparison("HCM", hcm_cases, train_cases)
    print_comparison("NOR", nor_cases, train_cases)

if __name__ == "__main__":
    main()
