 # See what's stored



























# import os
# from pathlib import Path

# TRAIN_FOLDER = Path(r"C:\Users\daims\Desktop\dissertation\code\data\Training")
# print(f"Training folder exists: {TRAIN_FOLDER.exists()}")
# if TRAIN_FOLDER.exists():
#     subjects = [child.name for child in TRAIN_FOLDER.iterdir() if child.is_dir()]
#     print(f"Subject folders found: {subjects}")
#     print(f"Total subject count: {len(subjects)}")
# else:
#     print("Training folder does not exist!")
    
# import pandas as pd

# INFO_FILE = Path(r"C:\Users\daims\Desktop\dissertation\code\211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
# print(f"CSV file exists: {INFO_FILE.exists()}")

# if INFO_FILE.exists():
#     df = pd.read_csv(INFO_FILE)
#     print(f"CSV has {len(df)} rows")
#     print(f"Column names: {df.columns.tolist()}")
#     print(f"Available centers: {df['Centre'].unique()}")
#     print(f"Available pathologies: {df['Pathology'].unique()}")
    
#     # Check for your specific center
#     vdh_rows = df[df['Centre'] == "Vall d'Hebron"]
#     print(f"Vall d'Hebron rows: {len(vdh_rows)}")
    
#     # Check pathology filter
#     valid_pathologies = df[df['Pathology'].isin(['NOR', 'HCM'])]
#     print(f"Rows with NOR/HCM pathologies: {len(valid_pathologies)}")

# # Check if subjects in folder match CSV entries
# if TRAIN_FOLDER.exists() and INFO_FILE.exists():
#     folder_subjects = set([child.name for child in TRAIN_FOLDER.iterdir() if child.is_dir()])
#     csv_subjects = set(df['External code'].tolist())
    
#     print(f"Subjects in folder: {len(folder_subjects)}")
#     print(f"Subjects in CSV: {len(csv_subjects)}")
#     print(f"Matching subjects: {len(folder_subjects.intersection(csv_subjects))}")
    
#     # Show some examples
#     print(f"First 5 folder subjects: {list(folder_subjects)[:5]}")
#     print(f"First 5 CSV subjects: {list(csv_subjects)[:5]}")
    
# ACDC_TRAIN_PATH = Path(r"C:\Users\daims\Desktop\dissertation\code\data\ACDC\Training")
# print(f"ACDC Training folder exists: {ACDC_TRAIN_PATH.exists()}")

# if ACDC_TRAIN_PATH.exists():
#     acdc_subjects = [child.name for child in ACDC_TRAIN_PATH.iterdir() if child.is_dir()]
#     print(f"ACDC subjects found: {acdc_subjects}")
    
#     # Check a sample subject's Info.cfg
#     if acdc_subjects:
#         sample_subject = acdc_subjects[0]
#         info_path = ACDC_TRAIN_PATH / sample_subject / 'Info.cfg'
#         print(f"Sample Info.cfg exists: {info_path.exists()}")
        
#         if info_path.exists():
#             with open(info_path, 'r') as f:
#                 content = f.read()
#                 print(f"Sample Info.cfg content:\n{content}")