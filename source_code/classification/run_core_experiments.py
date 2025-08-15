import yaml
import subprocess
import shutil
import sys
from pathlib import Path

def run_experiment(exp_name, config_updates):
    print(f"\n=== Running: {exp_name} ===")

    # Load original config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Apply updates specific to the method
    for k, v in config_updates.items():
        # support nested dict updates (e.g., federated.isTrue)
        if isinstance(v, dict):
            for subk, subv in v.items():
                config[k][subk] = subv
        else:
            config[k] = v

    # Save modified config
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Launch train.py with --exp
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--exp", exp_name]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# # ────────────── SETUP ──────────────
# PROJECT_ROOT = Path(__file__).resolve().parent
# CONFIG_PATH  = PROJECT_ROOT / "config.yaml"
# TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

# ───────────── CONFIGURE HERE ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / r"classification\config.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / r"classification\train.py"

EXPERIMENTS = [
    # {
    #     "name": "Centralized_LCO",
    #     "config": {
    #         "federated": {"isTrue": False},
    #         "method": None,
    #     }
    # },
    
    {
        "name": "FedBN_testing_code",
        "config": {
            "federated": {"isTrue": True, "type": "FedBN"},
            "method": None,
        }
    }
    # ,
    # {
    #     "name": "FedDANN_LCO",
    #     "config": {
    #         "federated": {"isTrue": True, "type": "FL"},
    #         "method": "feddann",
    #         "dann_weight": 0.1
    #     }
    # }
    # ,
    # {
    #     "name": "FedAvg_LCO",
    #     "config": {
    #         "federated": {"isTrue": True, "type": "FL"},
    #         "method": None,
    #     }
    # }
]

# ────────────── RUN ALL CORE EXPERIMENTS ──────────────
# Backup original config
shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix(".orig"))

try:
    for exp in EXPERIMENTS:
        run_experiment(exp["name"], exp["config"])
finally:
    shutil.move(CONFIG_PATH.with_suffix(".orig"), CONFIG_PATH)
    print("\n✅ All core experiments completed. Original config restored.")


# import yaml
# import subprocess
# import shutil
# import sys
# from pathlib import Path

# def run_experiment(method, federated, exp_name, dann_weight=None):
#     print(f"\n=== Running {method} ({'Federated' if federated else 'Centralized'}) ===")
    
#     # Load config
#     with open(CONFIG_PATH, "r") as f:
#         config = yaml.safe_load(f)
    
#     # Update key fields
#     config["method"] = method
#     config["federated"]["isTrue"] = federated

#     if federated:
#         config["federated"]["type"] = "FL" if method == "fedavg" else "FedBN"
#     else:
#         config["federated"]["type"] = None

#     if method == "feddann":
#         config["dann_weight"] = dann_weight or 1.0

#     # Save modified config
#     with open(CONFIG_PATH, "w") as f:
#         yaml.safe_dump(config, f, sort_keys=False)

#     # Run the training
#     cmd = [sys.executable, str(TRAIN_SCRIPT), "--exp", exp_name]
#     print("Launching:", " ".join(cmd))
#     subprocess.run(cmd, check=True)

# # ───────────── CONFIGURE HERE ─────────────
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# CONFIG_PATH  = PROJECT_ROOT / r"classification\config.yaml"
# TRAIN_SCRIPT = PROJECT_ROOT / r"classification\train.py"

# METHODS = [
#     ("none", False, "Centralized_LCO"),
#     ("FL",      True,  "FedAvg_LCO"),
#     ("FedBN",       True,  "FedBN_LCO"),
#     ("feddann",     True,  "FedDANN_LCO"),
# ]

# # ───────────── RUN ALL CORE EXPERIMENTS ─────────────
# # Backup original config
# shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix(".orig"))

# try:
#     for method, is_federated, exp_name in METHODS:
#         run_experiment(method, is_federated, exp_name)
# finally:
#     # Restore config
#     shutil.move(CONFIG_PATH.with_suffix(".orig"), CONFIG_PATH)
#     print("\n✅ All core experiments completed and config restored.")
