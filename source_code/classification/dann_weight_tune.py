#!/usr/bin/env python3
import yaml
import subprocess
import shutil
import sys
from pathlib import Path

def main():
    # Paths
    root = Path(__file__).resolve().parent.parent
    config_path = root / "classification\config.yaml"
    backup_path = root / "classification\config.yaml.bak"

    # 1. Backup original config
    print(f"Backing up {config_path} → {backup_path}")
    shutil.copy(config_path, backup_path)

    # 2. Load config once
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 3. Sweep weights
    for w in [0.1, 1, 10]:
        print(f"\n=== Running FedDANN with dann_weight = {w} ===")
        config["dann_weight"] = w

        # Write modified config
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"  Updated config.yaml with dann_weight: {w}")

        # Experiment name
        exp_name = f"FedDANN_w{w}"
        cmd = [sys.executable, str(root / r"classification\train.py"), "--exp", exp_name]
        print("  Launching:", " ".join(cmd))

        # Run training
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: training failed for weight={w}")
            break

    # 4. Restore original config
    print(f"\nRestoring original config from {backup_path} → {config_path}")
    shutil.move(backup_path, config_path)
    print("Sweep complete.")

if __name__ == "__main__":
    main()
