import os
import json
import math
import subprocess
import time
from glob import glob
from itertools import product
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ------------------ CONFIG ------------------
learning_rates = [1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
seeds = [0, 1]
head_updates_per_encoder_update = [5] #, 3, 7

nb_steps = 5000
save_steps = 0

base_dirs = {
    "ilm": "runs_ilm",
    "ilmg": "runs_ilmg",
    "elm": "runs_elm",
}

val_file = "donnees/val_test/val.txt"


# ------------------ TRAINING ------------------
def launch_training(model_key):
    base_dir = base_dirs[model_key]

    if model_key == "elm":
        train_file = "donnees"
        mode = "ilm"

    elif model_key == "ilm":
        train_file = "donnees/envs"
        mode = "ilm"
    
    elif model_key == "ilmg":
        train_file = "donnees/envs"
        mode = "game"
    

    os.makedirs(base_dir, exist_ok=True)
    for head_updates in head_updates_per_encoder_update:
        for lr in learning_rates:
            for seed in seeds:
                    exp_name = f"model_{base_dir}_lr{lr}_seed{seed}_steps{nb_steps}"
                    out_dir = os.path.join(base_dir, exp_name)
                    os.makedirs(out_dir, exist_ok=True)

                    print(f"\n Lancement de l'entraînement: {exp_name}")

                    cmd = [
                        "python3", "-m", "torch.distributed.run",
                        "--nproc_per_node=1",
                        "--master_port", "29501",
                        "run_invariant_mlm.py",
                        "--model_name_or_path", "distilbert-base-uncased",
                        "--mode", mode,
                        "--train_file", train_file,
                        "--validation_file", val_file,
                        "--do_train", "--do_eval",
                        "--output_dir", out_dir,
                        "--overwrite_output_dir",
                        "--nb_steps_model_saving", str(save_steps),
                        "--max_seq_length", "128",
                        "--per_device_train_batch_size", "128",
                        "--gradient_accumulation_steps", "4",
                        "--preprocessing_num_workers", "16",
                        "--per_device_eval_batch_size", "8",
                        "--learning_rate", str(lr),
                        "--nb_steps", str(nb_steps),
                        "--seed", str(seed),
                        "--run_name", exp_name,
                        "--fp16",
                        "--half_precision_backend", "cuda_amp",
                        
                        "--evaluation_strategy", "steps",
                        "--eval_steps", "500"
                    ]
                    # "--head_updates_per_encoder_update", str(head_updates),
                        
                    print(f"[TRAIN] Launching: {exp_name}")
                    subprocess.run(cmd)
                    print(f"[TRAIN] Finished: {exp_name}\n")


if __name__ == "__main__":
    t0 = time.time()
    #launch_training("elm")
    launch_training("ilm")
    launch_training("ilmg")
    print(f"[DONE] Temps total : {round(time.time() - t0, 2)}s")
