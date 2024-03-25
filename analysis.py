import numpy as np
import pandas as pd
import glob
import os

keys = [
    "uopt_r",
    "uopt_c",
    "opt_r",
    "opt_c",
    "behav_r",
    "behav_c",
    "basic_r",
    "basic_c",
    "basic_roi",
    "ucdice_r",
    "ucdice_c",
    "ucdice_roi",
    "cdice_r",
    "cdice_c",
    "cdice_roi",
    "ccdice_r",
    "ccdice_c",
    "ccdice_roi",
    "roidice_r",
    "roidice_c",
    "roidice_roi",
    "target_roi",
    "true_r",
    "true_c",
    "true_roi",
    "seed",
    "num_trajectories",
    "elapsed_time",
]

if __name__ == "__main__":
    file_list = sorted(glob.glob("./results/*.csv"))
    new_keys = []
    for k in keys:
        new_keys.append(f"{k}_mean")
        new_keys.append(f"{k}_std")

    stats = {k: [] for k in new_keys}
    for file in file_list:
        df = pd.read_csv(file)
        for k in keys:
            mean, std = np.mean(df[k]), np.std(df[k])
            stats[f"{k}_mean"].append(mean)
            stats[f"{k}_std"].append(std)

    tot_df = pd.DataFrame.from_dict(stats)
    tot_df.index = [s[10:-4] for s in file_list]
    os.makedirs("./results/tot/", exist_ok=True)
    tot_df.to_csv("./results/tot/tot_out.csv", index=True)