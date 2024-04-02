import numpy as np
import pandas as pd
import glob
import os

import argparse

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
    "max_roidice_r",
    "max_roidice_c",
    "max_roidice_roi",
    "max_lamb",
    "max_true_r",
    "max_true_c",
    "max_true_roi",
    "seed",
    "num_trajectories",
    "elapsed_time",
]


def is_runs(args):
    check_list = np.ones(len(keys))
    if not args.basic_rl:
        check_list[6:9] = 0
    if not args.unconstrained:
        check_list[9:12] = 0
    if not args.vanilla_constrained:
        check_list[12:15] = 0
    if not args.conservative_constrained:
        check_list[15:18] = 0
    if not args.roidice:
        check_list[18:21] = 0
        check_list[22:25] = 0
    if not args.max_roidice:
        check_list[25:32] = 0

    return check_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='save file name')
    parser.add_argument('--basic_rl', type=int, default=0)
    parser.add_argument('--unconstrained', type=int, default=0)
    parser.add_argument('--vanilla_constrained', type=int, default=0)
    parser.add_argument('--conservative_constrained', type=int, default=0)
    parser.add_argument('--roidice', type=int, default=0)
    parser.add_argument('--max_roidice', type=int, default=0)
    args = parser.parse_args()

    file_list = sorted(glob.glob("./results/*.csv"))
    new_keys = []
    check_list = is_runs(args)
    for i, k in enumerate(keys):
        if check_list[i]:
            new_keys.append(f"{k}_mean")
            new_keys.append(f"{k}_std")

    stats = {k: [] for k in new_keys}
    for file in file_list:
        df = pd.read_csv(file)
        for i, k in enumerate(keys):
            if check_list[i]:
                mean, std = np.mean(df[k]), np.std(df[k])
                stats[f"{k}_mean"].append(mean)
                stats[f"{k}_std"].append(std)

    tot_df = pd.DataFrame.from_dict(stats)
    tot_df.index = [s[10:-4] for s in file_list]
    os.makedirs("./results/tot/", exist_ok=True)
    tot_df.to_csv(f"./results/tot/{args.name}.csv", index=True)