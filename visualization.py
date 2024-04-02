import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "./results/tot/tot_out.csv"
    tot = pd.read_csv(path)

    keys = [
        "uopt_r", "uopt_c",
        "opt_r", "opt_c",
        "behav_r", "behav_c",
        "basic_r", "basic_c", "basic_roi",
        "ucdice_r", "ucdice_c", "ucdice_roi",
        "cdice_r", "cdice_c", "cdice_roi",
        "ccdice_r", "ccdice_c", "ccdice_roi",
        "roidice_r", "roidice_c", "roidice_roi",
        "target_roi",
        "true_r", "true_c", "true_roi",
        "seed",
        "num_trajectories",
        "elapsed_time",
    ]

    alpha = [0.001, 0.01, 0.1, 1.0]
    target_roi = [0.1, 1.0, 5.0, 10.0]

    plot_keys = keys[6: 21] + keys[22: 25]

    tot_arr = np.zeros((len(alpha), len(target_roi), len(plot_keys)))
    
    for i in range(3, len(tot["Unnamed: 0"]), 4):
        tot.iloc[i], tot.iloc[i-1] = tot.iloc[i-1], tot.iloc[i].copy() # swap
    
    pair_name = tot["Unnamed: 0"]

    for i in range(len(plot_keys)):
        tot_arr[:,:,i] = tot[f"{plot_keys[i]}_mean"].to_numpy().reshape(len(alpha), len(target_roi))

    # plot figure
    fig = plt.figure(figsize=(18, 16))
    _xlabel = [str(a) for a in alpha]
    _ylabel = ["REWARD", "COST", "ROI"]
    _tmp = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(len(target_roi)*3):
        plt.subplot(3, 4, i+1)
        plt.xticks(range(len(alpha)), _xlabel, rotation=45)
        plt.xlabel("alpha")
        plt.ylabel(_ylabel[_tmp[i]])
        plt.title(f"target roi {target_roi[i%len(target_roi)]}")
        for v in range(0, len(plot_keys), 3):
            y = tot_arr[:, i%len(target_roi), v+_tmp[i]]
            plt.plot(range(len(alpha)), y, 'o-', label=plot_keys[v+_tmp[i]])
        plt.legend()

    # save
    fig.savefig("./results/plots/alpha_target_roi.png")