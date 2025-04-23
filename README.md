# ROIDICE: Offline Return on Investment Maximization for Efficient Decision Making
This is the official implementation of the paper [**"ROIDICE: Offline Return on Investment Maximization for Efficient Decision Making", NeurIPS 2024**.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/178022c409938a9d634b88ce924c4b14-Abstract-Conference.html)

# Requirements
- python 3.9+
- gym==0.17.3
- mujoco==3.1.3
- mujoco-py==1.50.1.68
- jax==0.4.26
- jaxlib==0.4.26+cuda12.cudnn89
- d4rl @ git+https://github.com/Farama-Foundation/d4rl
- neorl @ git+https://https://github.com/polixir/NeoRL

# Offline dataset
We utilze D4RL dataset for locomotion tasks, and NeoRL dataset for financial task.

# How to run
Train and evaluate tabular environment:
```
$ cd ROIDICE
$ python tabular/run_random_cmdp_roidice.py
```

Train and evaluate locomotion environment:
```
$ cd ROIDICE
$ ./scripts/run_mujoco.sh 0 ROIDICE hopper-expert-v2 SoftChi roidice
```

Train and evaluate finance environment:
```
$ cd ROIDICE
$ ./scripts/run_finance.sh 0 ROIDICE finance-high-100 SoftChi roidice
```
