# ROIDICE
Genius M.S. Kim's 3rd DICE research

# How to run
Train and evaluate mujoco environment
```
$ cd ROIDICE
$ ./scripts/run_mujoco.sh 0 ROIDICE hopper-medium-v2 SoftChiT 1000 roidice
```

Train and evaluate safety-gym environmet
```
$ cd ROIDICE
$ ./scripts/run_safety_gym.sh 0 COptiDICE Safexp-PointPush1-v0 SoftChi roidice


Train and evaluate tabular environment
```
$ cd ROIDICE
$ python tabular/run_random_cmdp_roidice.py
```
