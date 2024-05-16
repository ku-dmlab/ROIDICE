import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import d4rl  # noqa: F401
import gym
import numpy as np
import safety_gym  # noqa: F401
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm
from typing_extensions import assert_never

import algorithm
import environment
import wandb
import wrappers
from dataset_utils import (
    ConstrainedD4RLDataset,
    D4RLDataset,
    Log,
    SafetyGymDataset,
    split_into_trajectories,
)
from divergence import FDivergence
from environment import (
    EnvironmentName,
    SafetyGymEnvironmentName,
    MujocoEnvironmentName,
    MazeEnvironmentName,
)
from evaluation import evaluate
from learner import Learner
from recording_video import recorde_video

FLAGS = flags.FLAGS
flags.DEFINE_string("proj_name", "debug", "Project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./results/neural/", "Tensorboard logging dir.")
flags.DEFINE_string("alg", "OptiDICE", None)
flags.DEFINE_enum("divergence", "Chi", FDivergence, None)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_boolean("log_video", False, "Whether log eval video.")
flags.DEFINE_integer("video_interval", 10000, "Video interval.")
flags.DEFINE_integer("video_steps", 2000, "Run steps for video recording.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_string("mix_dataset", "None", "mix the dataset")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("alpha", 1.0, "temperature")
flags.DEFINE_float("lr_ratio", 0.01, None)
flags.DEFINE_float("gradient_penalty_coeff", 1e-5, None)
flags.DEFINE_float("cost_ub", 100, None)
flags.DEFINE_float("initial_lambda", 1.0, None)
flags.DEFINE_string("ckpt_dir", None, None, required=False)
flags.DEFINE_string("eval_ckpt_dir", None, None, required=False)
flags.DEFINE_string(
    "cost_type", "ctrl", "Type of cost value assignment - max/avg/min/ctrl (default: ctrl)"
)
flags.DEFINE_float("cost_weight", 1.0, "Weight of cost.")
flags.DEFINE_float("cost_lb", 0.1, "Lower bound of cost.")
flags.DEFINE_float("cost_ub_epsilon", 0.0, None)

flags.DEFINE_string("entity", "hy-dmlab", "wandb log entity.")

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
        dataset.costs,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)
    minmax = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards /= minmax
    dataset.rewards *= 1000.0

    dataset.costs /= minmax
    dataset.costs *= 1000.0


def make_env_and_dataset(
    env_name: EnvironmentName, seed: int
) -> Tuple[gym.Env, D4RLDataset | SafetyGymDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    if isinstance(env_name, SafetyGymEnvironmentName):
        env = wrappers.CostLowerBound(env)
    elif isinstance(env_name, MujocoEnvironmentName):
        env = wrappers.ActionRelevantCost(
            env, env_name, FLAGS.cost_type, FLAGS.cost_weight, FLAGS.cost_lb
        )

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if isinstance(env_name, SafetyGymEnvironmentName):

        def get_fname(env_name) -> str:
            match env_name:
                case environment.Point.GOAL:
                    return "ppo_lagrangian_PointGoal1_new.pickle"
                case environment.Point.BUTTON:
                    return "ppo_lagrangian_PointButton1_new.pickle"
                case environment.Point.PUSH:
                    return "ppo_lagrangian_PointPush1_new.pickle"
                case environment.Car.GOAL:
                    return "ppo_lagrangian_CarGoal1_new.pickle"
                case environment.Car.BUTTON:
                    return "ppo_lagrangian_CarButton1_new.pickle"
                case environment.Car.PUSH:
                    return "ppo_lagrangian_CarPush1_new.pickle"
                case _:
                    assert_never(env_name)

        dataset = SafetyGymDataset(
            Path("datasets/") / get_fname(env_name),
        )
    elif isinstance(env_name, MujocoEnvironmentName) or isinstance(env_name, MazeEnvironmentName):
        # dataset = D4RLDataset(env)
        dataset = ConstrainedD4RLDataset(
            env, env_name, FLAGS.cost_type, FLAGS.cost_weight, FLAGS.cost_lb
        )
    else:
        raise NotImplementedError

    if env_name in environment.AntMaze:
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
        dataset.rewards -= 1.0
    elif (
        env_name in environment.Hopper
        or env_name in environment.Halfcheetah
        or env_name in environment.Walker2D
    ):
        normalize(dataset)

    return env, dataset


def main(_):
    # set seed
    np.random.seed(FLAGS.seed)

    env_name = environment.parse_string(FLAGS.env_name)
    alg = algorithm.parse_string(FLAGS.alg)
    divergence = FDivergence(FLAGS.divergence)

    env, dataset = make_env_and_dataset(env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    kwargs["alpha"] = FLAGS.alpha
    kwargs["lr_ratio"] = FLAGS.lr_ratio
    kwargs["alg"] = alg
    kwargs["divergence"] = divergence
    kwargs["initial_lambda"] = FLAGS.initial_lambda
    kwargs["cost_ub"] = FLAGS.cost_ub
    kwargs["gradient_penalty_coeff"] = FLAGS.gradient_penalty_coeff
    kwargs["cost_ub_epsilon"] = FLAGS.cost_ub_epsilon

    timestamp = datetime.fromtimestamp(time.time()).strftime("%m_%d_%H_%M_%S")
    ckpt_dir = Path(
        f"checkpoints/{env_name}_{alg}_{divergence}_{FLAGS.alpha}_{FLAGS.seed}_{timestamp}"
    )
    ckpt_eval_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    agent = Learner(
        FLAGS.seed,
        np.append(env.observation_space.sample(), 0)[np.newaxis],
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        ckpt_dir=ckpt_dir,
        ckpt_eval_dir=ckpt_eval_dir,
        **kwargs,
    )
    kwargs["seed"] = FLAGS.seed
    kwargs["env_name"] = env_name

    kwargs["cost_lb"] = FLAGS.cost_lb
    kwargs["cost_weight"] = FLAGS.cost_weight

    wandb.init(
        entity=FLAGS.entity,
        project=FLAGS.proj_name,
        group=env_name,
        name=f"{alg}_alpha{FLAGS.alpha}",
        tags=[
            env_name,
            alg,
            # f"COST_WEIGHT{FLAGS.cost_weight}",
            # f"COST_LB{FLAGS.cost_lb}",
            FLAGS.divergence,
            f"ALPHA{FLAGS.alpha}",
            f"SEED{FLAGS.seed}",
            f"COST_UB{FLAGS.cost_ub}",
            f"COST_UB_EPSILON{FLAGS.cost_ub_epsilon}",
            f"INITIAL_LAMBDA{FLAGS.initial_lambda}",
        ],
        config=kwargs,
        mode="online",
    )

    log = Log(Path("benchmark") / env_name, kwargs)
    log(f"Log dir: {log.dir}")

    # Train models
    if FLAGS.ckpt_dir is None:
        i = 0
        for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
            batch, unnormalized_return = dataset.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            # debug
            # tqdm.write(f"===i: {i}\n" + str(update_info))

            if i % FLAGS.log_interval == 0:
                wandb.log(update_info, i - 1)

            if i % FLAGS.eval_interval == 0:
                (
                    normalized_return,
                    discounted_return,
                    undiscounted_cost,
                    average_discounted_cost,
                    undiscounted_roi,
                    discounted_roi,
                ) = evaluate(env_name, agent, env, FLAGS.eval_episodes)

                wandb.log(
                    {
                        "average_return": normalized_return,
                        "discounted_return": discounted_return,
                        "undiscounted_cost": undiscounted_cost,
                        "discounted_cost": average_discounted_cost,
                        "undiscounted_roi": undiscounted_roi,
                        "discounted_roi": discounted_roi,
                    },
                    i - 1,
                )

            # if i == 1 or FLAGS.log_video and i == FLAGS.max_steps:
            #     # logging args
            #     logging_path = os.path.join(
            #         FLAGS.save_dir,
            #         f"{env_name}/{alg}/alpha{FLAGS.alpha}/cost_ub{FLAGS.cost_ub}/seed{FLAGS.seed}/log{i}",
            #     )
            #     recorde_video(env_name, agent, env, logging_path, FLAGS.video_steps)

            # if i == FLAGS.max_steps:
            #     agent.save_ckpt(i)
    else:
        agent.load_ckpt(Path(FLAGS.ckpt_dir), FLAGS.max_steps)
        log(f"Loaded checkpoints from {FLAGS.ckpt_dir}")

    # Off-policy evaluation
    (
        normalized_return,
        average_discounted_return,
        undiscounted_cost,
        average_discounted_cost,
        undiscounted_roi,
        discounted_roi,
    ) = evaluate(env_name, agent, env, FLAGS.eval_episodes)

    # logging args
    # if FLAGS.log_video:
    #     logging_path = os.path.join(
    #         FLAGS.save_dir,
    #         f"{env_name}/{alg}/alpha{FLAGS.alpha}/cost_ub{FLAGS.cost_ub}/seed{FLAGS.seed}/ope/",
    #     )
    #     recorde_video(env_name, agent, env, logging_path, FLAGS.video_steps)

    # logging
    # tqdm.write(
    #     str(
    #         {
    #             "off_policy_eval/average_return": normalized_return,
    #             "off_policy_eval/average_discounted_return": average_discounted_return,
    #             "off_policy_eval/discounted_return": discounted_return,
    #             "off_policy_eval/undiscounted_cost": undiscounted_cost,
    #             "off_policy_eval/discounted_cost": average_discounted_cost,
    #             "off_policy_eval/undiscounted_roi": undiscounted_roi,
    #             "off_policy_eval/discounted_roi": discounted_roi,
    #         }
    #     )
    # )
    # wandb.log(
    #     {
    #         "off_policy_eval/average_return": normalized_return,
    #         "off_policy_eval/average_discounted_return": average_discounted_return,
    #         "off_policy_eval/discounted_return": discounted_return,
    #         "off_policy_eval/undiscounted_cost": undiscounted_cost,
    #         "off_policy_eval/discounted_cost": average_discounted_cost,
    #         "off_policy_eval/undiscounted_roi": undiscounted_roi,
    #         "off_policy_eval/discounted_roi": discounted_roi,
    #     }
    # )

    log.close()
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
