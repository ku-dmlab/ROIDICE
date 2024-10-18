import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import d4rl  # noqa: F401
import neorl
import gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm

import algorithm
import environment
import wandb
import wrappers
from dataset_utils import (
    ConstrainedD4RLDataset,
    ConstrainedFinanceDataset,
    split_into_trajectories,
)
from divergence import FDivergence
from environment import (
    EnvironmentName,
    MujocoEnvironmentName,
    FinanceEnvironmentName,
)
from evaluation import evaluate
from learner import Learner
from recording_video import record_video

FLAGS = flags.FLAGS
flags.DEFINE_string("proj_name", "debug", "Project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./results/neural/", "Tensorboard logging dir.")
flags.DEFINE_string("alg", "ROIDICE", None)
flags.DEFINE_enum("divergence", "ChiT", FDivergence, None)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_boolean("log_video", False, "Whether log eval video.")
flags.DEFINE_integer("video_interval", 10000, "Video interval.")
flags.DEFINE_integer("video_steps", 1000, "Run steps for video recording.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_string("mix_dataset", "None", "mix the dataset")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("alpha", 1.0, "temperature")
flags.DEFINE_float("lr_ratio", 0.01, None)
flags.DEFINE_float("gradient_penalty_coeff", 1e-5, None)
flags.DEFINE_float("cost_weight", 0.0001, "Weight of cost.")
flags.DEFINE_float("cost_lb", 0.1, "Lower bound of cost.")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")

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
) -> Tuple[gym.Env, ConstrainedD4RLDataset | ConstrainedFinanceDataset]:
    if isinstance(env_name, MujocoEnvironmentName):
        env = gym.make(env_name)
    elif isinstance(env_name, FinanceEnvironmentName):
        env = neorl.make("finance")

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    if isinstance(env_name, MujocoEnvironmentName):
        env = wrappers.ActionRelevantCost(
            env, env_name, FLAGS.cost_weight, FLAGS.cost_lb
        )
    elif isinstance(env_name, FinanceEnvironmentName):
        env = wrappers.TradeFeeCost(
            env, env_name, FLAGS.reward_scale, FLAGS.cost_weight, FLAGS.cost_lb
        )

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    if isinstance(env_name, MujocoEnvironmentName):
        dataset = ConstrainedD4RLDataset(
            env, env_name, FLAGS.cost_weight, FLAGS.cost_lb
        )
    elif isinstance(env_name, FinanceEnvironmentName):
        dataset = ConstrainedFinanceDataset(
            env, env_name, FLAGS.reward_scale, FLAGS.cost_weight, FLAGS.cost_lb
        )
    else:
        raise NotImplementedError

    if (
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
    kwargs["gradient_penalty_coeff"] = FLAGS.gradient_penalty_coeff

    agent = Learner(
        FLAGS.seed,
        np.append(env.observation_space.sample(), 0)[np.newaxis],
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        **kwargs,
    )
    kwargs["seed"] = FLAGS.seed
    kwargs["env_name"] = env_name

    kwargs["cost_lb"] = FLAGS.cost_lb
    kwargs["cost_weight"] = FLAGS.cost_weight
    kwargs["reward_scale"] = FLAGS.reward_scale

    wandb.init(
        entity=FLAGS.entity,
        project=FLAGS.proj_name,
        group=env_name,
        name=f"{alg}_alpha{FLAGS.alpha}",
        tags=[
            env_name,
            alg,
            FLAGS.divergence,
            f"ALPHA{FLAGS.alpha}",
            f"SEED{FLAGS.seed}",
        ],
        config=kwargs,
        mode="online",
    )

    # Train models
    for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch, unnormalized_return = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

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

        if FLAGS.log_video and (i + 1) == FLAGS.video_interval:
            if not isinstance(env_name, FinanceEnvironmentName):
                # logging args
                logging_path = os.path.join(
                    FLAGS.save_dir,
                    f"{env_name}/{alg}/{divergence}/alpha{FLAGS.alpha}/seed{FLAGS.seed}/log{i}",
                )
                record_video(env_name, agent, env, logging_path, FLAGS.video_steps)

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
    if FLAGS.log_video and not isinstance(env_name, FinanceEnvironmentName):
        # logging args
        logging_path = os.path.join(
            FLAGS.save_dir,
            f"{env_name}/{alg}/{divergence}/alpha{FLAGS.alpha}/seed{FLAGS.seed}/ope",
        )
        record_video(env_name, agent, env, logging_path, FLAGS.video_steps)

    wandb.log(
        {
            "off_policy_eval/average_return": normalized_return,
            "off_policy_eval/average_discounted_return": average_discounted_return,
            "off_policy_eval/discounted_return": discounted_return,
            "off_policy_eval/undiscounted_cost": undiscounted_cost,
            "off_policy_eval/discounted_cost": average_discounted_cost,
            "off_policy_eval/undiscounted_roi": undiscounted_roi,
            "off_policy_eval/discounted_roi": discounted_roi,
        }
    )

    wandb.finish()

if __name__ == "__main__":
    app.run(main)
