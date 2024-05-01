import csv
import json
import pickle
import random
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import d4rl

# TODO use ultra only for antmaze-ultra
# import d4rlultra.d4rl as d4rl
import gym
import numpy as np
from gym.spaces import Box
from tqdm import tqdm

from common import Batch, ConstrainedBatch


def split_into_trajectories(
    observations,
    actions,
    rewards,
    masks,
    dones_float,
    next_observations,
    costs,
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
                costs[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        timesteps: np.ndarray,
        initial_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.unnormalized_rewards = np.copy(self.rewards)
        self.timesteps = timesteps
        self.initial_observations = initial_observations
        self.size = size
        self.initial_size = len(initial_observations)

    def sample(self, batch_size: int) -> tuple[Batch, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        initial_indexes = np.random.randint(self.initial_size, size=batch_size)

        return (
            Batch(
                observations=self.observations[indx],
                actions=self.actions[indx],
                rewards=self.rewards[indx],
                masks=self.masks[indx],
                next_observations=self.next_observations[indx],
                timesteps=self.timesteps[indx],
                initial_observations=self.initial_observations[initial_indexes],
            ),
            self.unnormalized_rewards[indx],
        )


class ConstrainedDatasets(Dataset):
    def __init__(self, *args, costs: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = costs

    def sample(self, batch_size: int) -> tuple[ConstrainedBatch, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        initial_indexes = np.random.randint(self.initial_size, size=batch_size)

        return (
            ConstrainedBatch(
                observations=self.observations[indx],
                actions=self.actions[indx],
                rewards=self.rewards[indx],
                masks=self.masks[indx],
                next_observations=self.next_observations[indx],
                timesteps=self.timesteps[indx],
                initial_observations=self.initial_observations[initial_indexes],
                costs=self.costs[indx],
            ),
            self.unnormalized_rewards[indx],
        )


class D4RLDataset(Dataset):
    def __init__(
        self,
        env: gym.Env,
        add_env: Optional[gym.Env] = None,
        expert_ratio: float = 1.0,
        clip_to_eps: bool = True,
        heavy_tail: bool = False,
        heavy_tail_higher: float = 0.0,
        eps: float = 1e-5,
    ):
        dataset = d4rl.qlearning_dataset(env)
        if add_env is not None:
            add_data = d4rl.qlearning_dataset(add_env)
            if expert_ratio >= 1:
                raise ValueError("in the mix setting, the expert_ratio must < 1")
            length_add_data = int(add_data["rewards"].shape[0] * (1 - expert_ratio))
            length_expert_data = int(length_add_data * expert_ratio)
            for k, _ in dataset.items():
                dataset[k] = np.concatenate(
                    [
                        add_data[k][:-length_expert_data],
                        dataset[k][:length_expert_data],
                    ],
                    axis=0,
                )
            print("-------------------------------")
            print(
                f"we are in the mix data regimes, len(expert):{length_expert_data} | len(add_data): {length_add_data} | expert ratio: {expert_ratio}"
            )
            print("-------------------------------")

        if heavy_tail:
            dataset = d4rl.qlearning_dataset(
                env, heavy_tail=True, heavy_tail_higher=heavy_tail_higher
            )
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            observation_gap = float(
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
            )

            if observation_gap > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        # Create timestep informations.
        t = 0
        timesteps = np.zeros_like(dataset["rewards"], dtype=np.int64)
        for i in range(len(dataset["observations"])):
            timesteps[i] = t

            if dones_float[i] == 1.0:
                t = 0
            else:
                t += 1

        # Extract initial observations.
        (terminal_indexes,) = np.where(dones_float == 1.0)  # noqa: E712
        terminal_indexes = np.insert(terminal_indexes, 0, -1)[:-1]
        initial_observations = dataset["observations"][terminal_indexes + 1]  # type: ignore

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            timesteps=timesteps,
            initial_observations=initial_observations.astype(np.float32),
            size=len(dataset["observations"]),
        )


class ConstrainedD4RLDataset(ConstrainedDatasets):
    def __init__(
        self,
        env: gym.Env,
        env_name: str,
        cost_type: str,
        cost_weight: float,
        cost_lb: float,
        add_env: Optional[gym.Env] = None,
        expert_ratio: float = 1.0,
        clip_to_eps: bool = True,
        heavy_tail: bool = False,
        heavy_tail_higher: float = 0.0,
        eps: float = 1e-5,
    ):
        dataset = d4rl.qlearning_dataset(env)
        if add_env is not None:
            add_data = d4rl.qlearning_dataset(add_env)
            if expert_ratio >= 1:
                raise ValueError("in the mix setting, the expert_ratio must < 1")
            length_add_data = int(add_data["rewards"].shape[0] * (1 - expert_ratio))
            length_expert_data = int(length_add_data * expert_ratio)
            for k, _ in dataset.items():
                dataset[k] = np.concatenate(
                    [
                        add_data[k][:-length_expert_data],
                        dataset[k][:length_expert_data],
                    ],
                    axis=0,
                )
            print("-------------------------------")
            print(
                f"we are in the mix data regimes, len(expert):{length_expert_data} | len(add_data): {length_add_data} | expert ratio: {expert_ratio}"
            )
            print("-------------------------------")

        if heavy_tail:
            dataset = d4rl.qlearning_dataset(
                env, heavy_tail=True, heavy_tail_higher=heavy_tail_higher
            )
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            observation_gap = float(
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
            )

            if observation_gap > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        # Create timestep informations.
        t = 0
        timesteps = np.zeros_like(dataset["rewards"], dtype=np.int64)
        for i in range(len(dataset["observations"])):
            timesteps[i] = t

            if dones_float[i] == 1.0:
                t = 0
            else:
                t += 1

        # Extract initial observations.
        (terminal_indexes,) = np.where(dones_float == 1.0)  # noqa: E712
        terminal_indexes = np.insert(terminal_indexes, 0, -1)[:-1]
        initial_observations = dataset["observations"][terminal_indexes + 1]  # type: ignore

        # absorbing state
        absorbing_dim = np.zeros(len(dataset["observations"]))
        observations = np.concatenate(
            (dataset["observations"], absorbing_dim[:, np.newaxis]), axis=1
        )
        next_observations = np.concatenate(
            (dataset["next_observations"], absorbing_dim[:, np.newaxis]), axis=1
        )
        actions = dataset["actions"].copy()
        absorbing_state = np.zeros(len(observations[0]))
        absorbing_action = np.zeros(len(actions[0]))
        for n, t in enumerate(terminal_indexes): # TODO: too slow!
            insert_idx = t + (n * 2) + 1
            # observations
            observations = np.insert(
                observations,
                insert_idx,
                np.vstack((next_observations[insert_idx - 1], absorbing_state)),
                axis=0,
            )
            observations[i + 1, -1] = 1
            next_observations = np.insert(
                next_observations, insert_idx, np.vstack((absorbing_state, absorbing_state)), axis=0
            )
            next_observations[i : i + 2, -1] = 1
            # actions
            actions = np.insert(
                actions, insert_idx, np.vstack((absorbing_action, absorbing_action)), axis=0
            )
        observations = np.vstack((observations, next_observations[-1], absorbing_state))
        observations[-1, -1] = 1
        next_observations = np.vstack((next_observations, absorbing_state, absorbing_state))
        next_observations[-2:, -1] = 1

        # cost assignment
        if cost_type == "max":
            costs = np.max(abs(dataset['actions']), axis=1) + eps
        elif cost_type == "avg":
            costs = np.mean(abs(dataset['actions']), axis=1) + eps
        elif cost_type == "min":
            costs = np.min(abs(dataset["actions"]), axis=1) + eps
        elif cost_type == "ctrl":
            costs = np.sum(dataset["actions"]**2, axis=1)
        else:
            raise NotImplementedError

        # add ctrl_cost
        if "half" in env_name:
            ctrl_cost_weight = 0.1
            healty_reward = 0.0
        else:  # hopper, walker2d
            ctrl_cost_weight = 0.001
            healty_reward = 1.0
        ctrl_cost = ctrl_cost_weight * costs
        pure_rewards = dataset["rewards"] - healty_reward + ctrl_cost  # forward_reward

        # set cost func
        costs = cost_weight * costs + cost_lb

        # for absorbing state
        for _ in range(2):
            rewards = np.insert(pure_rewards, terminal_indexes+1, 0.0, axis=0)
            costs = np.insert(costs, terminal_indexes+1, 0.001, axis=0)
        rewards = np.concatenate((rewards, np.zeros(2)))
        costs = np.concatenate((costs, np.zeros(2) + 0.001))

        super().__init__(
            observations=observations.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations.astype(np.float32),
            timesteps=timesteps,
            initial_observations=initial_observations.astype(np.float32),
            size=len(observations),
            costs=costs,
        )


class SafetyGymDataset(ConstrainedDatasets):
    def __init__(
        self,
        dataset_fname: Path,
    ):
        with open(dataset_fname, "rb") as fp:
            dataset = pickle.load(fp)

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminateds"].astype(np.float32),
            dones_float=dataset["terminateds"].astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            timesteps=dataset["timesteps"].astype(np.float32),
            initial_observations=dataset["initial_observations"].astype(np.float32),
            costs=dataset["costs"],
            size=len(dataset["observations"]),
        )


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: Box, action_dim: int, capacity: int):
        observations = np.empty((capacity, *observation_space.shape), dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )

        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            initial_observations=np.copy(observations),
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert self.insert_index == 0, "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


def _gen_dir_name():
    now_str = datetime.now().strftime("%m-%d-%y_%H.%M.%S")
    rand_str = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{now_str}_{rand_str}"


class Log:
    def __init__(
        self,
        root_log_dir,
        cfg_dict,
        txt_filename="log.txt",
        csv_filename="progress.csv",
        cfg_filename="config.json",
        flush=True,
    ):
        self.dir = Path(root_log_dir) / _gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir / txt_filename, "w")
        self.csv_file = None
        (self.dir / cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end="\n"):
        now_str = datetime.now().strftime("%H:%M:%S")
        message = f"[{now_str}] " + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir / self.csv_filename, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
