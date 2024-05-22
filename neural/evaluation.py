import os
import typing

import d4rl
import gym
import numpy as np

from environment import EnvironmentName

if typing.TYPE_CHECKING:
    from learner import Learner


def evaluate(
    env_name: EnvironmentName,
    agent: "Learner",
    env: gym.Env,
    num_episodes: int,
    discount: float = 0.99,
    max_step: int = 1000,
) -> tuple[float, float, float, float]:
    total_cost_ = []
    total_reward_ = []
    total_roi_ = []
    discounted_total_cost_ = []
    discounted_total_reward_ = []
    discounted_total_roi_ = []
    for i in range(num_episodes):
        observation: np.ndarray = env.reset()  # type: ignore
        done = False

        total_reward = 0.0
        discounted_total_reward = 0.0
        total_cost = 0.0
        discounted_total_cost = 0.0
        cumulated_discount = 1
        cnt = 0
        while not done and cnt < max_step:
            observation = np.append(observation, 0) # add absorbing dim
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            discounted_total_reward += cumulated_discount * reward
            total_cost += info["cost"]
            discounted_total_cost += cumulated_discount * info["cost"]
            cumulated_discount *= discount
            cnt += 1

        # compute roi = r / c
        assert total_cost != 0.0, f"Err: Division by Zero (total_cost: {total_cost})"
        if total_cost != 0.0:
            total_roi_.append(total_reward / total_cost)
        assert (
            discounted_total_cost != 0.0
        ), f"Err: Division by Zero (discounted_total_cost: {discounted_total_cost})"
        if discounted_total_cost != 0.0:
            discounted_total_roi_.append(discounted_total_reward / discounted_total_cost)

        total_reward_.append(total_reward)
        discounted_total_reward_.append(discounted_total_reward)
        total_cost_.append(total_cost)
        discounted_total_cost_.append(discounted_total_cost)

    average_return = np.array(total_reward_).mean()
    average_discounted_return = np.array(discounted_total_reward_).mean()

    average_undiscounted_cost = np.array(total_cost_).mean()
    average_discounted_cost = np.array(discounted_total_cost_).mean()

    average_undiscounted_roi = np.array(total_roi_).mean()
    average_discounted_roi = np.array(discounted_total_roi_).mean()

    return (
        average_return,
        average_discounted_return,
        average_undiscounted_cost,
        average_discounted_cost,
        average_undiscounted_roi,
        average_discounted_roi,
    )
