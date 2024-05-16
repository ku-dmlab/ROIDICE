import gym
import numpy as np
from environment import MazeEnvironmentName, MujocoEnvironmentName

class ActionRelevantCost(gym.Wrapper):
    def __init__(self, env, env_name, option, cost_weight, cost_lb, eps=1e-5):
        super().__init__(env)
        self._eps = eps
        self._env_name = env_name
        self._option = option
        self._cost_weight = cost_weight
        self._cost_lb = cost_lb

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        if self._option == "max":
            info['cost'] = np.max(abs(action)) + self._eps
        elif self._option == "avg":
            info['cost'] = np.mean(abs(action)) + self._eps
        elif self._option == "min":
            info['cost'] = np.min(abs(action)) + self._eps
        elif self._option == "ctrl":
            info['cost'] = np.sum(action ** 2)     
        else:
            raise NotImplementedError

        if isinstance(self._env_name, MazeEnvironmentName):
            pure_rewards = rewards
        elif isinstance(self._env_name, MujocoEnvironmentName):
            # add ctrl_cost
            if 'half' in self._env_name:
                ctrl_cost_weight = 0.1
                healthy_reward = 0.0
            else: # hopper, walker2d
                ctrl_cost_weight = 0.001
                healthy_reward = 1.0
            ctrl_cost = ctrl_cost_weight * np.sum(action ** 2)
            pure_rewards = rewards + ctrl_cost - healthy_reward # forward_reward
        else:
            raise NotImplementedError

        # set cost func
        info['cost'] = self._cost_weight * np.mean(action ** 2) + self._cost_lb
        
        return obs, pure_rewards, done, info

class CostLowerBound(gym.Wrapper):
    def __init__(self, env, eps=1e-5):
        super().__init__(env)
        self._eps = eps

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        info['cost'] += self._eps
        return obs, rewards, done, info
