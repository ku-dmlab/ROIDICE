import gym
import numpy as np

class ActionRelevantCost(gym.Wrapper):
    def __init__(self, env, option="avg", eps=1e-5):
        super().__init__(env)
        self._eps = eps
        self._option = option

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        if self._option == "max":
            info['cost'] = np.max(abs(action)) + self._eps
        elif self._option == "avg":
            info['cost'] = np.mean(abs(action)) + self._eps
        elif self._option == "min":
            info['cost'] = np.min(abs(action)) + self._eps
        else:
            raise NotImplementedError
        return obs, rewards, done, info

class CostLowerBound(gym.Wrapper):
    def __init__(self, env, eps=1e-5):
        super().__init__(env)
        self._eps = eps

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        info['cost'] += self._eps
        return obs, rewards, done, info