import gym
import numpy as np

class ActionRelevantCost(gym.Wrapper):
    def __init__(self, env, env_name, cost_weight, cost_lb, eps=1e-5):
        super().__init__(env)
        self._eps = eps
        self._env_name = env_name
        self._cost_weight = cost_weight
        self._cost_lb = cost_lb

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        
        # add ctrl_cost
        if 'half' in self._env_name:
            ctrl_cost_weight = 0.1
            healthy_reward = 0.0
        else: # hopper, walker2d
            ctrl_cost_weight = 0.001
            healthy_reward = 1.0
        ctrl_cost = ctrl_cost_weight * np.sum(action ** 2)
        pure_rewards = rewards + ctrl_cost - healthy_reward # forward_reward

        # set cost func
        info['cost'] = self._cost_weight * np.mean(action ** 2) + self._cost_lb
        
        return obs, pure_rewards, done, info

class TradeFeeCost(gym.Wrapper):
    def __init__(self, env, env_name, reward_scale, cost_weight, cost_lb):
        super().__init__(env)
        self._env_name = env_name
        self._reward_scale = reward_scale
        self._cost_weight = cost_weight
        self._cost_lb = cost_lb

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # set reward func
        reward = self._reward_scale * reward
        info['cost'] = self._cost_weight * self.env.cost + self._cost_lb

        return obs, reward, done, info
