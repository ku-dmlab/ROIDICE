import os
import typing

import d4rl
import gym
import numpy as np
from PIL import Image

import wandb
from environment import EnvironmentName

if typing.TYPE_CHECKING:
    from learner import Learner

def record_video(
    env_name: EnvironmentName,
    agent: "Learner",
    env: gym.Env,
    logging_path: str,
    max_steps: int = 1500,
):
    os.makedirs(logging_path, exist_ok=True)

    observation: np.ndarray = env.reset()  # type: ignore
    frames = []
    for i in range(max_steps):
        observation = np.append(observation, 0) # add absorbing dim
        action = agent.sample_actions(observation, temperature=0.0)
        observation, reward, done, info = env.step(action)
        frame = env.render('rgb_array').astype(np.uint8)
        frames.append(frame) # for wandb video
        frame = Image.fromarray(frame)
        frame.save(os.path.join(logging_path, f"step{i:04d}.png"))

    # frames = np.transpose(np.array(frames), (0, 3, 1, 2)) # (t, c, h, w)
    # wandb.log({f"video/{env_name}": wandb.Video(frames, fps=16)})
