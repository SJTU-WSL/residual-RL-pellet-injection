"""Common wrappers for the batch environment."""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore


class ActionClipWrapper(gym.ActionWrapper):
    """Clip actions to the environment's action_space bounds."""

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class ActionNormalizeWrapper(gym.ActionWrapper):
    """Map normalized [-1, 1] actions to the environment's action_space bounds."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low = np.asarray(self.action_space.low, dtype=np.float32)
        self._high = np.asarray(self.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._low.shape,
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        scaled = (action + 1.0) * 0.5
        return self._low + scaled * (self._high - self._low)


class BatchObsFlattenWrapper(gym.ObservationWrapper):
    """Flatten a (B, D) observation into a 1D vector (B*D,)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        if shape is None or len(shape) != 2:
            raise ValueError("BatchObsFlattenWrapper expects (B, D) observations")
        flat_dim = int(np.prod(shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale reward by a constant factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale


class EpisodeInfoWrapper(gym.Wrapper):
    """Add episode progress info to info dict."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_steps = 0

    def reset(self, **kwargs):
        self.episode_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        info = dict(info)
        info["episode_steps"] = self.episode_steps
        return obs, reward, terminated, truncated, info


class BatchFrameStackWrapper(gym.Wrapper):
    """Stack the last N batch observations along the feature axis.

    Expects observations of shape (B, D) and outputs (B, D * num_stack).
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        if num_stack <= 0:
            raise ValueError("num_stack must be > 0")
        self.num_stack = int(num_stack)
        obs_space = env.observation_space
        if getattr(obs_space, "shape", None) is None or len(obs_space.shape) != 2:
            raise ValueError("BatchFrameStackWrapper expects observation shape (B, D)")

        b, d = obs_space.shape
        low = np.tile(obs_space.low, (1, self.num_stack))
        high = np.tile(obs_space.high, (1, self.num_stack))
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(b, d * self.num_stack),
            dtype=obs_space.dtype,
        )
        self._frames = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
        obs = np.asarray(obs, dtype=np.float32)
        self._frames = [obs] * self.num_stack
        stacked = np.concatenate(self._frames, axis=1)
        if "gymnasium" in gym.__name__:
            return stacked, info
        return stacked

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if self._frames is None:
            self._frames = [obs] * self.num_stack
        else:
            self._frames.pop(0)
            self._frames.append(obs)
        stacked = np.concatenate(self._frames, axis=1)
        return stacked, reward, terminated, truncated, info
