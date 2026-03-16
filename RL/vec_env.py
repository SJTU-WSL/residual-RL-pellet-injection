"""VecEnv wrapper with step_async/step_wait for a single batch env."""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except Exception as exc:  # pragma: no cover
    raise ImportError("stable_baselines3 is required for RL/vec_env.py") from exc

class SingleBatchVecEnv(VecEnv):
    """SB3 VecEnv wrapper around a single batch environment."""

    def __init__(self, env):
        self.env = env
        super().__init__(
            num_envs=1,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        self._actions = None

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs
        return obs

    def step_async(self, actions):
        if isinstance(actions, (list, tuple)):
            actions = np.asarray(actions)
        if isinstance(actions, np.ndarray) and actions.shape[:1] == (1,):
            actions = actions[0]
        self._actions = actions

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)
        # print('find info')
        # print(info)
        # exit()
        done = bool(terminated or truncated)
        rewards = np.array([reward], dtype=np.float32)
        dones = np.array([done], dtype=bool)
        infos: List[dict] = [info]
        if done:
            infos[0] = dict(infos[0])
            infos[0]["terminal_observation"] = obs
            obs = self.reset()
        return obs, rewards, dones, infos

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def seed(self, seed: int | None = None) -> List[int]:
        self.env.reset(seed=seed)
        return [seed if seed is not None else 0]

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: Sequence[int] | None = None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices: Sequence[int] | None = None) -> List[bool]:
        return [isinstance(self.env, wrapper_class)]


class BatchAsVecEnv(VecEnv):
    """Expose a batched env as N independent VecEnv instances."""

    def __init__(self, env):
        self.env = env
        base_env = getattr(env, "unwrapped", env)
        self.batch_size = base_env.batch_size
        obs_space = env.observation_space
        act_space = env.action_space
        if getattr(obs_space, "shape", None) is None or len(obs_space.shape) != 2:
            raise ValueError("BatchAsVecEnv expects observation shape (B, D)")
        if obs_space.shape[0] != self.batch_size:
            raise ValueError("Observation batch dimension does not match batch_size")
        if getattr(act_space, "shape", None) is None or len(act_space.shape) != 2:
            raise ValueError("BatchAsVecEnv expects action shape (B, A)")
        if act_space.shape[0] != self.batch_size:
            raise ValueError("Action batch dimension does not match batch_size")

        per_obs_shape = obs_space.shape[1:]
        per_act_shape = act_space.shape[1:]
        per_obs_low = obs_space.low[0]
        per_obs_high = obs_space.high[0]
        per_act_low = act_space.low[0]
        per_act_high = act_space.high[0]
        super().__init__(
            num_envs=self.batch_size,
            observation_space=type(obs_space)(low=per_obs_low, high=per_obs_high, shape=per_obs_shape, dtype=obs_space.dtype),
            action_space=type(act_space)(low=per_act_low, high=per_act_high, shape=per_act_shape, dtype=act_space.dtype),
        )
        self._per_action_shape = per_act_shape
        self._actions = None

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs
        return obs

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.batch_size,) + self._per_action_shape
        if actions.shape != expected_shape:
            raise ValueError(f"Expected actions shape {expected_shape}, got {actions.shape}")
        self._actions = actions
    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)

        # 1) Try to get per-batch done signals from info (shape: (B,))
        terminated_batch = info.get("terminated_batch", None)
        truncated_batch = info.get("truncated_batch", None)

        if terminated_batch is not None or truncated_batch is not None:
            # If one exists, treat missing one as all-False
            if terminated_batch is None:
                terminated_batch = np.zeros((self.batch_size,), dtype=bool)
            if truncated_batch is None:
                truncated_batch = np.zeros((self.batch_size,), dtype=bool)

            terminated_batch = np.asarray(terminated_batch, dtype=bool).reshape(self.batch_size)
            truncated_batch = np.asarray(truncated_batch, dtype=bool).reshape(self.batch_size)
            dones = terminated_batch | truncated_batch                      # (B,)
            done_any = bool(np.any(dones))                                  # any env done this step
            done_all = bool(np.all(dones))                                  # all env done this step
        else:
            # 2) Fallback: global done (sync episode)
            done_scalar = bool(terminated or truncated)
            dones = np.full((self.batch_size,), done_scalar, dtype=bool)    # (B,)
            done_any = done_scalar
            done_all = done_scalar

        # 3) Rewards
        reward_batch = info.get("reward_batch", None)
        if reward_batch is None:
            rewards = np.full((self.batch_size,), float(reward), dtype=np.float32)
        else:
            rewards = np.asarray(reward_batch, dtype=np.float32).reshape(self.batch_size)

        # 4) Build infos list (len=B). Only attach terminal_observation for done indices.
        infos = []
        for idx in range(self.batch_size):
            item = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray) and value.shape[:1] == (self.batch_size,):
                    item[key] = value[idx]
                else:
                    item[key] = value
            if dones[idx]:
                item["terminal_observation"] = obs[idx]
            infos.append(item)

        # 5) Reset handling
        # If your base env only supports full reset, we can only reset when ALL are done,
        # otherwise we would wipe trajectories of unfinished envs.
        if done_all:
            obs = self.reset()

        return obs, rewards, dones, infos
    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def seed(self, seed: int | None = None) -> List[int]:
        self.env.reset(seed=seed)
        return [seed + i if seed is not None else 0 for i in range(self.batch_size)]

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: Sequence[int] | None = None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices: Sequence[int] | None = None) -> List[bool]:
        return [isinstance(self.env, wrapper_class)]
"""
    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)
        #done = bool(terminated or truncated)
        terminated_batch = info.get("terminated_batch")  # shape (B,)
        truncated_batch  = info.get("truncated_batch")   # shape (B,)
        done = np.asarray(terminated_batch) | np.asarray(truncated_batch)
        reward_batch = info.get("reward_batch")
        if reward_batch is None:
            rewards = np.full((self.batch_size,), reward, dtype=np.float32)
        else:
            rewards = np.asarray(reward_batch, dtype=np.float32).reshape(self.batch_size)

        dones = np.full((self.batch_size,), done, dtype=bool)

        infos = []
        for idx in range(self.batch_size):
            item = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray) and value.shape[:1] == (self.batch_size,):
                    item[key] = value[idx]
                else:
                    item[key] = value
            if done:
                item["terminal_observation"] = obs[idx]
            infos.append(item)

        if done:
            obs = self.reset()

        return obs, rewards, dones, infos
"""
