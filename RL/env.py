"""Gymnasium-style environment for torax + pellet batch control."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import jax

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

from simulator.torax_simulator import TransportSimulator
from simulator.FPAD_simulator import PelletSimulator
from .reward import compute_reward, evaluate_unsafe_conditions


class ToraxPelletBatchEnv(gym.Env):
    """Single gym.Env that handles an internal batch.

    Action shape: (B, 3) => [trigger, velocity, thickness]
    Observation shape: (B, D)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        torax_config_path: str = "config/ITER.py",
        batch_size: int = 128,
        max_steps: int = 10,
        device: Optional[str] = None,
        trigger_range: Tuple[float, float] = (-1.0, 1.0),
        velocity_range: Tuple[float, float] = (100.0, 1000.0),
        thickness_range: Tuple[float, float] = (0.002, 0.005),
        randomize_on_reset: bool = True,
        warmup_steps: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.torax_config_path = torax_config_path
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.randomize_on_reset = bool(randomize_on_reset)
        self.warmup_steps = int(warmup_steps)
        self._rng = np.random.default_rng(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self._init_simulators()

        # Action space: (B, 3)
        low = np.array([trigger_range[0], velocity_range[0], thickness_range[0]], dtype=np.float32)
        high = np.array([trigger_range[1], velocity_range[1], thickness_range[1]], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.repeat(low[None, :], self.batch_size, axis=0),
            high=np.repeat(high[None, :], self.batch_size, axis=0),
            dtype=np.float32,
        )

        # Observation space is inferred from current simulator state
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        self._last_obs = obs

    def _init_simulators(self) -> None:
        self.env = TransportSimulator(self.torax_config_path, total_batch_size=self.batch_size)
        self.pellet_sim = PelletSimulator(device=str(self.device))
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self.env.reset()
        self.step_count = 0

        self.current_triggers = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        self.current_locs = torch.full((self.batch_size,), 0.5, device=self.device, dtype=torch.float32)
        self.current_widths = torch.full((self.batch_size,), 0.05, device=self.device, dtype=torch.float32)
        self.current_rates = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)

    def _sanitize_action(self, action: Any) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (3,):
            action = np.repeat(action[None, :], self.batch_size, axis=0)
        if action.shape != (self.batch_size, 3):
            raise ValueError(f"Action shape {action.shape} does not match (B,3) with B={self.batch_size}")
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _action_to_controls(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trigger_vals = action[:, 0]
        velocity_vals = action[:, 1]
        thickness_vals = action[:, 2]

        triggers = torch.from_numpy(trigger_vals > 0.0).to(device=self.device)
        velocities = torch.zeros((self.batch_size, 2), device=self.device, dtype=torch.float32)
        velocities[:, 0] = torch.from_numpy(velocity_vals).to(device=self.device)
        velocities[:, 1] = 0.0
        thicknesses = torch.from_numpy(thickness_vals).to(device=self.device, dtype=torch.float32)

        return triggers, velocities, thicknesses

    def _collect_tree_features(self, pytree: Any) -> np.ndarray:
        leaves = jax.tree_util.tree_leaves(pytree)
        features = []
        for leaf in leaves:
            if leaf is None:
                continue
            if isinstance(leaf, (int, float)):
                arr = np.array(leaf, dtype=np.float32)
            else:
                try:
                    arr = np.array(jax.device_get(leaf), dtype=np.float32)
                except Exception:
                    continue

            if arr.ndim >= 2 and arr.shape[0] == self.env.num_devices and arr.shape[1] == self.env.batch_per_device:
                arr = arr.reshape((self.batch_size,) + arr.shape[2:])
            elif arr.ndim >= 1 and arr.shape[0] == self.batch_size:
                pass
            elif arr.ndim == 0:
                arr = np.full((self.batch_size, 1), arr, dtype=np.float32)
                features.append(arr.reshape(self.batch_size, -1))
                continue
            else:
                arr = np.broadcast_to(arr, (self.batch_size,) + arr.shape).astype(np.float32)

            features.append(arr.reshape(self.batch_size, -1))

        if len(features) == 0:
            return np.zeros((self.batch_size, 0), dtype=np.float32)
        return np.concatenate(features, axis=1)

    def _get_obs(self) -> np.ndarray:
        state_feats = self._collect_tree_features(self.env.current_states)
        output_feats = self._collect_tree_features(self.env.last_outputs)
        if output_feats.size == 0:
            obs = state_feats
        elif state_feats.size == 0:
            obs = output_feats
        else:
            obs = np.concatenate([state_feats, output_feats], axis=1)
        obs = obs.astype(np.float32, copy=False)
        # Guard against NaN/Inf from simulator state
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _advance(self, action: np.ndarray, count_step: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 1) Step torax with current pellet injection
        self.env.step(self.current_triggers, self.current_locs, self.current_widths, self.current_rates)

        # 2) Update plasma state for pellet model
        T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = self.env.get_plasma_tensor()
        self.pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)

        # 3) Compute next pellet injection from action
        triggers, velocities, thicknesses = self._action_to_controls(action)
        self.current_triggers = triggers

        # Pellet tracing is only needed for lanes that actually inject.
        active_idx = torch.nonzero(triggers, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            self.current_rates = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            sim_loc = self.current_locs.clone()
            sim_width = self.current_widths.clone()
            sim_rate = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)

            active_velocities = velocities.index_select(0, active_idx)
            active_thicknesses = thicknesses.index_select(0, active_idx)
            active_loc, active_width, active_rate = self.pellet_sim.simulate_pellet_injection(
                batch_size=int(active_idx.numel()),
                velocity=active_velocities,
                thickness=active_thicknesses,
            )

            sim_loc.index_copy_(0, active_idx, active_loc)
            sim_width.index_copy_(0, active_idx, active_width)
            sim_rate.index_copy_(0, active_idx, active_rate)

            self.current_locs = sim_loc
            self.current_widths = sim_width
            self.current_rates = sim_rate

        obs = self._get_obs()

        info: Dict[str, Any] = {}
        try:
            diagnostics = self.env.get_diagnostics()
            info.update(
                {
                    name: value.detach().cpu().numpy()
                    for name, value in diagnostics.items()
                }
            )
        except Exception as e:
            raise RuntimeError("Failed to get diagnostics from environment") from e

        if count_step:
            self.step_count += 1

        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_runtime()

        if self.randomize_on_reset and self.warmup_steps > 0:
            for _ in range(self.warmup_steps):
                random_action = self.action_space.sample().astype(np.float32)
                self._advance(random_action, count_step=False)

        obs = self._get_obs()
        self._last_obs = obs

        if "gymnasium" in gym.__name__:
            return obs, {}
        return obs

    def step(self, action: Any):
        action = self._sanitize_action(action)
        obs, info = self._advance(action, count_step=True)

        reward = compute_reward(self._last_obs, action, obs, info)
        unsafe_batch, unsafe_reason_batch = evaluate_unsafe_conditions(info, self.batch_size)
        unsafe_any = bool(np.any(unsafe_batch))
        if isinstance(reward, np.ndarray):
            info["reward_batch"] = reward
            reward = float(np.mean(reward))
        else:
            reward = float(reward)

        self._last_obs = obs

        terminated = unsafe_any
        truncated = self.step_count >= self.max_steps
        info["unsafe_batch"] = unsafe_batch.astype(bool)
        info["unsafe_reason_batch"] = unsafe_reason_batch
        info["unsafe_any"] = unsafe_any
        info["terminated_batch"] = np.full((self.batch_size,), terminated, dtype=bool)
        info["truncated_batch"] = np.full((self.batch_size,), truncated, dtype=bool)

        if "gymnasium" in gym.__name__:
            return obs, reward, terminated, truncated, info
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        return None
