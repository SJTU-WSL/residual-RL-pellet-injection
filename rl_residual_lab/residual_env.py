"""Residual control environments with a cached 2 s warm-start."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

from RL.env import ToraxPelletBatchEnv
from RL.wrappers import ActionNormalizeWrapper, BatchFrameStackWrapper


@dataclass
class WarmupSnapshot:
    """Cached simulator state after the startup warmup."""

    transport_states: Any
    transport_outputs: Any
    transport_step_count: int
    current_triggers: torch.Tensor
    current_locs: torch.Tensor
    current_widths: torch.Tensor
    current_rates: torch.Tensor
    last_obs: np.ndarray


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _sync_pellet_sim(base_env: ToraxPelletBatchEnv) -> None:
    """Rebuild the pellet simulator plasma profiles from the transport state."""
    T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = base_env.env.get_plasma_tensor()
    base_env.pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)


def _full_action(
    batch_size: int,
    trigger: float,
    velocity_mps: float,
    thickness_m: float,
) -> np.ndarray:
    action = np.zeros((batch_size, 3), dtype=np.float32)
    action[:, 0] = float(trigger)
    action[:, 1] = float(velocity_mps)
    action[:, 2] = float(thickness_m)
    return action


class WarmStartResidualWrapper(gym.Wrapper):
    """Residual control with a cached 2 s warm-start and fixed injection cadence."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: ToraxPelletBatchEnv,
        warmup_steps: int = 2000,
        episode_steps: int = 1000,
        reset_to_cached_warm_state: bool = True,
        inject_every: int = 100,
        inject_duration: int = 1,
        base_velocity_mps: float = 300.0,
        base_thickness_mm: float = 2.0,
        residual_velocity_range: tuple[float, float] = (-50.0, 50.0),
        residual_thickness_mm_range: tuple[float, float] = (-0.5, 0.5),
        append_schedule_features: bool = True,
    ) -> None:
        super().__init__(env)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if episode_steps <= 0:
            raise ValueError("episode_steps must be > 0")
        if inject_every <= 0:
            raise ValueError("inject_every must be > 0")
        if inject_duration <= 0 or inject_duration > inject_every:
            raise ValueError("inject_duration must satisfy 0 < inject_duration <= inject_every")

        base_env = getattr(env, "unwrapped", env)
        self.base_env = base_env
        self.batch_size = int(base_env.batch_size)
        self.warmup_steps = int(warmup_steps)
        self.episode_steps = int(episode_steps)
        self.reset_to_cached_warm_state = bool(reset_to_cached_warm_state)
        self.inject_every = int(inject_every)
        self.inject_duration = int(inject_duration)
        self.base_velocity_mps = float(base_velocity_mps)
        self.base_thickness_m = float(base_thickness_mm) * 1e-3
        self.append_schedule_features = bool(append_schedule_features)

        self._residual_velocity_range = (
            float(residual_velocity_range[0]),
            float(residual_velocity_range[1]),
        )
        self._residual_thickness_range_m = (
            float(residual_thickness_mm_range[0]) * 1e-3,
            float(residual_thickness_mm_range[1]) * 1e-3,
        )

        low = np.array(
            [self._residual_velocity_range[0], self._residual_thickness_range_m[0]],
            dtype=np.float32,
        )
        high = np.array(
            [self._residual_velocity_range[1], self._residual_thickness_range_m[1]],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.repeat(low[None, :], self.batch_size, axis=0),
            high=np.repeat(high[None, :], self.batch_size, axis=0),
            dtype=np.float32,
        )

        obs_space = env.observation_space
        if getattr(obs_space, "shape", None) is None or len(obs_space.shape) != 2:
            raise ValueError("WarmStartResidualWrapper expects observation shape (B, D)")
        if self.append_schedule_features:
            extra_low = np.repeat(
                np.array([[-1.0, -1.0, 0.0, 0.0]], dtype=np.float32),
                self.batch_size,
                axis=0,
            )
            extra_high = np.repeat(
                np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
                self.batch_size,
                axis=0,
            )
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([obs_space.low, extra_low], axis=1),
                high=np.concatenate([obs_space.high, extra_high], axis=1),
                dtype=np.float32,
            )
        else:
            self.observation_space = obs_space

        self._cached_seed: Optional[int] = None
        self._warm_snapshot: Optional[WarmupSnapshot] = None
        self._physical_step_index = 0
        self._control_step_index = 0
        self._control_active = False

    def _phase_features(self) -> np.ndarray:
        if self._control_active:
            phase = (self._control_step_index % self.inject_every) / float(self.inject_every)
            inject_now = 1.0 if (self._control_step_index % self.inject_every) < self.inject_duration else 0.0
            control_active = 1.0
        else:
            phase = 0.0
            inject_now = 0.0
            control_active = 0.0
        angle = 2.0 * np.pi * phase
        features = np.array([np.sin(angle), np.cos(angle), inject_now, control_active], dtype=np.float32)
        return np.repeat(features[None, :], self.batch_size, axis=0)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if not self.append_schedule_features:
            return obs
        return np.concatenate([obs, self._phase_features()], axis=1)

    def _sanitize_residual_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape == (2,):
            arr = np.repeat(arr[None, :], self.batch_size, axis=0)
        expected_shape = (self.batch_size, 2)
        if arr.shape != expected_shape:
            raise ValueError(f"Residual action shape {arr.shape} does not match {expected_shape}")
        return np.clip(arr, self.action_space.low, self.action_space.high)

    def _no_injection_action(self) -> np.ndarray:
        return _full_action(
            batch_size=self.batch_size,
            trigger=-1.0,
            velocity_mps=self.base_velocity_mps,
            thickness_m=self.base_thickness_m,
        )

    def _build_control_action(self, residual_action: np.ndarray) -> tuple[np.ndarray, bool]:
        inject_now = (self._control_step_index % self.inject_every) < self.inject_duration
        full_action = _full_action(
            batch_size=self.batch_size,
            trigger=1.0 if inject_now else -1.0,
            velocity_mps=self.base_velocity_mps,
            thickness_m=self.base_thickness_m,
        )
        if inject_now:
            full_action[:, 1] += residual_action[:, 0]
            full_action[:, 2] += residual_action[:, 1]
        full_action = np.clip(full_action, self.base_env.action_space.low, self.base_env.action_space.high)
        return full_action, inject_now

    def _snapshot_state(self) -> WarmupSnapshot:
        return WarmupSnapshot(
            transport_states=self.base_env.env.current_states,
            transport_outputs=self.base_env.env.last_outputs,
            transport_step_count=int(self.base_env.env.step_count),
            current_triggers=_clone_tensor(self.base_env.current_triggers),
            current_locs=_clone_tensor(self.base_env.current_locs),
            current_widths=_clone_tensor(self.base_env.current_widths),
            current_rates=_clone_tensor(self.base_env.current_rates),
            last_obs=np.array(self.base_env._last_obs, dtype=np.float32, copy=True),
        )

    def _restore_state(self, snapshot: WarmupSnapshot) -> None:
        self.base_env.env.current_states = snapshot.transport_states
        self.base_env.env.last_outputs = snapshot.transport_outputs
        self.base_env.env.step_count = snapshot.transport_step_count
        self.base_env.current_triggers = snapshot.current_triggers.to(device=self.base_env.device)
        self.base_env.current_locs = snapshot.current_locs.to(device=self.base_env.device)
        self.base_env.current_widths = snapshot.current_widths.to(device=self.base_env.device)
        self.base_env.current_rates = snapshot.current_rates.to(device=self.base_env.device)
        self.base_env._last_obs = np.array(snapshot.last_obs, dtype=np.float32, copy=True)
        _sync_pellet_sim(self.base_env)

    def _build_warm_cache(self, seed: Optional[int]) -> None:
        # Warm-cache construction must be allowed to reach the exact warmup
        # horizon without triggering a time-limit truncation.
        self.base_env.max_steps = self.warmup_steps + self.episode_steps
        self.base_env.reset(seed=seed)
        warmup_action = self._no_injection_action()
        for _ in range(self.warmup_steps):
            result = self.base_env.step(warmup_action)
            if len(result) == 5:
                _obs, _reward, terminated, truncated, _info = result
            else:  # pragma: no cover
                _obs, _reward, done, _info = result
                terminated, truncated = bool(done), False
            if terminated or truncated:
                raise RuntimeError(
                    "Warmup terminated before reaching the cached 2 s state. "
                    "The startup transient is not safe enough for cached warm-start training."
                )
        self._warm_snapshot = self._snapshot_state()
        self._cached_seed = seed

    def _ensure_warm_cache(self, seed: Optional[int]) -> None:
        if not self.reset_to_cached_warm_state:
            return
        if self._warm_snapshot is None or self._cached_seed != seed:
            self._build_warm_cache(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if self.reset_to_cached_warm_state:
            self._ensure_warm_cache(seed)
            assert self._warm_snapshot is not None
            self.base_env.max_steps = self.episode_steps
            self._restore_state(self._warm_snapshot)
            self.base_env.step_count = 0
            obs = np.array(self.base_env._last_obs, dtype=np.float32, copy=True)
            self._physical_step_index = self.warmup_steps
            self._control_step_index = 0
            self._control_active = True
            info: dict[str, Any] = {"reset_mode": "cached_warm"}
        else:
            self.base_env.max_steps = self.episode_steps
            obs = self.base_env.reset(seed=seed, options=options)
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}
            self._physical_step_index = 0
            self._control_step_index = 0
            self._control_active = False
            info = dict(info)
            info["reset_mode"] = "cold_start"
        obs = self._augment_obs(obs)
        if "gymnasium" in gym.__name__:
            return obs, info
        return obs

    def step(self, action: Any):
        residual_action = self._sanitize_residual_action(action)

        if self.reset_to_cached_warm_state or self._physical_step_index >= self.warmup_steps:
            self._control_active = True
            full_action, inject_now = self._build_control_action(residual_action)
        else:
            self._control_active = False
            residual_action = np.zeros_like(residual_action, dtype=np.float32)
            full_action = self._no_injection_action()
            inject_now = False

        result = self.base_env.step(full_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:  # pragma: no cover
            obs, reward, done, info = result
            terminated, truncated = bool(done), False

        info = dict(info)
        info["control_active"] = np.full((self.batch_size,), self._control_active, dtype=bool)
        info["warmup_steps"] = np.full((self.batch_size,), self.warmup_steps, dtype=np.int32)
        info["physical_step"] = np.full((self.batch_size,), self._physical_step_index, dtype=np.int32)
        info["control_step"] = np.full(
            (self.batch_size,),
            self._control_step_index if self._control_active else -1,
            dtype=np.int32,
        )
        info["inject_now"] = np.full((self.batch_size,), inject_now, dtype=np.float32)
        info["baseline_velocity_mps"] = np.full((self.batch_size,), self.base_velocity_mps, dtype=np.float32)
        info["baseline_thickness_mm"] = np.full((self.batch_size,), self.base_thickness_m * 1e3, dtype=np.float32)
        info["residual_velocity_mps"] = residual_action[:, 0].astype(np.float32)
        info["residual_thickness_mm"] = (residual_action[:, 1] * 1e3).astype(np.float32)
        info["actual_velocity_mps"] = full_action[:, 1].astype(np.float32)
        info["actual_thickness_mm"] = (full_action[:, 2] * 1e3).astype(np.float32)

        if self._control_active:
            self._control_step_index += 1
        self._physical_step_index += 1
        obs = self._augment_obs(obs)

        if "gymnasium" in gym.__name__:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info


def make_residual_env_fn(
    torax_config: str,
    batch_size: int,
    episode_steps: int,
    device: Optional[str],
    seed: Optional[int],
    warmup_steps: int,
    num_stack: int,
    inject_every: int,
    inject_duration: int,
    base_velocity_mps: float,
    base_thickness_mm: float,
    residual_velocity_max: float,
    residual_thickness_mm_max: float,
    normalize_actions: bool,
    append_schedule_features: bool,
    reset_to_cached_warm_state: bool,
) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        runtime_steps = max(int(episode_steps), int(warmup_steps))
        env: gym.Env = ToraxPelletBatchEnv(
            torax_config_path=torax_config,
            batch_size=batch_size,
            max_steps=runtime_steps,
            device=device,
            randomize_on_reset=False,
            warmup_steps=0,
            seed=seed,
        )
        env = WarmStartResidualWrapper(
            env,
            warmup_steps=warmup_steps,
            episode_steps=episode_steps,
            reset_to_cached_warm_state=reset_to_cached_warm_state,
            inject_every=inject_every,
            inject_duration=inject_duration,
            base_velocity_mps=base_velocity_mps,
            base_thickness_mm=base_thickness_mm,
            residual_velocity_range=(-residual_velocity_max, residual_velocity_max),
            residual_thickness_mm_range=(-residual_thickness_mm_max, residual_thickness_mm_max),
            append_schedule_features=append_schedule_features,
        )
        if normalize_actions:
            env = ActionNormalizeWrapper(env)
        if num_stack > 1:
            env = BatchFrameStackWrapper(env, num_stack=num_stack)
        return env

    return _make


def zero_residual_action(batch_size: int) -> np.ndarray:
    return np.zeros((batch_size, 2), dtype=np.float32)
