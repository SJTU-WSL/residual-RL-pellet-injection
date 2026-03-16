"""Full residual control environments with timing, velocity and thickness residuals."""
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
    T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = base_env.env.get_plasma_tensor()
    base_env.pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)


def _full_action(batch_size: int, trigger: float, velocity_mps: float, thickness_m: float) -> np.ndarray:
    action = np.zeros((batch_size, 3), dtype=np.float32)
    action[:, 0] = float(trigger)
    action[:, 1] = float(velocity_mps)
    action[:, 2] = float(thickness_m)
    return action


class WarmStartFullResidualWrapper(gym.Wrapper):
    """Warm-start residual control with event-level timing/velocity/thickness planning."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: ToraxPelletBatchEnv,
        warmup_steps: int = 2000,
        episode_steps: int = 1000,
        sim_steps_per_rl_step: int = 10,
        reset_to_cached_warm_state: bool = True,
        base_interval_steps: int = 100,
        inject_duration: int = 1,
        min_interval_steps: int = 20,
        max_interval_steps: int = 200,
        base_velocity_mps: float = 300.0,
        base_thickness_mm: float = 2.0,
        residual_interval_range: tuple[float, float] = (-20.0, 20.0),
        residual_velocity_range: tuple[float, float] = (-50.0, 50.0),
        residual_thickness_mm_range: tuple[float, float] = (-0.5, 0.5),
        append_scheduler_features: bool = True,
    ) -> None:
        super().__init__(env)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if episode_steps <= 0:
            raise ValueError("episode_steps must be > 0")
        if sim_steps_per_rl_step <= 0:
            raise ValueError("sim_steps_per_rl_step must be > 0")
        if base_interval_steps <= 0:
            raise ValueError("base_interval_steps must be > 0")
        if inject_duration <= 0:
            raise ValueError("inject_duration must be > 0")
        if min_interval_steps <= 0 or max_interval_steps < min_interval_steps:
            raise ValueError("Require 0 < min_interval_steps <= max_interval_steps")

        base_env = getattr(env, "unwrapped", env)
        self.base_env = base_env
        self.batch_size = int(base_env.batch_size)
        self.warmup_steps = int(warmup_steps)
        self.episode_steps = int(episode_steps)
        self.sim_steps_per_rl_step = int(sim_steps_per_rl_step)
        self.reset_to_cached_warm_state = bool(reset_to_cached_warm_state)
        self.base_interval_steps = int(base_interval_steps)
        self.inject_duration = int(inject_duration)
        self.min_interval_steps = int(min_interval_steps)
        self.max_interval_steps = int(max_interval_steps)
        self.base_velocity_mps = float(base_velocity_mps)
        self.base_thickness_m = float(base_thickness_mm) * 1e-3
        self.append_scheduler_features = bool(append_scheduler_features)

        self._residual_interval_range = (
            float(residual_interval_range[0]),
            float(residual_interval_range[1]),
        )
        self._residual_velocity_range = (
            float(residual_velocity_range[0]),
            float(residual_velocity_range[1]),
        )
        self._residual_thickness_range_m = (
            float(residual_thickness_mm_range[0]) * 1e-3,
            float(residual_thickness_mm_range[1]) * 1e-3,
        )

        low = np.array(
            [
                self._residual_interval_range[0],
                self._residual_velocity_range[0],
                self._residual_thickness_range_m[0],
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self._residual_interval_range[1],
                self._residual_velocity_range[1],
                self._residual_thickness_range_m[1],
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.repeat(low[None, :], self.batch_size, axis=0),
            high=np.repeat(high[None, :], self.batch_size, axis=0),
            dtype=np.float32,
        )

        obs_space = env.observation_space
        if getattr(obs_space, "shape", None) is None or len(obs_space.shape) != 2:
            raise ValueError("WarmStartFullResidualWrapper expects observation shape (B, D)")
        if self.append_scheduler_features:
            extra_low = np.repeat(
                np.array([[-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                self.batch_size,
                axis=0,
            )
            extra_high = np.repeat(
                np.array([[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
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
        self._control_active = np.zeros((self.batch_size,), dtype=bool)
        self._planner_due = np.zeros((self.batch_size,), dtype=bool)
        self._planned_interval_steps = np.full((self.batch_size,), self.base_interval_steps, dtype=np.int32)
        self._planned_velocity_mps = np.full((self.batch_size,), self.base_velocity_mps, dtype=np.float32)
        self._planned_thickness_m = np.full((self.batch_size,), self.base_thickness_m, dtype=np.float32)
        self._schedule_origin_step = np.zeros((self.batch_size,), dtype=np.int32)
        self._next_inject_step = np.full((self.batch_size,), -1, dtype=np.int32)
        self._last_actual_velocity_mps = np.full((self.batch_size,), self.base_velocity_mps, dtype=np.float32)
        self._last_actual_thickness_m = np.full((self.batch_size,), self.base_thickness_m, dtype=np.float32)
        self._last_applied_interval_steps = np.full((self.batch_size,), self.base_interval_steps, dtype=np.int32)
        self._last_plan_residual = np.zeros((self.batch_size, 3), dtype=np.float32)

    def _aggregate_macro_infos(
        self,
        infos: list[dict[str, Any]],
        reward_batches: list[np.ndarray],
        macro_steps_executed: int,
    ) -> dict[str, Any]:
        merged = dict(infos[-1])
        merged["reward_batch"] = np.sum(reward_batches, axis=0).astype(np.float32)
        merged["macro_steps_executed"] = np.full((self.batch_size,), macro_steps_executed, dtype=np.int32)
        merged["sim_steps_per_rl_step"] = np.full((self.batch_size,), self.sim_steps_per_rl_step, dtype=np.int32)

        for key in ("inject_now", "plan_applied", "actual_velocity_mps", "actual_thickness_mm"):
            values = [np.asarray(info[key]) for info in infos if key in info]
            if not values:
                continue
            arr = np.stack(values, axis=0)
            merged[key] = np.max(arr, axis=0)

        return merged

    def _scheduler_features(self) -> np.ndarray:
        phase = np.zeros((self.batch_size,), dtype=np.float32)
        time_to_next = np.zeros((self.batch_size,), dtype=np.float32)
        active_mask = self._control_active & (~self._planner_due) & (self._next_inject_step >= 0) & (self._planned_interval_steps > 0)
        if np.any(active_mask):
            elapsed = np.maximum(self._control_step_index - self._schedule_origin_step[active_mask], 0)
            phase[active_mask] = np.clip(elapsed / self._planned_interval_steps[active_mask].astype(np.float32), 0.0, 1.0)
            time_to_next[active_mask] = np.maximum(self._next_inject_step[active_mask] - self._control_step_index, 0).astype(np.float32)

        angle = 2.0 * np.pi * phase
        interval_norm = self._planned_interval_steps.astype(np.float32) / float(self.base_interval_steps)
        time_to_next_norm = time_to_next / float(self.base_interval_steps)
        vel_ratio = self._last_actual_velocity_mps / max(self.base_velocity_mps, 1e-6)
        thk_ratio = self._last_actual_thickness_m / max(self.base_thickness_m, 1e-9)
        return np.stack(
            [
                np.sin(angle),
                np.cos(angle),
                time_to_next_norm,
                interval_norm,
                self._planner_due.astype(np.float32),
                self._control_active.astype(np.float32),
                vel_ratio.astype(np.float32),
                thk_ratio.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if not self.append_scheduler_features:
            return obs
        return np.concatenate([obs, self._scheduler_features()], axis=1)

    def _sanitize_residual_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape == (3,):
            arr = np.repeat(arr[None, :], self.batch_size, axis=0)
        expected_shape = (self.batch_size, 3)
        if arr.shape != expected_shape:
            raise ValueError(f"Full residual action shape {arr.shape} does not match {expected_shape}")
        return np.clip(arr, self.action_space.low, self.action_space.high)

    def _no_injection_action(self) -> np.ndarray:
        return _full_action(
            batch_size=self.batch_size,
            trigger=-1.0,
            velocity_mps=self.base_velocity_mps,
            thickness_m=self.base_thickness_m,
        )

    def _plan_from_residual(self, residual_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = np.rint(self.base_interval_steps + residual_action[:, 0]).astype(np.int32)
        interval = np.clip(interval, self.min_interval_steps, self.max_interval_steps)
        velocity = np.clip(
            self.base_velocity_mps + residual_action[:, 1],
            self.base_env.action_space.low[:, 1],
            self.base_env.action_space.high[:, 1],
        ).astype(np.float32)
        thickness = np.clip(
            self.base_thickness_m + residual_action[:, 2],
            self.base_env.action_space.low[:, 2],
            self.base_env.action_space.high[:, 2],
        ).astype(np.float32)
        return interval, velocity, thickness

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

    def _reset_scheduler(self, control_active: bool, physical_step_index: int) -> None:
        self._physical_step_index = physical_step_index
        self._control_step_index = 0
        self._control_active.fill(control_active)
        self._planner_due.fill(control_active)
        self._planned_interval_steps.fill(self.base_interval_steps)
        self._planned_velocity_mps.fill(self.base_velocity_mps)
        self._planned_thickness_m.fill(self.base_thickness_m)
        self._schedule_origin_step.fill(0)
        self._next_inject_step.fill(-1)
        self._last_actual_velocity_mps.fill(self.base_velocity_mps)
        self._last_actual_thickness_m.fill(self.base_thickness_m)
        self._last_applied_interval_steps.fill(self.base_interval_steps)
        self._last_plan_residual = np.zeros((self.batch_size, 3), dtype=np.float32)

    def _build_warm_cache(self, seed: Optional[int]) -> None:
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
                    "Warmup terminated before reaching the cached warm state. "
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
            self._reset_scheduler(control_active=True, physical_step_index=self.warmup_steps)
            info: dict[str, Any] = {"reset_mode": "cached_warm"}
        else:
            self.base_env.max_steps = self.warmup_steps + self.episode_steps
            obs = self.base_env.reset(seed=seed, options=options)
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}
            self._reset_scheduler(control_active=False, physical_step_index=0)
            info = dict(info)
            info["reset_mode"] = "cold_start"
        obs = self._augment_obs(obs)
        if "gymnasium" in gym.__name__:
            return obs, info
        return obs

    def _step_one_simulator_step(self, residual_action: np.ndarray, allow_plan: bool):
        plan_applied_mask = np.zeros((self.batch_size,), dtype=bool)

        if (not np.all(self._control_active)) and self._physical_step_index >= self.warmup_steps:
            self._control_active[:] = True
            self._planner_due[:] = True

        plan_mask = self._control_active & self._planner_due if allow_plan else np.zeros((self.batch_size,), dtype=bool)
        if np.any(plan_mask):
            interval, velocity, thickness = self._plan_from_residual(residual_action)
            self._planned_interval_steps[plan_mask] = interval[plan_mask]
            self._planned_velocity_mps[plan_mask] = velocity[plan_mask]
            self._planned_thickness_m[plan_mask] = thickness[plan_mask]
            self._schedule_origin_step[plan_mask] = self._control_step_index
            self._next_inject_step[plan_mask] = self._control_step_index + self._planned_interval_steps[plan_mask]
            self._planner_due[plan_mask] = False
            self._last_plan_residual[plan_mask] = residual_action[plan_mask]
            plan_applied_mask[plan_mask] = True

        inject_mask = (
            self._control_active
            & (~self._planner_due)
            & (self._next_inject_step >= 0)
            & (self._control_step_index >= self._next_inject_step)
        )

        full_action = self._no_injection_action()
        if np.any(inject_mask):
            full_action[inject_mask, 0] = 1.0
            full_action[inject_mask, 1] = self._planned_velocity_mps[inject_mask]
            full_action[inject_mask, 2] = self._planned_thickness_m[inject_mask]

        result = self.base_env.step(full_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:  # pragma: no cover
            obs, reward, done, info = result
            terminated, truncated = bool(done), False

        info = dict(info)
        next_inject_in_steps = np.where(
            self._next_inject_step >= 0,
            np.maximum(self._next_inject_step - self._control_step_index, 0),
            0,
        ).astype(np.int32)
        info["control_active"] = self._control_active.astype(bool)
        info["planner_due"] = self._planner_due.astype(bool)
        info["plan_applied"] = plan_applied_mask.astype(bool)
        info["warmup_steps"] = np.full((self.batch_size,), self.warmup_steps, dtype=np.int32)
        info["physical_step"] = np.full((self.batch_size,), self._physical_step_index, dtype=np.int32)
        info["control_step"] = np.full((self.batch_size,), self._control_step_index if np.any(self._control_active) else -1, dtype=np.int32)
        info["inject_now"] = inject_mask.astype(np.float32)
        info["base_interval_steps"] = np.full((self.batch_size,), self.base_interval_steps, dtype=np.float32)
        info["planned_interval_steps"] = self._planned_interval_steps.astype(np.float32)
        info["residual_interval_steps"] = self._last_plan_residual[:, 0].astype(np.float32)
        info["next_inject_in_steps"] = next_inject_in_steps.astype(np.float32)
        info["planned_velocity_mps"] = self._planned_velocity_mps.astype(np.float32)
        info["planned_thickness_mm"] = (self._planned_thickness_m * 1e3).astype(np.float32)
        info["residual_velocity_mps"] = self._last_plan_residual[:, 1].astype(np.float32)
        info["residual_thickness_mm"] = (self._last_plan_residual[:, 2] * 1e3).astype(np.float32)
        info["actual_velocity_mps"] = np.where(inject_mask, self._planned_velocity_mps, 0.0).astype(np.float32)
        info["actual_thickness_mm"] = np.where(inject_mask, self._planned_thickness_m * 1e3, 0.0).astype(np.float32)
        info["last_inject_velocity_mps"] = self._last_actual_velocity_mps.astype(np.float32)
        info["last_inject_thickness_mm"] = (self._last_actual_thickness_m * 1e3).astype(np.float32)

        if np.any(inject_mask):
            self._last_actual_velocity_mps[inject_mask] = self._planned_velocity_mps[inject_mask]
            self._last_actual_thickness_m[inject_mask] = self._planned_thickness_m[inject_mask]
            self._last_applied_interval_steps[inject_mask] = self._planned_interval_steps[inject_mask]
            self._planner_due[inject_mask] = True
            self._next_inject_step[inject_mask] = -1

        if np.any(self._control_active):
            self._control_step_index += 1
        self._physical_step_index += 1
        obs = self._augment_obs(obs)

        if "gymnasium" in gym.__name__:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info

    def step(self, action: Any):
        residual_action = self._sanitize_residual_action(action)
        macro_rewards = []
        macro_infos = []
        obs = None
        terminated = False
        truncated = False
        reward_scalar = 0.0
        plan_consumed = False

        for _ in range(self.sim_steps_per_rl_step):
            obs, reward, terminated, truncated, info = self._step_one_simulator_step(
                residual_action,
                allow_plan=not plan_consumed,
            )
            reward_batch = info.get("reward_batch")
            if reward_batch is None:
                reward_batch = np.full((self.batch_size,), float(reward), dtype=np.float32)
            reward_batch = np.asarray(reward_batch, dtype=np.float32).reshape(self.batch_size)
            macro_rewards.append(reward_batch)
            macro_infos.append(info)
            reward_scalar += float(np.mean(reward_batch))
            if np.any(np.asarray(info.get("plan_applied", np.zeros((self.batch_size,), dtype=bool)), dtype=bool)):
                plan_consumed = True
            if terminated or truncated:
                break

        assert obs is not None
        merged_info = self._aggregate_macro_infos(
            macro_infos,
            macro_rewards,
            macro_steps_executed=len(macro_rewards),
        )

        if "gymnasium" in gym.__name__:
            return obs, reward_scalar, terminated, truncated, merged_info
        return obs, reward_scalar, terminated or truncated, merged_info


def make_full_residual_env_fn(
    torax_config: str,
    batch_size: int,
    episode_steps: int,
    device: Optional[str],
    seed: Optional[int],
    warmup_steps: int,
    sim_steps_per_rl_step: int,
    num_stack: int,
    base_interval_steps: int,
    inject_duration: int,
    min_interval_steps: int,
    max_interval_steps: int,
    base_velocity_mps: float,
    base_thickness_mm: float,
    residual_interval_max: float,
    residual_velocity_max: float,
    residual_thickness_mm_max: float,
    normalize_actions: bool,
    append_scheduler_features: bool,
    reset_to_cached_warm_state: bool,
) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        runtime_steps = max(int(episode_steps), int(warmup_steps)) if reset_to_cached_warm_state else int(warmup_steps + episode_steps)
        env: gym.Env = ToraxPelletBatchEnv(
            torax_config_path=torax_config,
            batch_size=batch_size,
            max_steps=runtime_steps,
            device=device,
            randomize_on_reset=False,
            warmup_steps=0,
            seed=seed,
        )
        env = WarmStartFullResidualWrapper(
            env,
            warmup_steps=warmup_steps,
            episode_steps=episode_steps,
            sim_steps_per_rl_step=sim_steps_per_rl_step,
            reset_to_cached_warm_state=reset_to_cached_warm_state,
            base_interval_steps=base_interval_steps,
            inject_duration=inject_duration,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
            base_velocity_mps=base_velocity_mps,
            base_thickness_mm=base_thickness_mm,
            residual_interval_range=(-residual_interval_max, residual_interval_max),
            residual_velocity_range=(-residual_velocity_max, residual_velocity_max),
            residual_thickness_mm_range=(-residual_thickness_mm_max, residual_thickness_mm_max),
            append_scheduler_features=append_scheduler_features,
        )
        if normalize_actions:
            env = ActionNormalizeWrapper(env)
        if num_stack > 1:
            env = BatchFrameStackWrapper(env, num_stack=num_stack)
        return env

    return _make


def zero_full_residual_action(batch_size: int) -> np.ndarray:
    return np.zeros((batch_size, 3), dtype=np.float32)
