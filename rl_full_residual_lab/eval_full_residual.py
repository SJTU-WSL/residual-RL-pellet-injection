"""Evaluate timing/velocity/thickness full residual policy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
except Exception as exc:  # pragma: no cover
    raise ImportError("stable-baselines3 is required to run eval_full_residual.py") from exc

from RL.vec_env import BatchAsVecEnv
from rl_full_residual_lab.full_residual_env import make_full_residual_env_fn, zero_full_residual_action


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.array(arr.tolist())
    return arr


def aggregate_infos(info_history: list[dict[str, Any]]) -> dict[str, float]:
    if not info_history:
        return {}

    keys = set()
    for info in info_history:
        keys.update(info.keys())

    summary: dict[str, float] = {}
    for key in sorted(keys):
        values = []
        for info in info_history:
            if key not in info:
                continue
            try:
                arr = to_numpy(info[key]).astype(np.float64).reshape(-1)
            except Exception:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size:
                values.append(arr)
        if not values:
            continue
        merged = np.concatenate(values, axis=0)
        summary[f"{key}_mean"] = float(np.mean(merged))
        summary[f"{key}_std"] = float(np.std(merged))
        summary[f"{key}_min"] = float(np.min(merged))
        summary[f"{key}_max"] = float(np.max(merged))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO with timing/velocity/thickness residual control.")
    parser.add_argument("--model-path", default=None, help="Optional PPO model path (.zip). If omitted, evaluate zero-residual baseline.")
    parser.add_argument("--torax-config", default="config/ITER.py")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000, help="Post-warmup control horizon during eval.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--sim-steps-per-rl-step", type=int, default=10)
    parser.add_argument("--num-stack", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10_000, help="Physical simulator steps to evaluate.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--save-dir", default="eval_logs")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--base-interval-steps", type=int, default=100)
    parser.add_argument("--inject-duration", type=int, default=1)
    parser.add_argument("--min-interval-steps", type=int, default=20)
    parser.add_argument("--max-interval-steps", type=int, default=200)
    parser.add_argument("--baseline-velocity", type=float, default=300.0)
    parser.add_argument("--baseline-thickness-mm", type=float, default=2.0)
    parser.add_argument("--residual-interval-max", type=float, default=20.0)
    parser.add_argument("--residual-velocity-max", type=float, default=50.0)
    parser.add_argument("--residual-thickness-mm-max", type=float, default=0.5)
    parser.add_argument("--append-scheduler-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torax_config = str(resolve_project_path(args.torax_config))
    model_path = resolve_project_path(args.model_path) if args.model_path else None
    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    policy_name = "zero_full_residual_baseline" if model_path is None else "ppo_full_residual"
    run_name = args.run_name or f"{policy_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    set_random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with (save_dir / f"{run_name}_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "torax_config": torax_config,
                "model_path": str(model_path) if model_path else None,
                "save_dir": str(save_dir),
            },
            f,
            indent=2,
        )

    env_fn = make_full_residual_env_fn(
        torax_config=torax_config,
        batch_size=args.batch_size,
        episode_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        sim_steps_per_rl_step=args.sim_steps_per_rl_step,
        num_stack=args.num_stack,
        base_interval_steps=args.base_interval_steps,
        inject_duration=args.inject_duration,
        min_interval_steps=args.min_interval_steps,
        max_interval_steps=args.max_interval_steps,
        base_velocity_mps=args.baseline_velocity,
        base_thickness_mm=args.baseline_thickness_mm,
        residual_interval_max=args.residual_interval_max,
        residual_velocity_max=args.residual_velocity_max,
        residual_thickness_mm_max=args.residual_thickness_mm_max,
        normalize_actions=args.normalize_actions,
        append_scheduler_features=args.append_scheduler_features,
        reset_to_cached_warm_state=False,
    )
    base_env = env_fn()
    vec_env = BatchAsVecEnv(base_env)

    model = PPO.load(str(model_path), env=vec_env, device=args.device) if model_path is not None else None
    obs = vec_env.reset()

    reward_history: list[np.ndarray] = []
    action_history: list[np.ndarray] = []
    info_history: list[dict[str, Any]] = []
    episode_rewards: list[np.ndarray] = []
    episode_lengths: list[int] = []
    current_episode_reward = np.zeros((args.batch_size,), dtype=np.float64)
    current_episode_step = 0
    start_time = time.time()

    target_sim_steps = int(args.eval_steps)
    sim_steps_done = 0
    rl_steps_done = 0

    while sim_steps_done < target_sim_steps:
        if model is None:
            action = zero_full_residual_action(args.batch_size)
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)

        obs, rewards, dones, infos = vec_env.step(action)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(args.batch_size)
        current_episode_reward += rewards
        macro_steps = 0
        if infos:
            try:
                macro_steps = int(np.max([int(info.get("macro_steps_executed", 1)) for info in infos]))
            except Exception:
                macro_steps = 1
        macro_steps = max(1, macro_steps)
        sim_steps_done += macro_steps
        rl_steps_done += 1
        current_episode_step += macro_steps
        reward_history.append(rewards.copy())
        action_history.append(np.asarray(action, dtype=np.float64).copy())
        info_history.extend(infos)

        if rl_steps_done == 1 or rl_steps_done % max(1, args.print_every) == 0:
            elapsed = time.time() - start_time
            sps = sim_steps_done / elapsed if elapsed > 0 else 0.0
            print(
                f"[sim_step {sim_steps_done:5d} | rl_step {rl_steps_done:4d}] "
                f"reward_mean={float(np.mean(rewards)):+.6f} "
                f"action_mean={float(np.mean(action)):+.6f} sps={sps:.1f}"
            )

        if np.any(dones):
            episode_rewards.append(current_episode_reward.copy())
            episode_lengths.append(current_episode_step)
            current_episode_reward.fill(0.0)
            current_episode_step = 0
            obs = vec_env.reset()

    if np.any(np.abs(current_episode_reward) > 0.0):
        episode_rewards.append(current_episode_reward.copy())
        episode_lengths.append(current_episode_step)

    vec_env.close()

    reward_all = np.concatenate([r.reshape(-1) for r in reward_history], axis=0) if reward_history else np.empty((0,))
    action_all = np.concatenate([a.reshape(-1) for a in action_history], axis=0) if action_history else np.empty((0,))
    episode_reward_all = np.concatenate([r.reshape(-1) for r in episode_rewards], axis=0) if episode_rewards else np.empty((0,))
    info_summary = aggregate_infos(info_history)

    summary = {
        "run_name": run_name,
        "policy": policy_name,
        "eval_steps": args.eval_steps,
        "sim_steps_per_rl_step": args.sim_steps_per_rl_step,
        "rl_steps_executed": int(rl_steps_done),
        "sim_steps_executed": int(sim_steps_done),
        "batch_size": args.batch_size,
        "reward_mean": float(np.mean(reward_all)) if reward_all.size else float("nan"),
        "reward_std": float(np.std(reward_all)) if reward_all.size else float("nan"),
        "reward_min": float(np.min(reward_all)) if reward_all.size else float("nan"),
        "reward_max": float(np.max(reward_all)) if reward_all.size else float("nan"),
        "action_mean": float(np.mean(action_all)) if action_all.size else float("nan"),
        "action_std": float(np.std(action_all)) if action_all.size else float("nan"),
        "action_min": float(np.min(action_all)) if action_all.size else float("nan"),
        "action_max": float(np.max(action_all)) if action_all.size else float("nan"),
        "num_episodes": int(len(episode_lengths)),
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "episode_reward_mean": float(np.mean(episode_reward_all)) if episode_reward_all.size else float("nan"),
        "episode_reward_std": float(np.std(episode_reward_all)) if episode_reward_all.size else float("nan"),
        **info_summary,
    }

    summary_path = save_dir / f"{run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Full residual evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
