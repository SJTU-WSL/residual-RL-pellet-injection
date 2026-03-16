"""Train PPO on timing/velocity/thickness full residual control."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecMonitor
except Exception as exc:  # pragma: no cover
    raise ImportError("stable-baselines3 is required to run train_full_residual.py") from exc

from RL.vec_env import BatchAsVecEnv
from rl_full_residual_lab.full_residual_env import make_full_residual_env_fn


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


class FullResidualStatsLogger(BaseCallback):
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            if rewards.size:
                self.logger.record_mean("rollout/reward_step_mean", float(np.mean(rewards)))
                self.logger.record_mean("rollout/reward_step_std", float(np.std(rewards)))
                self.logger.record("rollout/reward_step_env0", float(rewards[0]))
                self.logger.record("rollout/reward_step_last", float(rewards[-1]))

        infos = self.locals.get("infos", [])
        if infos:
            for key in (
                "inject_now",
                "plan_applied",
                "planner_due",
                "planned_interval_steps",
                "residual_interval_steps",
                "next_inject_in_steps",
                "planned_velocity_mps",
                "planned_thickness_mm",
                "residual_velocity_mps",
                "residual_thickness_mm",
                "actual_velocity_mps",
                "actual_thickness_mm",
            ):
                values = [float(info[key]) for info in infos if key in info]
                if values:
                    self.logger.record_mean(f"rollout/{key}_mean", float(np.mean(values)))

            episode_rewards = []
            for info in infos:
                episode = info.get("episode")
                if isinstance(episode, dict) and episode.get("r") is not None:
                    episode_rewards.append(float(episode["r"]))
            if episode_rewards:
                self.logger.record("episode/reward_raw", episode_rewards[0])
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with timing/velocity/thickness residual control.")
    parser.add_argument("--torax-config", default="config/ITER.py")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000, help="Post-warmup control horizon.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--sim-steps-per-rl-step", type=int, default=10)
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

    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--ppo-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--initial-residual-std", type=float, default=0.2)

    parser.add_argument("--log-dir", default="rl_logs")
    parser.add_argument("--save-dir", default="rl_models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--num-stack", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def initialize_zero_full_residual_policy(model: PPO, initial_std: float) -> None:
    action_net = getattr(model.policy, "action_net", None)
    log_std = getattr(model.policy, "log_std", None)
    if action_net is None or log_std is None or action_net.bias is None:
        print("[Warn] policy does not expose action_net/log_std, skip initialization.")
        return
    if action_net.bias.numel() != 3 or log_std.numel() != 3:
        print(
            f"[Warn] expected 3-D action policy, got bias={action_net.bias.numel()} "
            f"log_std={log_std.numel()}, skip initialization."
        )
        return

    residual_std = max(float(initial_std), 1e-4)
    with torch.no_grad():
        action_net.weight.zero_()
        action_net.bias.zero_()
        log_std.fill_(float(np.log(residual_std)))

    print(f"[Init] Zero full-residual policy initialized with std={residual_std:.4f}")


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log_root = resolve_project_path(args.log_dir)
    save_root = resolve_project_path(args.save_dir)
    torax_config = str(resolve_project_path(args.torax_config))
    log_root.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"ppo_full_residual_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = log_root / run_name
    model_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "torax_config": torax_config,
                "run_dir": str(run_dir),
                "model_dir": str(model_dir),
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
        reset_to_cached_warm_state=True,
    )
    base_env = env_fn()
    vec_env = BatchAsVecEnv(base_env)
    vec_env = VecMonitor(vec_env, filename=str(run_dir / "monitor.csv"))

    tensorboard_log: str | None = str(run_dir / "tensorboard")
    try:
        import tensorboard  # noqa: F401
    except Exception:
        tensorboard_log = None
        print("[Warn] tensorboard is not installed, disable tensorboard logging.")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=args.device,
        n_steps=args.ppo_steps,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        vf_coef=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.Tanh,
        ),
    )
    initialize_zero_full_residual_policy(model, args.initial_residual_std)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(model_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="tensorboard",
        callback=[checkpoint_cb, FullResidualStatsLogger()],
        log_interval=max(1, args.log_interval),
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    vec_env.close()

    print(f"Training finished. Final full residual model saved to: {final_path}.zip")
    print(f"Run logs saved to: {run_dir}")
    print(f"Model checkpoints saved to: {model_dir}")


if __name__ == "__main__":
    main()
