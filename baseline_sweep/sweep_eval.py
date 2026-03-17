"""Grid search over baseline injection parameters (inject_every, velocity, thickness).

Runs each combination through the raw simulator (no SB3) and records
physics diagnostics + reward for later comparison.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.FPAD_simulator import PelletSimulator
from simulator.torax_simulator import TransportSimulator
from RL.reward import compute_reward, evaluate_unsafe_conditions, REWARD_INFO_KEYS


# ── default parameter grid ──
DEFAULT_INJECT_EVERY = [50, 75, 100, 125, 150, 200]
DEFAULT_VELOCITY = [150, 200, 250, 300, 400, 500]
DEFAULT_THICKNESS_MM = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tensor_mean(value: torch.Tensor | np.ndarray | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def _combo_filename(inject_every: int, velocity: float, thickness_mm: float) -> str:
    return f"combo_ie{inject_every}_v{velocity:.0f}_t{thickness_mm:.1f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline injection parameter sweep.")
    p.add_argument("--torax-config", default="config/ITER.py")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--control-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jax-cache-dir", default=".runtime_cache/jax_cache")
    p.add_argument("--log-dir", default="baseline_sweep/eval_logs")
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--resume", action="store_true", help="Skip combos with existing results.")
    p.add_argument(
        "--inject-every-values", type=str, default=None,
        help="Comma-separated inject_every values (default: 50,75,100,125,150,200)",
    )
    p.add_argument(
        "--velocity-values", type=str, default=None,
        help="Comma-separated velocity values in m/s (default: 150,200,250,300,400,500)",
    )
    p.add_argument(
        "--thickness-mm-values", type=str, default=None,
        help="Comma-separated thickness values in mm (default: 1.0,1.5,2.0,2.5,3.0,4.0)",
    )
    return p.parse_args()


def parse_float_list(s: str | None, default: list) -> list:
    if s is None:
        return default
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list(s: str | None, default: list) -> list:
    if s is None:
        return default
    return [int(x.strip()) for x in s.split(",")]


def run_single_combo(
    env: TransportSimulator,
    pellet_sim: PelletSimulator,
    *,
    batch_size: int,
    warmup_steps: int,
    control_steps: int,
    inject_every: int,
    velocity_mps: float,
    thickness_mm: float,
    device: torch.device,
    log_interval: int,
) -> dict:
    """Run warmup + control for one parameter combo. Return aggregated metrics."""

    env.reset()

    # Shared tensors
    zero_triggers = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    zero_rates = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    default_locs = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
    default_widths = torch.full((batch_size,), 0.05, dtype=torch.float32, device=device)

    current_triggers = zero_triggers.clone()
    current_locs = default_locs.clone()
    current_widths = default_widths.clone()
    current_rates = zero_rates.clone()

    # ── Warmup: no injection ──
    for _ in range(warmup_steps):
        env.step(current_triggers, current_locs, current_widths, current_rates)

    # Sync pellet sim plasma state after warmup
    T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = env.get_plasma_tensor()
    pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)

    # Pre-compute pellet trajectory for this velocity/thickness (same every injection)
    vel_tensor = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    vel_tensor[:, 0] = velocity_mps
    thick_tensor = torch.full((batch_size,), thickness_mm * 1e-3, dtype=torch.float32, device=device)
    ref_loc, ref_width, ref_rate = pellet_sim.simulate_pellet_injection(
        batch_size=batch_size, velocity=vel_tensor, thickness=thick_tensor,
    )

    # ── Control phase: fixed-interval injection ──
    # history collectors
    step_rewards: list[np.ndarray] = []
    step_diagnostics: dict[str, list[np.ndarray]] = {k: [] for k in REWARD_INFO_KEYS}
    step_triple: list[np.ndarray] = []
    unsafe_count = 0

    start_time = time.time()

    for step in range(control_steps):
        # Decide injection for NEXT step
        in_window = (step % inject_every) == 0
        if in_window:
            next_triggers = torch.ones((batch_size,), dtype=torch.bool, device=device)
            next_locs = ref_loc.clone()
            next_widths = ref_width.clone()
            next_rates = ref_rate.clone()
        else:
            next_triggers = zero_triggers.clone()
            next_locs = default_locs.clone()
            next_widths = default_widths.clone()
            next_rates = zero_rates.clone()

        # Step transport with CURRENT injection
        env.step(current_triggers, current_locs, current_widths, current_rates)

        # Update pellet sim plasma
        T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = env.get_plasma_tensor()
        pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)

        # Re-compute pellet trajectory (plasma state may have changed)
        if in_window:
            ref_loc, ref_width, ref_rate = pellet_sim.simulate_pellet_injection(
                batch_size=batch_size, velocity=vel_tensor, thickness=thick_tensor,
            )
            next_locs = ref_loc.clone()
            next_widths = ref_width.clone()
            next_rates = ref_rate.clone()

        # Collect diagnostics
        diagnostics = env.get_diagnostics()
        info: dict[str, np.ndarray] = {}
        for key in REWARD_INFO_KEYS:
            arr = tensor_to_numpy(diagnostics[key])
            info[key] = arr
            step_diagnostics[key].append(arr)

        # Compute reward via RL.reward (dummy obs, reward only uses info)
        dummy_obs = np.zeros((batch_size, 1), dtype=np.float32)
        dummy_action = np.zeros((batch_size, 2), dtype=np.float32)
        reward_batch = compute_reward(dummy_obs, dummy_action, dummy_obs, info)
        if not isinstance(reward_batch, np.ndarray):
            reward_batch = np.full((batch_size,), reward_batch, dtype=np.float32)
        step_rewards.append(reward_batch)

        # Triple product
        ne = np.clip(info["n_e_volume_avg"], 1e-12, None)
        te = np.clip(info["T_e_volume_avg"], 1e-6, None)
        ti = np.clip(info["T_i_volume_avg"], 1e-6, None)
        tau = np.clip(info["tau_E"], 1e-6, None)
        t_avg = 0.5 * (te + ti)
        triple = ne * t_avg * tau
        step_triple.append(triple)

        # Safety check
        unsafe_mask, _ = evaluate_unsafe_conditions(info, batch_size)
        unsafe_count += int(np.sum(unsafe_mask))

        # Advance
        current_triggers = next_triggers
        current_locs = next_locs
        current_widths = next_widths
        current_rates = next_rates

        if step % max(1, log_interval) == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0.0
            r_mean = float(np.mean(reward_batch))
            fgw = float(np.mean(info["fgw_n_e_volume_avg"]))
            print(
                f"  step {step:5d}/{control_steps} | "
                f"reward={r_mean:+.4f} | fgw={fgw:.3f} | "
                f"triple={float(np.mean(triple)):.3e} | SPS={sps:.1f}"
            )

    elapsed = time.time() - start_time

    # ── Aggregate ──
    all_rewards = np.concatenate(step_rewards)  # (control_steps * batch_size,)
    all_triple = np.concatenate(step_triple)

    result: dict = {
        "inject_every": inject_every,
        "velocity_mps": velocity_mps,
        "thickness_mm": thickness_mm,
        "warmup_steps": warmup_steps,
        "control_steps": control_steps,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "sps": control_steps / elapsed if elapsed > 0 else 0.0,
        "unsafe_step_count": unsafe_count,
        # reward
        "reward_mean": float(np.nanmean(all_rewards)),
        "reward_std": float(np.nanstd(all_rewards)),
        "reward_min": float(np.nanmin(all_rewards)),
        "reward_max": float(np.nanmax(all_rewards)),
        # triple product
        "triple_product_mean": float(np.nanmean(all_triple)),
        "triple_product_std": float(np.nanstd(all_triple)),
        "triple_product_min": float(np.nanmin(all_triple)),
        "triple_product_max": float(np.nanmax(all_triple)),
    }

    # Per diagnostic key
    for key in REWARD_INFO_KEYS:
        arr = np.concatenate(step_diagnostics[key])
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            result[f"{key}_mean"] = float(np.mean(finite))
            result[f"{key}_std"] = float(np.std(finite))
            result[f"{key}_min"] = float(np.min(finite))
            result[f"{key}_max"] = float(np.max(finite))
        else:
            result[f"{key}_mean"] = float("nan")
            result[f"{key}_std"] = float("nan")
            result[f"{key}_min"] = float("nan")
            result[f"{key}_max"] = float("nan")

    return result


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inject_every_values = parse_int_list(args.inject_every_values, DEFAULT_INJECT_EVERY)
    velocity_values = parse_float_list(args.velocity_values, DEFAULT_VELOCITY)
    thickness_mm_values = parse_float_list(args.thickness_mm_values, DEFAULT_THICKNESS_MM)

    combos = list(itertools.product(inject_every_values, velocity_values, thickness_mm_values))
    total = len(combos)
    print(f"Parameter sweep: {len(inject_every_values)} x {len(velocity_values)} x {len(thickness_mm_values)} = {total} combinations")
    print(f"  inject_every : {inject_every_values}")
    print(f"  velocity     : {velocity_values}")
    print(f"  thickness_mm : {thickness_mm_values}")
    print(f"  warmup={args.warmup_steps}, control={args.control_steps}, batch={args.batch_size}")
    print()

    # JAX cache
    cache_path = resolve_project_path(args.jax_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Initialize simulators once
    print("[Init] Creating TransportSimulator ...")
    env = TransportSimulator(str(resolve_project_path(args.torax_config)), total_batch_size=args.batch_size)
    pellet_sim = PelletSimulator(device=str(device))
    print("[Init] Simulators ready.\n")

    all_results: list[dict] = []
    sweep_start = time.time()

    for idx, (ie, vel, thk) in enumerate(combos):
        combo_name = _combo_filename(ie, vel, thk)
        result_path = output_dir / f"{combo_name}.json"

        if args.resume and result_path.exists():
            print(f"[{idx+1}/{total}] SKIP (exists) inject_every={ie}, velocity={vel}, thickness={thk}")
            with result_path.open("r") as f:
                all_results.append(json.load(f))
            continue

        print(f"[{idx+1}/{total}] inject_every={ie}, velocity={vel} m/s, thickness={thk} mm")

        result = run_single_combo(
            env, pellet_sim,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            control_steps=args.control_steps,
            inject_every=ie,
            velocity_mps=vel,
            thickness_mm=thk,
            device=device,
            log_interval=args.log_interval,
        )

        # Save per-combo result
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)
        elapsed_total = time.time() - sweep_start
        avg_per_combo = elapsed_total / (idx + 1)
        eta = avg_per_combo * (total - idx - 1)
        print(
            f"  -> reward_mean={result['reward_mean']:+.4f} | "
            f"triple={result['triple_product_mean']:.3e} | "
            f"fgw={result['fgw_n_e_volume_avg_mean']:.3f} | "
            f"elapsed={elapsed_total:.0f}s | ETA={eta:.0f}s ({eta/3600:.1f}h)\n"
        )

    # ── Write summary CSV ──
    if all_results:
        csv_path = output_dir / "sweep_summary.csv"
        fieldnames = list(all_results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"Summary CSV saved to: {csv_path}")

    # Save sweep config
    config_path = output_dir / "sweep_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "inject_every_values": inject_every_values,
                "velocity_values": velocity_values,
                "thickness_mm_values": thickness_mm_values,
                "total_combos": total,
            },
            f,
            indent=2,
        )

    total_elapsed = time.time() - sweep_start
    print(f"\nSweep complete: {total} combos in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
