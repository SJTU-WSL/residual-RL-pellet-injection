"""Run parallel simulator in no-injection mode (no SB3 wrapper)."""
from __future__ import annotations

import argparse
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


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def tensor_mean(value: torch.Tensor | np.ndarray | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-injection debug run on raw simulator.")
    parser.add_argument("--torax-config", default="config/ITER.py")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-steps", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jax-cache-dir", default="jax_cache_dir")
    parser.add_argument("--log-dir", default="example_logs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"no_injection_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = resolve_project_path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = resolve_project_path(args.jax_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    print(f"[System] JAX Compilation Cache enabled at: {cache_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(device)

    env = TransportSimulator(str(resolve_project_path(args.torax_config)), total_batch_size=args.batch_size)
    pellet_sim = PelletSimulator(device=device)

    current_triggers = torch.zeros((args.batch_size,), dtype=torch.bool, device=device)
    current_locs = torch.full((args.batch_size,), 0.5, dtype=torch.float32, device=device)
    current_widths = torch.full((args.batch_size,), 0.05, dtype=torch.float32, device=device)
    current_rates = torch.zeros((args.batch_size,), dtype=torch.float32, device=device)

    ne_history: list[float] = []
    fgw_history: list[float] = []
    te_core_history: list[float] = []
    ti_core_history: list[float] = []
    inject_history: list[float] = []

    start_time = time.time()

    for step in range(args.run_steps):
        env.step(current_triggers, current_locs, current_widths, current_rates)

        T_e, n_e, _, T_i, n_i, _, sp_D, sp_T = env.get_plasma_tensor()
        pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)

        diagnostics = env.get_diagnostics()
        ne_mean = tensor_mean(diagnostics["n_e_volume_avg"])
        fgw_mean = tensor_mean(diagnostics["fgw_n_e_volume_avg"])
        te_core_mean = tensor_mean(diagnostics["T_e_core"])
        ti_core_mean = tensor_mean(diagnostics["T_i_core"])
        inject_rate = float(current_triggers.float().mean().item())

        ne_history.append(ne_mean)
        fgw_history.append(fgw_mean)
        te_core_history.append(te_core_mean)
        ti_core_history.append(ti_core_mean)
        inject_history.append(inject_rate)

        if step % max(1, args.log_interval) == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:5d} | inject={inject_rate * 100:5.1f}% | "
                f"ne={ne_mean:.3e} | fgw={fgw_mean:.3f} | "
                f"Te_core={te_core_mean:.3f} | Ti_core={ti_core_mean:.3f} | SPS={sps:.1f}"
            )

        # no-injection mode: keep next-step controls as zero injection
        current_triggers.zero_()
        current_rates.zero_()

    elapsed = time.time() - start_time
    summary = {
        "mode": "no_injection",
        "run_name": run_name,
        "steps": args.run_steps,
        "batch_size": args.batch_size,
        "elapsed_sec": elapsed,
        "sps": args.run_steps / elapsed if elapsed > 0 else 0.0,
        "inject_rate_mean": float(np.mean(inject_history)),
        "n_e_volume_avg_mean": float(np.nanmean(ne_history)),
        "fgw_mean": float(np.nanmean(fgw_history)),
        "T_e_core_mean": float(np.nanmean(te_core_history)),
        "T_i_core_mean": float(np.nanmean(ti_core_history)),
    }

    summary_path = output_dir / f"{run_name}_summary.json"
    config_path = output_dir / f"{run_name}_config.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
