# NuclearRL

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Backend-JAX-red)](https://github.com/google/jax)

Reinforcement learning and simulation project for fusion pellet injection control.  
Main entry points are organized in `Examples/` (debug-first) and `rl_labs/` (SB3 training/evaluation).

## Project Structure

- `Examples/`: Runs parallel simulation directly with `TransportSimulator + PelletSimulator` (no SB3 wrapper)
  - `no_injection.py`
  - `random_injection.py`
  - `interval_injiction.py`
- `rl_labs/`: PPO training and evaluation entry points
  - `train_PPO.py`
  - `eval_PPO.py`
- `visualization/`: Qt-based desktop console for parallel pellet injection simulation and visualization
  - `app.py`
  - `simulator_worker.py`
  - `plotting.py`
  - `data_models.py`
- `simulator/`: Low-level physics simulator modules
- `RL/`: Environment and reward modules for RL workflows
- `legacy_root_scripts/`: Archived root-level legacy scripts (kept for reference)
- `requirement.txt`: Python dependency list

## Environment Setup

```bash
conda activate Nuclear
python3 -m pip install -r requirement.txt
```

Note: use `python3` as the default command.

## Qt Visualization Console

The project includes a Qt desktop interface for running and visualizing parallel pellet injection simulations.  
All UI code, preview logic, and saved run bundles are located under `visualization/`.

### Launch

```bash
conda activate Nuclear
python3 visualization/app.py
```

### Implemented Features

- Dark gray desktop UI built with Qt for local simulation and debugging
- Three-pane layout:
  - Left: simulation setup and output selection
  - Center: scalar trend plot, 1D radial profile, and 2D poloidal cross-section preview
  - Right: run status, save path, and runtime log
- ITER configuration file path selection (`config/ITER.py` by default)
- Shared pellet injection controls for all parallel batches:
  - Injection interval in `ms`
  - Pellet velocity in `m/s`
  - Pellet thickness in `mm`
- Parallel batch size control from `1` to `2048`
- Simulation length in steps (`dt = 1 ms` in the default ITER setup)
- Output field selection with preset checkboxes plus custom field extension
- Direct simulator execution through `TransportSimulator + PelletSimulator` without SB3
- Real-time preview for the first batch only:
  - Scalar monitoring
  - 1D radial profile visualization
  - 2D reactor cross-section heatmap
- Full multi-batch result saving to:
  - `visualization/runs/<run_id>/run_metadata.json`
  - `visualization/runs/<run_id>/run_results.pkl`
- Runtime cache and temporary visualization state stored under `visualization/.runtime_cache/`

## Examples (Recommended First)

### 1. No Injection Baseline

```bash
python3 Examples/no_injection.py --device cpu --batch-size 2 --run-steps 10
```

### 2. Random Injection

```bash
python3 Examples/random_injection.py --device cpu --batch-size 2 --run-steps 10 --inject-prob 0.05
```

### 3. Interval Injection

```bash
python3 Examples/interval_injiction.py --device cpu --batch-size 2 --run-steps 10 --inject-every 100 --inject-duration 1 --inject-fraction 0.2
```

Default outputs:
- Runtime logs: terminal output
- Config/statistics: `example_logs/*.json`

## RL Training and Evaluation (SB3)

### Training

```bash
python3 rl_labs/train_PPO.py --device cpu --batch-size 2 --max-steps 4 --total-timesteps 1000
```

### Evaluation

```bash
python3 rl_labs/eval_PPO.py --device cpu --model-path rl_models/<model>.zip --batch-size 2 --eval-steps 100
```

Default outputs:
- Training logs and models: `rl_logs/`, `rl_models/`
- Evaluation results: `eval_logs/*.json`

## Acknowledgement / Credits

This project includes code adapted from [Google DeepMind's Torax](https://github.com/google-deepmind/torax), licensed under Apache 2.0. The pellet injection module is a modification developed by .

## Notes

- `interval_injiction.py` keeps the historical spelling (`injiction`).
- `legacy_root_scripts/` is retained only for historical reference. New development should use `Examples/` and `rl_labs/`.
