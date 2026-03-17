# Full Residual RL for Tokamak Pellet Injection

面向托卡马克 pellet injection 控制的强化学习与物理仿真项目。该仓库将 `torax` 等离子体输运模拟、pellet 沉积模型和 PPO 训练流程集成在同一套 batch 仿真框架中，用于研究注入时机、注入速度和 pellet 厚度的联合学习。

## Overview

项目核心是一个 full residual controller。策略并不直接输出底层物理执行器的完整动作，而是围绕一个可解释的基线计划，学习三类残差修正：

- 下一次注入间隔
- 注入速度
- pellet 厚度

环境据此生成事件级注入计划，并在多个 simulator step 上执行。这种设计兼顾了物理先验、控制可解释性和训练稳定性，适合做调度策略、密度控制和聚变性能对比实验。

## Core Features

- Full residual action space：同时学习 timing, velocity, thickness 三维控制量
- Event-level scheduling：策略在 planner 触发时生成下一次注入计划，而不是逐步硬编码触发信号
- Warm-start training：训练默认先经过约 2 秒物理预热，再从稳定状态开始控制
- Macro-step RL rollout：支持 `sim_steps_per_rl_step`，用单次策略决策驱动多个物理步
- Batched simulation：底层环境原生支持 batch，并通过 SB3 适配为并行环境
- Physics-aware reward：奖励结合三乘积、密度/温度约束
- Visualization workflow：提供 PyQt 可视化界面做并行仿真与结果查看

## Repository Layout

- `rl_lab/`
  PPO 训练、评估和 full residual 环境封装

- `rl_residual_lab/`
  固定间隔残差 RL 训练/评估（velocity + thickness 残差）

- `baseline_sweep/`
  基线注入参数网格搜索（inject_every × velocity × thickness）

- `RL/`
  通用 batch 环境、reward 和 VecEnv 适配器

- `simulator/`
  物理模拟核心，包括 `torax` 输运求解和 pellet 沉积实现

- `Examples/`
  不经过 RL 的并行仿真脚本，用于快速验证模拟器行为

- `visualization/`
  PyQt 图形界面，用于交互式仿真和结果可视化

- `config/ITER.py`
  默认 ITER-like 配置

## Installation

推荐使用项目脚本默认的 conda 环境：

```bash
conda activate Nuclear
python3 -m pip install -r requirement.txt
```

主要依赖：

- `torch`
- `jax`
- `jaxlib`
- `torax`
- `stable-baselines3`
- `PyQt5`
- `pyqtgraph`

首次部署到新机器时，建议先从 `Examples/` 中的小规模 CPU 任务开始，确认 JAX 后端和本地扩展都能正常工作。

## Full Residual Environment

full residual 环境定义在 `rl_lab/full_residual_env.py`。

### Action Space

策略输出三维残差动作：

- `delta_interval_steps`
- `delta_velocity_mps`
- `delta_thickness_mm`

环境内部会把残差与基线计划组合，得到下一次注入事件的完整参数。默认基线包括：

- `base_interval_steps = 100`
- `baseline_velocity = 300 m/s`
- `baseline_thickness_mm = 2.0`

默认残差范围包括：

- 间隔修正 `[-20, 20]` steps
- 速度修正 `[-50, 50] m/s`
- 厚度修正 `[-0.5, 0.5] mm`

### Planner Logic

环境维护一个 event planner：

- 当 `planner_due` 为真时，策略给出下一次注入计划
- 计划包含下一次注入间隔、速度和厚度
- 到达计划时刻时触发注入
- 注入事件执行后，planner 重新进入待规划状态

因此，注入时机已经成为 RL 直接学习的一部分，而不是固定节拍上的外部规则。

### Warm Start and Macro Steps

训练默认先执行 `warmup_steps = 2000`。结合 `config/ITER.py` 中的 `fixed_dt = 0.001`，对应约 2 秒物理时间。

预热结束后，环境缓存等离子体状态。训练 episode reset 时直接回到 warm-start 状态，从而减少冷启动瞬态对 PPO 的干扰。

此外，`sim_steps_per_rl_step` 允许一次策略决策执行多个物理步。这有助于降低决策频率，并让 RL 更像事件级调度器而不是毫秒级开关控制器。

### Observations

基础观测由 `RL/env.py` 提供，会把输运状态和诊断输出展开为 `(B, D)` 的数值特征。

启用 `--append-scheduler-features` 后，还会附加调度相关特征，包括：

- 调度相位的正余弦编码
- 距离下一次注入的归一化时间
- 当前计划间隔
- planner 是否等待新计划
- 当前控制是否已激活
- 最近一次实际注入的速度和厚度比例

### Reward and Safety

奖励定义在 `RL/reward.py`，主要考虑：

- 聚变三乘积 `n T tau_E`（权重 0.40）
- Greenwald fraction 约束（权重 0.18 + 安全惩罚）
- 电子/离子体平均温度范围（各 0.14）
- Greenwald 安全裕度（0.14 正 / 0.40 罚）

环境会对明显异常的状态提前终止，例如：

- 非有限数值
- 非正密度或温度
- Greenwald fraction 过高
- 核心温度异常偏低或偏高

## Quick Start

### Raw Simulator Examples

无注入基线：

```bash
python3 Examples/no_injection.py --device cpu --batch-size 2 --run-steps 10
```

随机注入：

```bash
python3 Examples/random_injection.py --device cpu --batch-size 2 --run-steps 10 --inject-prob 0.05
```

固定间隔注入：

```bash
python3 Examples/interval_injiction.py --device cpu --batch-size 2 --run-steps 10 --inject-every 100 --inject-duration 1 --inject-fraction 0.2
```

这些脚本会将摘要和配置保存到 `example_logs/`。

### PPO Training

训练入口位于 `rl_lab/train_full_residual.py`。

一个最小化示例：

```bash
python3 rl_lab/train_full_residual.py \
  --torax-config config/ITER.py \
  --device cpu \
  --batch-size 4 \
  --max-steps 1000 \
  --warmup-steps 2000 \
  --sim-steps-per-rl-step 10 \
  --base-interval-steps 100 \
  --min-interval-steps 20 \
  --max-interval-steps 200 \
  --baseline-velocity 300 \
  --baseline-thickness-mm 2.0 \
  --residual-interval-max 20.0 \
  --residual-velocity-max 50.0 \
  --residual-thickness-mm-max 0.5 \
  --total-timesteps 10000
```

固定间隔残差训练（仅学习 velocity + thickness 残差）：

```bash
bash rl_residual_lab/run_train_residual.sh
```

训练输出默认保存在：

- `rl_logs/<run_name>/`
- `rl_models/<run_name>/`

常见产物包括：

- `config.json`
- `monitor.csv`
- `tensorboard/`
- `checkpoint_*.zip`
- `final_model.zip`

### Policy Evaluation

评估入口位于 `rl_lab/eval_full_residual.py`。

评估一个训练好的模型：

```bash
python3 rl_lab/eval_full_residual.py \
  --model-path rl_models/<run_name>/final_model.zip \
  --device cpu \
  --batch-size 1 \
  --eval-steps 10000 \
  --sim-steps-per-rl-step 10 \
  --deterministic
```

如果不提供 `--model-path`，评估脚本会使用零残差策略，作为 full residual 基线控制器的对照。

评估结果默认写入 `eval_logs/`，包括：

- `<run_name>_config.json`
- `<run_name>_summary.json`

## Baseline Parameter Sweep

Grid search over baseline injection parameters to find the optimal fixed schedule
before training the residual RL agent.

### Quick Start

```bash
bash baseline_sweep/run_sweep.sh
```

### Manual Usage

```bash
# Run evaluation grid (CPU, supports --resume for checkpointing)
python3 baseline_sweep/sweep_eval.py \
  --device cpu --batch-size 16 \
  --warmup-steps 2000 --control-steps 8000 \
  --resume

# Custom parameter grid
python3 baseline_sweep/sweep_eval.py \
  --inject-every-values 50,100,200 \
  --velocity-values 200,300,400 \
  --thickness-mm-values 1.5,2.0,3.0

# Visualize results
python3 baseline_sweep/sweep_visualize.py \
  --csv baseline_sweep/eval_logs/sweep_summary.csv \
  --top-n 20
```

### Default Grid (6×6×6 = 216 combinations)

| Parameter | Values |
|-----------|--------|
| inject_every (steps) | 50, 75, 100, 125, 150, 200 |
| velocity (m/s) | 150, 200, 250, 300, 400, 500 |
| thickness (mm) | 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 |

### Output

Results saved to `baseline_sweep/eval_logs/`:
- `combo_ie{X}_v{Y}_t{Z}.json` — per-combination metrics
- `sweep_summary.csv` — all combinations in one table
- `sweep_config.json` — sweep configuration
- `heatmap_*.png` — heatmap grids sliced by inject_every
- `sweep_sensitivity.png` — marginal sensitivity curves
- `sweep_top20.csv` — top 20 configurations ranked by reward

## Visualization

启动桌面程序：

```bash
python3 visualization/app.py
```

图形界面支持：

- 选择配置文件
- 设置 batch size、仿真步数、注入间隔、速度与厚度
- 选择标量和矢量诊断量
- 查看时间序列、径向 profile 和极向截面图
- 保存运行结果到 `visualization/runs/<run_id>/`

## Default Configuration

默认实验场景位于 `config/ITER.py`，当前包含：

- `1 ms` 时间步长
- `t_final = 100`
- ITER-like 几何参数
- 面向密度、温度和热输运实验的源项与输运配置

你可以替换为其他配置文件，并在 `Examples/`、`rl_lab/` 和 `visualization/` 三条工作流中复用。

## Main Entry Points

- `rl_lab/train_full_residual.py`
- `rl_lab/eval_full_residual.py`
- `rl_lab/full_residual_env.py`
- `rl_residual_lab/train_residual.py`
- `rl_residual_lab/eval_residual.py`
- `baseline_sweep/sweep_eval.py`
- `baseline_sweep/sweep_visualize.py`
- `RL/env.py`
- `RL/reward.py`
- `visualization/app.py`

## Acknowledgement

底层输运模拟工作流建立在 [Google DeepMind Torax](https://github.com/google-deepmind/torax) 相关生态之上，pellet injection 与 full residual RL 控制部分为本项目的研究实现。
