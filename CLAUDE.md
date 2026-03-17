# CLAUDE.md — Residual RL Pellet Injection

## Project Overview

NuclearRL: 使用残差强化学习控制托卡马克聚变装置中的弹丸注入。基于 ITER 构型，用 Google TORAX (JAX) 做等离子体输运仿真，Stable-Baselines3 PPO 训练控制策略。

**核心思路**: 智能体不从零学控制，而是在固定注入调度(每100步注入一次)上学习小幅残差修正(Δvelocity, Δthickness)。

## Architecture

```
simulator/                  # 物理仿真层
  torax_simulator.py          TransportSimulator — JAX pmap/jit 批量输运仿真
  FPAD_simulator.py           PelletSimulator — PyTorch 弹丸轨迹 (Parks 模型)
  run_loop_sim.py             TORAX 初始化、step function 构建
  src/                        底层物理: 平衡(TorchEquilibrium)、弹丸ODE、层模型

RL/                         # RL 环境层
  env.py                      ToraxPelletBatchEnv(gym.Env) — 核心 Gym 环境
  reward.py                   多分量奖励 + 安全终止条件
  vec_env.py                  BatchAsVecEnv — 将内部 batch 暴露为 SB3 VecEnv
  wrappers/common.py          ActionClip/Normalize, FrameStack, RewardScale

rl_residual_lab/            # 残差 RL 训练/评估 (主要工作区)
  residual_env.py             WarmStartResidualWrapper — 热启动缓存 + 残差动作空间
  train_residual.py           PPO 训练入口 (CLI)
  eval_residual.py            策略评估入口 (CLI)

baseline_sweep/             # 基线参数网格搜索
  sweep_eval.py               6×6×6 网格评估 (inject_every × velocity × thickness)
  sweep_visualize.py          可视化: 热力图 + Top-N 排行 + 边际敏感度
  run_sweep.sh                一键启动脚本

config/ITER.py              # TORAX ITER 物理参数
Examples/                   # 无 RL 的基线测试 (no_injection, random, interval)
visualization/              # PyQt5 仪表盘 (app.py, plotting.py, data_models.py)
```

## Module Dependency Graph

```
Examples/*.py ──→ TransportSimulator + PelletSimulator (直接使用)

ToraxPelletBatchEnv (RL/env.py)
  ├→ TransportSimulator (simulator/torax_simulator.py)
  ├→ PelletSimulator    (simulator/FPAD_simulator.py)
  └→ compute_reward / evaluate_unsafe_conditions (RL/reward.py)

WarmStartResidualWrapper (rl_residual_lab/residual_env.py)
  └→ ToraxPelletBatchEnv + wrappers

train_residual.py → make_residual_env_fn() → BatchAsVecEnv → PPO
eval_residual.py  → 同上 + PPO.load()

visualization/app.py → simulator_worker.py → TransportSimulator + PelletSimulator
```

## Key Design Decisions

### 1. Residual RL
- 固定基线: 每 100 步注入 1 步, velocity=300 m/s, thickness=2.0 mm
- 智能体输出: (Δvelocity, Δthickness), 范围 ±50 m/s / ±0.5 mm
- 仅在注入窗口激活时生效，其余时间不注入

### 2. Cached Warm-Start
- 前 2000 步 (2 sec) 无注入 warmup 到稳态
- 快照保存完整 JAX+Torch 状态，后续 episode 直接恢复
- `WarmupSnapshot` 按 seed 缓存，seed 变化时重建

### 3. JAX + PyTorch 混合
- TORAX (JAX): pmap 多 GPU 批量物理仿真
- 弹丸轨迹 (PyTorch): Parks 模型 ODE
- 数据转换: 经 CPU 中介避免 CUDA 上下文冲突 (`_torch_to_jax_sharded`)

### 4. Batch-as-VecEnv
- 一个 ToraxPelletBatchEnv(batch_size=128) 暴露为 128 个独立 SB3 env
- 全 batch 同步 reset (仅当所有 env 都 done 时)

## RL Environment Spec

### Observation
- TORAX pytree 叶节点展平: Te, ne, Ti, ni 剖面 + 诊断量
- 残差 wrapper 追加 4 维: sin(phase), cos(phase), inject_now, control_active
- NaN→0, Inf→clip ±1e6

### Action
- **Base env**: (B, 3) = [trigger∈(-1,1), velocity∈(100,1000), thickness∈(0.002,0.005)]
- **Residual env**: (B, 2) = [Δvelocity_mps, Δthickness_mm]

### Reward (权重)
| Component | Weight | Target |
|-----------|--------|--------|
| Triple product | 0.40 | log(1 + triple/3e21) |
| Density band | 0.18 | fGW=0.70 ± 0.10 |
| Te band | 0.14 | 10 keV ± 3 keV |
| Ti band | 0.14 | 12 keV ± 3.5 keV |
| Greenwald safety | 0.14 正 / 0.40 罚 | Gaussian@0.72 + softplus>0.90 |

### Safety Termination (penalty = -10.0)
- 温度越界: Te_core∉[0.5,30]keV, Ti_core∉[0.5,35]keV
- 密度崩溃: n_e ≤ 0
- Greenwald 分数 ≥ 1.0
- 诊断量非有限值 (NaN/Inf)
- P_fusion < 0, τ_E ≤ 0

## Training Configuration (PPO defaults)

```
policy:          MlpPolicy, pi=[128,128], vf=[128,128], Tanh
n_steps:         256 (rollout buffer)
batch_size:      1024
n_epochs:        4
lr:              1e-4
gamma:           0.999
clip_range:      0.2
ent_coef:        0.001
max_grad_norm:   0.5
initial_std:     0.25 (零均值初始化)
num_envs:        128 (batch-as-vecenv)
total_timesteps: 2M (default)
checkpoint_freq: 50k steps
```

## Common Commands

```bash
# 基线测试 (无 RL)
python3 Examples/no_injection.py --batch-size 128 --run-steps 2000
python3 Examples/interval_injiction.py --inject-every 100 --run-steps 5000

# 基线参数网格搜索 (CPU, 216 组合, ~40-75h)
bash baseline_sweep/run_sweep.sh
# 或手动分步:
python3 baseline_sweep/sweep_eval.py --device cpu --resume   # 支持断点续跑
python3 baseline_sweep/sweep_visualize.py                     # 生成图表
# 自定义网格:
python3 baseline_sweep/sweep_eval.py \
  --inject-every-values 50,100,200 \
  --velocity-values 200,300,400 \
  --thickness-mm-values 1.5,2.0,3.0

# 训练
bash rl_residual_lab/run_train_residual.sh

# 评估 (训练模型 vs 零残差基线)
python3 rl_residual_lab/eval_residual.py \
  --model-path rl_models/20260313_112014/final_model.zip \
  --eval-steps 10000 --deterministic
python3 rl_residual_lab/eval_residual.py --eval-steps 10000  # 零基线

# 可视化
python3 visualization/app.py

# Shell 脚本
bash rl_residual_lab/run_train_residual.sh
bash rl_residual_lab/run_eval_residual_model.sh
bash rl_residual_lab/run_eval_residual_interval_baseline.sh
```

## Output Directories

```
rl_logs/<timestamp>/        训练日志 (config.json, monitor.csv, tensorboard/)
rl_models/<timestamp>/      模型文件 (final_model.zip, checkpoint_*.zip)
eval_logs/                  评估结果 (*_summary.json, *_config.json)
baseline_sweep/eval_logs/   网格搜索结果 (combo_*.json, sweep_summary.csv, heatmap_*.png)
example_logs/               Examples 输出
visualization/runs/<id>/    可视化运行结果 (run_metadata.json, run_results.pkl)
jax_cache_dir/              JAX 编译缓存
```

## Environment Variables

```bash
JAX_COMPILATION_CACHE_DIR=jax_cache_dir    # JAX 编译缓存
XDG_CACHE_HOME=.runtime_cache              # visualization 缓存
MPLCONFIGDIR=.runtime_cache/matplotlib     # matplotlib 缓存
XLA_PYTHON_CLIENT_PREALLOCATE=false        # 禁止 JAX 预分配全部 GPU 显存
```

## ITER Physics Config (config/ITER.py)

- Plasma: D:T = 50:50, Z_eff=1.6
- Geometry: R=6.2m, a=2.0m, B₀=5.3T, CHEASE (iterhybrid.mat2cols)
- Current: 15 MA, Heating: 20 MW (50% e / 50% i)
- Initial T: ~10 keV core → 0.2 keV edge
- Transport: QLKnn, patched chi_e=3.5/2.0
- Numerics: dt=1ms, t_final=100s
- Pedestal: T_ped=3 keV, n_ped=0.3e20 m⁻³

## Gotchas & Non-Obvious Behavior

1. **Torch↔JAX 转换**: 必须经 CPU 中介, 直接 GPU 转换会导致 CUDA 上下文冲突
2. **多 GPU 分片**: batch_size 必须能被 GPU 数整除, 否则静默回退到单设备 JIT
3. **Warmup 重建**: `WarmStartResidualWrapper` 在 seed 变化时重新执行 2000 步 warmup (慢)
4. **Batch 同步 Reset**: base env 只支持全 batch reset, BatchAsVecEnv 仅在全部 done 时触发
5. **注入时序**: 注入命令和应用有一步延迟 (off-by-one in command vs application)
6. **Obs 顺序**: JAX pytree 展平顺序不保证稳定, 若 TORAX 版本变化可能影响 obs 维度
7. **奖励裁剪**: τ_E clip 到 1e-6, density clip 到 1e-12, 非有限值直接 penalty=-10
8. **文件命名**: `interval_injiction.py` 历史拼写错误, 保留未改
9. **首次运行慢**: JAX 需要编译; 后续运行使用 jax_cache_dir 缓存
10. **单位约定**: TORAX 输出 keV, 弹丸模拟器内部用 eV; density 有 1e20 归一化

## Key Files for Modification

| 需求 | 修改文件 |
|------|----------|
| 探索最优基线参数 | `baseline_sweep/sweep_eval.py` + `sweep_visualize.py` |
| 修改奖励函数 | `RL/reward.py` |
| 修改观测/动作空间 | `RL/env.py` + `rl_residual_lab/residual_env.py` |
| 调整注入基线调度 | `rl_residual_lab/residual_env.py` (inject_every, base_velocity 等) |
| 修改网络结构/超参 | `rl_residual_lab/train_residual.py` |
| 修改物理参数 | `config/ITER.py` |
| 添加新诊断量 | `simulator/torax_simulator.py:get_diagnostics()` + `RL/env.py:_get_obs()` |
| 修改弹丸物理 | `simulator/FPAD_simulator.py` |
| 修改可视化 | `visualization/` 下各文件 |

## Dependencies

Core: jax, jaxlib, torax, torch, stable-baselines3, gymnasium, numpy, scipy
Viz: PyQt5, pyqtgraph, matplotlib, contourpy
Config: omfit-classes (EQDSK utilities)
