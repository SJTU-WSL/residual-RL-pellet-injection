# Residual RL for Tokamak Pellet Injection

这个仓库用于研究托卡马克等离子体中的 pellet injection 控制，核心目标是把 `torax` 传输模拟器与 pellet 沉积模型结合起来，在批量并行仿真上训练一个 residual reinforcement learning 控制器。

当前仓库里最重要的训练入口是 `rl_residual_lab/`。旧版 README 里提到的 `rl_labs/`、`train_PPO.py`、`eval_PPO.py` 等路径已经不是当前实现。

## 当前核心思路

`rl_residual_lab` 不是直接让策略网络从零决定“是否注入 + 注入参数”，而是采用固定节拍基线控制器 + residual policy 的形式：

- 基线调度器固定决定注入节拍：每隔 `inject_every` 个控制步触发一次，持续 `inject_duration` 步。
- RL 策略只输出两维 residual 动作：
  - `delta_velocity`
  - `delta_thickness`
- 真正送入物理环境的动作仍然是三维：
  - `trigger`
  - `velocity`
  - `thickness`
- 当不在注入窗口时，环境自动关闭注入，residual 动作不会生效。
- 训练默认先做 `warmup_steps=2000` 的冷启动预热。由于 `config/ITER.py` 使用 `fixed_dt = 0.001`，这对应约 2 秒物理时间。
- 训练环境会缓存 warm-start 状态，后续 reset 直接从 2 秒后的等离子体状态开始，避免每个 episode 都重复走启动瞬态。

这套设计主要在 [rl_residual_lab/residual_env.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/residual_env.py)、[rl_residual_lab/train_residual.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/train_residual.py)、[rl_residual_lab/eval_residual.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/eval_residual.py) 中实现。

## 仓库结构

- `rl_residual_lab/`
  - residual RL 的训练、评估与封装脚本
  - 训练入口：`train_residual.py`
  - 评估入口：`eval_residual.py`
  - 便捷脚本：`run_train_residual.sh`、`run_train_residual_cpu.sh`、`run_eval_residual_model.sh`
- `RL/`
  - 通用 gym/gymnasium 环境、reward、VecEnv 封装
  - `ToraxPelletBatchEnv` 把一个内部 batch 环境暴露给 SB3
- `simulator/`
  - 底层物理模拟
  - `torax_simulator.py` 负责 JAX/Torax 侧等离子体演化
  - `FPAD_simulator.py` 负责 pellet 沉积模拟
- `Examples/`
  - 不经过 SB3 的并行仿真脚本，适合先验证模拟器是否能跑通
- `visualization/`
  - PyQt 可视化桌面程序，用于手工设置参数并查看仿真曲线和剖面
- `config/ITER.py`
  - 默认 ITER 场景配置
- `requirement.txt`
  - Python 依赖

## 环境依赖

推荐先使用项目脚本默认的 conda 环境名：

```bash
conda activate Nuclear
python3 -m pip install -r requirement.txt
```

主要依赖包括：

- `torch`
- `jax`, `jaxlib`
- `torax`
- `stable-baselines3`
- `PyQt5`, `pyqtgraph`（可视化可选）

仓库里还包含部分预编译扩展与平台相关二进制文件，首次在新环境运行时，建议先从 `Examples/` 或小 batch 的 CPU 训练开始确认依赖和本地运行时都正常。

## `rl_residual_lab` 训练机制

### 观测

基础观测来自 [RL/env.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/env.py)，会把 `TransportSimulator` 的状态树和输出树展开成 `(B, D)` 的数值特征。

如果启用 `--append-schedule-features`，还会额外拼接 4 个与固定注入节拍相关的特征：

- `sin(phase)`
- `cos(phase)`
- `inject_now`
- `control_active`

### 动作

Residual 环境对 SB3 暴露的是二维动作：

- `delta_velocity`，默认范围 `[-50, 50] m/s`
- `delta_thickness`，默认范围 `[-0.5, 0.5] mm`

环境内部再把它和基线注入参数合成：

- 基线速度：`--baseline-velocity`，默认 `300 m/s`
- 基线厚度：`--baseline-thickness-mm`，默认 `2.0 mm`

### 并行方式

训练时底层环境本身就是一个 batch 环境，形状是 `(B, D)`。`BatchAsVecEnv` 会把这个 batch 解释成 SB3 视角下的 `B` 个并行环境，从而直接让 PPO 使用一批仿真样本。

### Reward 与安全终止

Reward 定义在 [RL/reward.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/reward.py)，核心会综合考虑：

- 三乘积 `n T tau_E`
- `Q_fusion`
- Greenwald fraction 密度带宽
- 电子/离子体平均温度带宽
- pellet 使用量与注入速度惩罚

出现明显非物理或数值异常时会触发 unsafe termination，例如：

- 非有限数值
- 非正密度或温度
- `fgw_n_e_volume_avg >= 1.0`
- 核心温度过低或过高

## 快速开始

### 1. 先跑原始模拟器，不走 RL

无注入基线：

```bash
python3 Examples/no_injection.py --device cpu --batch-size 2 --run-steps 10
```

随机注入：

```bash
python3 Examples/random_injection.py --device cpu --batch-size 2 --run-steps 10 --inject-prob 0.05
```

固定节拍注入：

```bash
python3 Examples/interval_injiction.py --device cpu --batch-size 2 --run-steps 10 --inject-every 100 --inject-duration 1 --inject-fraction 0.2
```

这些脚本会把摘要和参数保存到 `example_logs/`。

### 2. 启动 residual PPO 训练

最直接的方式是运行训练脚本：

```bash
python3 rl_residual_lab/train_residual.py \
  --torax-config config/ITER.py \
  --device cpu \
  --batch-size 4 \
  --max-steps 1000 \
  --warmup-steps 2000 \
  --inject-every 100 \
  --inject-duration 1 \
  --baseline-velocity 300 \
  --baseline-thickness-mm 2.0 \
  --residual-velocity-max 50.0 \
  --residual-thickness-mm-max 0.5 \
  --total-timesteps 10000
```

如果希望直接复用仓库里的默认训练参数：

```bash
bash rl_residual_lab/run_train_residual.sh
```

CPU 调试版本：

```bash
bash rl_residual_lab/run_train_residual_cpu.sh
```

训练输出：

- 日志目录：`rl_logs/<run_name>/`
- 模型目录：`rl_models/<run_name>/`
- 典型文件：
  - `config.json`
  - `monitor.csv`
  - `tensorboard/`
  - `train.stdout.log`
  - `checkpoint_*.zip`
  - `final_model.zip`

### 3. 评估 residual 模型

评估某个训练好的模型：

```bash
python3 rl_residual_lab/eval_residual.py \
  --model-path rl_models/<run_name>/final_model.zip \
  --device cpu \
  --batch-size 1 \
  --eval-steps 10000 \
  --deterministic
```

或使用封装脚本：

```bash
bash rl_residual_lab/run_eval_residual_model.sh rl_models/<run_name>/final_model.zip cpu
```

如果不传 `--model-path`，评估脚本会退化为“零 residual”的固定节拍基线控制器，用来和训练后的策略做对照。

评估输出保存在 `eval_logs/`，主要是：

- `<run_name>_config.json`
- `<run_name>_summary.json`

## 可视化界面

仓库带了一个 PyQt 桌面程序，可以做非 RL 的批量仿真和结果可视化：

```bash
python3 visualization/app.py
```

当前界面支持：

- 选择 `config/ITER.py` 或其他配置文件
- 设置 batch size、仿真步数、注入间隔、速度、厚度
- 选择要监控的标量和矢量物理量
- 查看标量时间序列、径向 1D profile、极向截面图
- 保存运行结果到 `visualization/runs/<run_id>/`

## 当前默认配置

默认实验配置位于 [config/ITER.py](/Users/sheldonwang/residual-RL-pellet-injection/config/ITER.py)：

- 时间步长：`1 ms`
- 默认总时长：`t_final = 100`
- 主要场景：ITER-like geometry
- 已经针对温度、密度和输运做过一轮手工调参

如果你改了配置文件，`Examples/`、`visualization/`、`rl_residual_lab/` 这三条工作流都可以直接复用新的配置路径。

## 需要注意的点

- `interval_injiction.py` 保留了历史拼写 `injiction`，文件名不是笔误。
- 训练脚本默认假设 `stable-baselines3` 已安装，否则会直接报错退出。
- `run_train_residual.sh` 会设置 `.runtime_cache/` 下的 matplotlib 和 JAX 编译缓存目录。
- 这个仓库的 RL 训练目前围绕 `rl_residual_lab` 展开，旧 README 里提到的 `rl_labs/` 已不再适用。

## 致谢

底层传输模拟部分基于 [Google DeepMind Torax](https://github.com/google-deepmind/torax) 生态做了适配与扩展，pellet injection 模块与 RL 控制逻辑为本仓库的研究实现。
