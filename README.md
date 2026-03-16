# Residual RL for Tokamak Pellet Injection

面向托卡马克 pellet injection 控制的强化学习与物理仿真项目。仓库将 `torax` 等离子体输运模拟、pellet 沉积模型与 residual reinforcement learning 训练流程集成到统一工作流中，用于研究固定注入节拍上的残差控制策略。

## 项目概览

本项目关注的不是“从零开始决定每一次是否注入”，而是在一个可解释、可复现的基线注入节拍上学习小幅修正：

- 基线控制器负责注入时机
- RL 策略负责修正注入速度与 pellet 厚度
- 环境在批量并行仿真上运行，适合 PPO 等 on-policy 算法
- 训练和评估均可直接复用同一套物理配置

这种设计兼顾了物理先验、训练稳定性和策略可解释性，适合做基线对照、超参数实验和控制策略分析。

## 核心特性

- Residual action design：策略只输出二维残差动作，分别对应速度和厚度修正
- Warm-start training：训练默认经过约 2 秒物理时间预热，再从稳定启动状态开始控制
- Batched simulation：底层环境原生支持 batch，并通过 SB3 接口映射为并行环境
- Physics-aware reward：奖励综合考虑三乘积、`Q_fusion`、密度/温度约束与 pellet 使用代价
- Safety termination：对非物理或数值异常状态进行提前终止
- Visualization support：提供 PyQt 桌面程序做并行仿真与结果可视化

## 仓库结构

- `rl_residual_lab/`
  residual RL 的主要入口，包括训练、评估与实验脚本

- `RL/`
  通用环境封装、reward、VecEnv 适配器

- `simulator/`
  物理模拟核心，包括 `torax` 侧输运求解与 pellet 沉积模型

- `Examples/`
  不经过 RL 的并行仿真示例，适合先做环境连通性与依赖验证

- `visualization/`
  PyQt 图形界面，用于交互式仿真和结果查看

- `config/ITER.py`
  默认 ITER-like 配置

- `requirement.txt`
  Python 依赖列表

## 环境准备

推荐使用项目脚本默认的 conda 环境：

```bash
conda activate Nuclear
python3 -m pip install -r requirement.txt
```

主要依赖包括：

- `torch`
- `jax`
- `jaxlib`
- `torax`
- `stable-baselines3`
- `PyQt5`
- `pyqtgraph`

首次在新机器上运行时，建议先从 `Examples/` 中的小规模 CPU 任务开始，确认依赖、JAX 后端与本地二进制扩展均可正常工作。

## Residual RL 机制

### 控制形式

环境对策略暴露的是二维 residual 动作：

- `delta_velocity`
- `delta_thickness`

环境内部会把这两个量与固定基线参数组合成完整控制动作：

- `trigger`
- `velocity`
- `thickness`

默认基线参数为：

- 注入周期 `inject_every = 100`
- 单次注入持续 `inject_duration = 1`
- 基线速度 `300 m/s`
- 基线厚度 `2.0 mm`

当控制步不处于注入窗口内时，环境会自动关闭注入，residual 动作不会被应用到物理系统。

### Warm-start 训练

训练环境默认先执行 `warmup_steps = 2000` 步。结合 [config/ITER.py](/Users/sheldonwang/residual-RL-pellet-injection/config/ITER.py) 中的 `fixed_dt = 0.001`，这对应约 2 秒物理时间。

预热结束后，环境会缓存等离子体状态。后续训练 episode 在 reset 时直接回到该 warm-start 状态，而不是反复从冷启动开始。这能显著减少启动瞬态对 PPO 训练的干扰。

### 观测

基础观测由 [RL/env.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/env.py) 提供，会把输运状态树与诊断输出展开为 `(B, D)` 形式的数值特征。

启用 `--append-schedule-features` 后，环境还会拼接 4 个节拍特征：

- `sin(phase)`
- `cos(phase)`
- `inject_now`
- `control_active`

### Reward 与安全终止

Reward 实现在 [RL/reward.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/reward.py)，主要关注：

- 聚变三乘积 `n T tau_E`
- `Q_fusion`
- Greenwald fraction 约束
- 电子/离子体平均温度范围
- pellet 使用量和注入速度带来的代价

环境会对明显异常的状态进行安全终止，例如：

- 非有限数值
- 非正密度或温度
- Greenwald fraction 过高
- 核心温度异常偏低或偏高

## 快速开始

### 原始模拟器示例

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

示例脚本会将摘要与配置保存到 `example_logs/`。

### PPO 训练

直接运行训练入口：

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

使用默认训练脚本：

```bash
bash rl_residual_lab/run_train_residual.sh
```

CPU 调试版本：

```bash
bash rl_residual_lab/run_train_residual_cpu.sh
```

训练产物默认保存在：

- `rl_logs/<run_name>/`
- `rl_models/<run_name>/`

常见输出包括：

- `config.json`
- `monitor.csv`
- `tensorboard/`
- `train.stdout.log`
- `checkpoint_*.zip`
- `final_model.zip`

### 模型评估

评估训练好的 residual 策略：

```bash
python3 rl_residual_lab/eval_residual.py \
  --model-path rl_models/<run_name>/final_model.zip \
  --device cpu \
  --batch-size 1 \
  --eval-steps 10000 \
  --deterministic
```

也可以使用封装脚本：

```bash
bash rl_residual_lab/run_eval_residual_model.sh rl_models/<run_name>/final_model.zip cpu
```

若不提供 `--model-path`，评估脚本会使用零残差策略，作为固定基线控制器的对照。

评估结果默认写入 `eval_logs/`，包括：

- `<run_name>_config.json`
- `<run_name>_summary.json`

## 可视化界面

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

## 默认配置

默认实验场景位于 [config/ITER.py](/Users/sheldonwang/residual-RL-pellet-injection/config/ITER.py)，当前特征包括：

- 时间步长 `1 ms`
- 总时长 `t_final = 100`
- ITER-like 几何参数
- 已集成温度、密度、热输运与源项的实验配置

你也可以替换为其他配置文件，并在 `Examples/`、`rl_residual_lab/` 与 `visualization/` 三条工作流中复用。

## 主要入口文件

- [rl_residual_lab/train_residual.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/train_residual.py)
- [rl_residual_lab/eval_residual.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/eval_residual.py)
- [rl_residual_lab/residual_env.py](/Users/sheldonwang/residual-RL-pellet-injection/rl_residual_lab/residual_env.py)
- [RL/env.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/env.py)
- [RL/reward.py](/Users/sheldonwang/residual-RL-pellet-injection/RL/reward.py)
- [visualization/app.py](/Users/sheldonwang/residual-RL-pellet-injection/visualization/app.py)

## 致谢

底层输运模拟工作流建立在 [Google DeepMind Torax](https://github.com/google-deepmind/torax) 相关生态之上，pellet injection 与 residual RL 控制部分为本项目的研究实现。
