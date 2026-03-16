"""Reward interface - ROBUST PHYSICS VERSION with Assertions"""
from typing import Any, Dict
import numpy as np

def compute_reward(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:

    if obs is None or next_obs is None:
        return 0.0
    
    batch_size = obs.shape[0] if obs.ndim >= 2 else 1
    if batch_size == 1:
        obs = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        action = action.reshape(1, -1)
    
    # 2. 提取物理诊断数据 (确保从 info 拿到的量级是合理的，如 MW 或 1e20)
    # 默认值设为安全量级，防止除零
    ne = np.nan_to_num(np.asarray(info.get("ne_volume_avg", 1.0e19)), nan=1.0e19)
    te = np.nan_to_num(np.asarray(info.get("te_core", 1.0)), nan=1.0)
    p_fusion = np.nan_to_num(np.asarray(info.get("P_fusion", 0.0)), nan=0.0)
    p_heating = np.nan_to_num(np.asarray(info.get("P_heating", 1.0)), nan=1.0)
    
    # 3. 核心指标归一化处理
    ne_norm = np.clip(ne / 1.0e20, 0.01, 2.0)
    te_norm = np.clip(te / 15.0, 0.01, 3.0)
    q_val = p_fusion / np.maximum(p_heating, 0.1)

    # --- 分项奖励计算 ---
    
    # A. 密度平滑奖励 [0, 0.2]
    # 使用高斯核：在 1.0e20 附近获得最大值
    density_reward = 0.2 * np.exp(-0.5 * ((ne_norm - 1.0) / 0.3)**2)
    
    # B. Q 值性能奖励 [0, 0.5]
    # 使用对数映射：Q=10 时约得到 0.35
    q_reward = 0.5 * (np.log1p(q_val) / np.log1p(20.0))
    
    # C. 动作惩罚 [-0.05, 0] !!! 关键修正点 !!!
    # 必须将 action 映射回 [-1, 1] 区间再计算平方，否则 1000^2 会毁掉所有信号
    # 假设 action[:, 0] 是 trigger [-1, 1]
    # 假设 action[:, 1] 是速度 [100, 1000]，我们对其手动归一化
    v_normalized = (action[:, 1] - 550.0) / 450.0 
    t_normalized = action[:, 0]
    
    # 惩罚系数设为 0.02，保证它只是微调
    action_penalty = -0.02 * (np.square(t_normalized) + np.square(v_normalized))

    # D. 基础生存奖励
    base_reward = 0.01

    # 4. 汇总
    reward = density_reward + q_reward + action_penalty + base_reward
    
    # --- 数值安全性检查 (Assertions) ---
    # 检查是否有 NaN
    if np.isnan(reward).any():
        print(f"DEBUG - ne: {ne}, te: {te}, P_f: {p_fusion}, P_h: {p_heating}")
        raise AssertionError("Reward 包含 NaN 值！请检查物理输入。")
    
    # 检查 Action Penalty 是否溢出 (之前发生的 -166 错误)
    if (action_penalty < -1.0).any():
        print(f"DEBUG - Raw Action: {action[0]}, Normalized V: {v_normalized[0]}")
        raise AssertionError(f"Action Penalty 异常过大 ({np.min(action_penalty)})，请检查归一化逻辑。")

    # 5. 最终裁剪 [-1, 1]
    reward = np.clip(reward, -1.0, 1.0)

    # 6. 随机采样打印 (1/1000 概率)
    if np.random.rand() < 0.01:
        q_print = np.atleast_1d(q_val)[0]
        ne_print = np.atleast_1d(ne_norm)[0]
        d_rew_print = np.atleast_1d(density_reward)[0]
        q_rew_print = np.atleast_1d(q_reward)[0]
        a_pen_print = np.atleast_1d(action_penalty)[0]
        r_total_print = np.atleast_1d(reward)[0]

        print(f"\n>>> [Reward Check] Q={q_print:.2f}, ne_norm={ne_print:.2f}")
        print(f">>> Components: Density={d_rew_print:.3f}, Q_Gain={q_rew_print:.3f}, Act_Pen={a_pen_print:.4f}")
        print(f">>> FINAL REWARD: {r_total_print:.4f}\n")

    return reward.astype(np.float32)