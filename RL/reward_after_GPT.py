"""Reward interface - TRANSPORT FOCUSED, CONTINUOUS + FAIL-FAST VERSION"""
from typing import Any, Dict, Sequence
import numpy as np


EPS = 1e-12


class SafetyViolationError(RuntimeError):
    """Catastrophic physical or numerical state; training should stop immediately."""
    pass


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # 这里只限制 exp 的输入，防止数值溢出；不是对物理量做裁剪
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _as_batch_array(value: Any, batch_size: int, name: str, allow_none: bool = False):
    if value is None:
        if allow_none:
            return None
        raise KeyError(f"Reward 缺少必要字段: {name}")

    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return np.full((batch_size,), float(arr), dtype=np.float64)

    arr = arr.reshape(-1)
    if arr.size == 1 and batch_size > 1:
        return np.full((batch_size,), float(arr[0]), dtype=np.float64)

    if arr.size != batch_size:
        raise ValueError(f"{name} shape={arr.shape} 与 batch_size={batch_size} 不一致")

    return arr


def _get_first(
    info: Dict[str, Any],
    keys: Sequence[str],
    batch_size: int,
    *,
    required: bool = True,
    default: Any = None,
):
    for k in keys:
        if k in info and info[k] is not None:
            return _as_batch_array(info[k], batch_size, name=k)

    if default is not None:
        return _as_batch_array(default, batch_size, name="default")

    if required:
        raise KeyError(f"Reward 缺少字段，期待其中之一: {keys}")

    return None


def _assert_all_finite(name: str, x: np.ndarray):
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(f"{name} 含 NaN/Inf: {x}")


def _band_reward(
    x: np.ndarray,
    *,
    target: float,
    sigma: float,
    low: float,
    high: float,
    low_scale: float,
    high_scale: float,
    w_pos: float,
    w_low: float,
    w_high: float,
):
    """
    连续型区间奖励：
    - target 附近给正奖励
    - 低于 low 连续惩罚
    - 高于 high 连续惩罚
    """
    pos = w_pos * np.exp(-0.5 * ((x - target) / sigma) ** 2)
    low_pen = -w_low * _softplus((low - x) / low_scale)
    high_pen = -w_high * _softplus((x - high) / high_scale)
    total = pos + low_pen + high_pen
    return total, pos, low_pen, high_pen


def compute_reward(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    info: Dict[str, Any],
):
    if obs is None or next_obs is None:
        return np.float32(0.0)

    obs = np.asarray(obs)
    next_obs = np.asarray(next_obs)
    action = np.asarray(action, dtype=np.float64)
    info = info or {}

    single_input = obs.ndim == 1
    batch_size = 1 if single_input else obs.shape[0]

    if single_input:
        obs = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        action = action.reshape(1, -1)
    elif action.ndim == 1:
        action = action.reshape(batch_size, -1)

    # ------------------------------------------------------------------
    # 1) 读取物理量：不再用 nan_to_num 掩盖异常
    # ------------------------------------------------------------------
    ne = _get_first(
        info,
        ["ne_volume_avg", "ne_avg", "electron_density_volume_avg"],
        batch_size,
    )

    te = _get_first(
        info,
        ["te_core", "Te_core", "electron_temp_core"],
        batch_size,
    )

    ti = _get_first(
        info,
        ["ti_core", "Ti_core", "ion_temp_core"],
        batch_size,
        required=False,
        default=te,  # 如果没有 Ti，先退化成 Ti=Te
    )
    
    p_fusion = _get_first(
        info,
        ["P_fusion", "fusion_power"],
        batch_size,
    )

    # Q 建议优先用外部加热功率，而不是总加热功率
    p_heating = _get_first(
        info,
        ["P_external", "P_auxiliary", "P_heating", "external_heating_power"],
        batch_size,
    )

    # 三重积需要 tau_E；如果没有，尝试用 W_plasma / P_loss 估算
    tau_E = _get_first(
        info,
        ["tau_E", "tau_energy", "energy_confinement_time"],
        batch_size,
        required=False,
    )
    if tau_E is None:
        W_plasma = _get_first(
            info,
            ["W_plasma", "stored_energy"],
            batch_size,
            required=False,
        )
        P_loss = _get_first(
            info,
            ["P_loss", "power_loss"],
            batch_size,
            required=False,
        )

        if W_plasma is None or P_loss is None:
            raise KeyError(
                "三重积 reward 需要 tau_E，或同时提供 (W_plasma, P_loss) 用于估算 tau_E。"
            )
        if np.any(P_loss <= 0.0):
            raise SafetyViolationError(f"P_loss <= 0，无法计算 tau_E: {P_loss}")

        tau_E = W_plasma / P_loss

    # Greenwald 系数优先直接读；否则尝试自己算
    f_gw = _get_first(
        info,
        ["greenwald_fraction", "f_greenwald", "fgw", "n_over_nGW"],
        batch_size,
        required=False,
    )
    if f_gw is None:
        n_line = _get_first(
            info,
            ["ne_line_avg", "n_line_avg"],
            batch_size,
            required=False,
        )
        n_gw = _get_first(
            info,
            ["n_greenwald", "greenwald_limit_density", "n_GW"],
            batch_size,
            required=False,
        )

        if n_line is None or n_gw is None:
            raise KeyError(
                "Greenwald reward 需要 greenwald_fraction，或同时提供 (ne_line_avg, n_greenwald)."
            )
        if np.any(n_gw <= 0.0):
            raise SafetyViolationError(f"n_GW <= 0，无法计算 Greenwald 分数: {n_gw}")

        f_gw = n_line / n_gw

    # 可选：如果有前一步 Q，可增加稳定性项
    q_prev = _get_first(
        info,
        ["Q_prev", "prev_Q"],
        batch_size,
        required=False,
    )

    # ------------------------------------------------------------------
    # 2) Fail-fast 数值检查
    # ------------------------------------------------------------------
    for name, arr in {
        "ne": ne,
        "te": te,
        "ti": ti,
        "P_fusion": p_fusion,
        "P_heating": p_heating,
        "tau_E": tau_E,
        "f_gw": f_gw,
    }.items():
        _assert_all_finite(name, arr)

    if np.any(ne <= 0.0):
        raise SafetyViolationError(f"检测到 ne <= 0，属于非物理解: {ne}")

    if np.any(te <= 0.0) or np.any(ti <= 0.0):
        raise SafetyViolationError(f"检测到 Te/Ti <= 0，属于非物理解: Te={te}, Ti={ti}")

    if np.any(p_fusion < 0.0):
        raise SafetyViolationError(f"检测到 P_fusion < 0，属于非物理解: {p_fusion}")

    if np.any(p_heating <= 0.0):
        raise SafetyViolationError(f"检测到 P_heating <= 0，无法定义 Q: {p_heating}")

    if np.any(tau_E <= 0.0):
        raise SafetyViolationError(f"检测到 tau_E <= 0，属于非物理解: {tau_E}")

    if np.any(f_gw < 0.0):
        raise SafetyViolationError(f"检测到 f_gw < 0，属于非物理解: {f_gw}")

    # 如果模拟器已经显式告诉你安全终止，直接停
    for flag_key in [
        "disrupted",
        "plasma_disruption",
        "major_safety_violation",
        "terminated_on_safety",
    ]:
        if flag_key in info and np.any(np.asarray(info[flag_key]).astype(bool)):
            raise SafetyViolationError(
                f"检测到安全终止标志 {flag_key}={info[flag_key]}，终止训练。"
            )

    # ------------------------------------------------------------------
    # 3) 灾难性状态：直接 raise，不再靠 reward 裁剪掩盖
    #    这些阈值需要你按具体机型再标定
    # ------------------------------------------------------------------
    if np.any(f_gw >= 1.15):
        raise SafetyViolationError(f"Greenwald 分数过高，存在破裂风险: f_gw={f_gw}")

    if np.any(te >= 35.0) or np.any(ti >= 35.0):
        raise SafetyViolationError(f"核心温度过高，存在严重热安全问题: Te={te}, Ti={ti}")

    # ------------------------------------------------------------------
    # 4) 派生核心物理指标
    # ------------------------------------------------------------------
    q_val = p_fusion / p_heating
    t_avg = 0.5 * (te + ti)
    triple_product = ne * t_avg * tau_E   # 假设 T 单位是 keV

    _assert_all_finite("Q", q_val)
    _assert_all_finite("triple_product", triple_product)

    # ------------------------------------------------------------------
    # 5) 连续型 reward 分项
    # ------------------------------------------------------------------

    # (A) 聚变三重积：正向奖励，单调递增，log 压缩
    triple_reward = 0.35 * np.log1p(triple_product / 3.0e21)

    # (B) Q：既鼓励更大 Q，也显式鼓励 Q > 5
    q_gain_reward = 0.15 * np.log1p(q_val) / np.log1p(10.0)
    q_margin_reward = 0.25 * np.tanh((q_val - 5.0) / 1.5)
    q_reward = q_gain_reward + q_margin_reward

    # 可选：鼓励“稳定在 5 以上”，需要上一时刻 Q
    if q_prev is not None:
        _assert_all_finite("Q_prev", q_prev)
        q_stability_penalty = -0.05 * np.sqrt((q_val - q_prev) ** 2 + 1e-8)
    else:
        q_stability_penalty = np.zeros_like(q_val)

    # (C) 密度：防止熄灭，也防止过高
    density_reward, density_pos, density_low_pen, density_high_pen = _band_reward(
        ne,
        target=1.0e20,
        sigma=0.25e20,
        low=0.45e20,
        high=1.20e20,
        low_scale=0.08e20,
        high_scale=0.10e20,
        w_pos=0.10,
        w_low=0.18,
        w_high=0.12,
    )

    # (D) 电子温度
    te_reward, te_pos, te_low_pen, te_high_pen = _band_reward(
        te,
        target=12.0,
        sigma=4.0,
        low=4.0,
        high=22.0,
        low_scale=1.2,
        high_scale=2.0,
        w_pos=0.08,
        w_low=0.16,
        w_high=0.14,
    )

    # (E) 离子温度
    ti_reward, ti_pos, ti_low_pen, ti_high_pen = _band_reward(
        ti,
        target=14.0,
        sigma=4.5,
        low=4.0,
        high=25.0,
        low_scale=1.2,
        high_scale=2.0,
        w_pos=0.08,
        w_low=0.16,
        w_high=0.14,
    )

    # (F) Greenwald：安全区小正奖励，逼近极限时连续重罚
    gw_safe_bonus = 0.10 * np.exp(-0.5 * ((f_gw - 0.72) / 0.12) ** 2)
    gw_risk_penalty = -0.30 * _softplus((f_gw - 0.90) / 0.05)
    gw_reward = gw_safe_bonus + gw_risk_penalty

    # (G) Pellet 注入代价：
    #     action[:, 0] -> trigger
    #     action[:, 1] -> speed (m/s)
    if action.shape[1] < 2:
        raise ValueError(
            f"当前 reward 假设 action 至少有 2 维 [trigger, speed]，实际 shape={action.shape}"
        )

    trigger_raw = action[:, 0]
    pellet_speed = action[:, 1]

    # 兼容两种 trigger 编码：
    # - [0, 1]：用 0.5 作为软阈值
    # - [-1, 1]：用 0 作为软阈值
    if np.min(trigger_raw) >= -0.05 and np.max(trigger_raw) <= 1.05:
        inject_intensity = _sigmoid(10.0 * (trigger_raw - 0.5))
    else:
        inject_intensity = _sigmoid(10.0 * trigger_raw)

    if np.any(pellet_speed < 0.0):
        raise SafetyViolationError(f"检测到负 pellet speed，属于非物理解: {pellet_speed}")

    if np.any(pellet_speed > 2000.0):
        raise SafetyViolationError(
            f"pellet speed 异常过高，请检查动作映射或归一化: {pellet_speed}"
        )

    v_min, v_ref = 100.0, 1000.0
    v_norm = np.maximum((pellet_speed - v_min) / (v_ref - v_min), 0.0)

    # 一次注入本身就有成本；速度越高，额外成本越大
    pellet_penalty = -(0.03 * inject_intensity + 0.07 * inject_intensity * (v_norm ** 2))

    # ------------------------------------------------------------------
    # 6) 汇总：不做最终硬裁剪
    # ------------------------------------------------------------------
    reward = (
        triple_reward
        + q_reward
        + q_stability_penalty
        + density_reward
        + te_reward
        + ti_reward
        + gw_reward
        + pellet_penalty
    )

    _assert_all_finite("reward_before_return", reward)

    # 随机打印检查
    if np.random.rand() < 0.001:
        i = 0
        print(
            "\n>>> [Reward Check]"
            f"\n>>> Q={q_val[i]:.3f}, triple={triple_product[i]:.3e}, f_gw={f_gw[i]:.3f}"
            f"\n>>> ne={ne[i]:.3e}, Te={te[i]:.3f} keV, Ti={ti[i]:.3f} keV, tau_E={tau_E[i]:.3f} s"
            f"\n>>> triple_reward={triple_reward[i]:+.4f}"
            f"\n>>> q_reward={q_reward[i]:+.4f}, q_stab={q_stability_penalty[i]:+.4f}"
            f"\n>>> density={density_reward[i]:+.4f}, te={te_reward[i]:+.4f}, ti={ti_reward[i]:+.4f}"
            f"\n>>> gw={gw_reward[i]:+.4f}, pellet={pellet_penalty[i]:+.4f}"
            f"\n>>> TOTAL={reward[i]:+.4f}\n"
        )

    reward = reward.astype(np.float32)
    return reward[0] if single_input else reward