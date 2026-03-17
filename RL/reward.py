from typing import Any, Dict
import numpy as np
 
class SafetyViolationError(RuntimeError):
    """严重物理或数值异常，直接终止训练。"""


UNSAFE_TERMINATION_PENALTY = -10.0

# ── Triple product ──
TRIPLE_PRODUCT_REF = 3.0e21

# ── Greenwald fraction (density safety) ──
FGW_TARGET = 0.70
FGW_SIGMA = 0.10
FGW_LOW = 0.35
FGW_HIGH = 0.88
FGW_LOW_SCALE = 0.05
FGW_HIGH_SCALE = 0.04
FGW_ABORT = 1.00

# ── Electron temperature (volume-averaged) ──
TE_VOL_TARGET = 10.0
TE_VOL_SIGMA = 3.0
TE_VOL_LOW = 3.0
TE_VOL_HIGH = 18.0
TE_VOL_LOW_SCALE = 1.0
TE_VOL_HIGH_SCALE = 1.5

# ── Ion temperature (volume-averaged) ──
TI_VOL_TARGET = 12.0
TI_VOL_SIGMA = 3.5
TI_VOL_LOW = 3.0
TI_VOL_HIGH = 22.0
TI_VOL_LOW_SCALE = 1.0
TI_VOL_HIGH_SCALE = 1.5

TE_CORE_ABORT_LOW = 0.5
TE_CORE_ABORT_HIGH = 30.0
TI_CORE_ABORT_LOW = 0.5
TI_CORE_ABORT_HIGH = 35.0

REWARD_INFO_KEYS = (
    "fgw_n_e_volume_avg",
    "P_fusion",
    "tau_E",
    "Q_fusion",
    "P_external_total",
    "n_e_volume_avg",
    "T_e_volume_avg",
    "T_i_volume_avg",
    "S_pellet",
    "n_e_core",
    "T_e_core",
    "T_i_core",
)


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _to_batch_array(value: Any, batch_size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return np.full((batch_size,), float(arr), dtype=np.float64)

    arr = arr.reshape(-1)

    if arr.size == 1 and batch_size > 1:
        return np.full((batch_size,), float(arr[0]), dtype=np.float64)

    if arr.size != batch_size:
        raise ValueError(f"{name} shape={arr.shape} 与 batch_size={batch_size} 不一致")

    return arr


def _read_info(info: Dict[str, Any], key: str, batch_size: int) -> np.ndarray:
    if key not in info:
        raise KeyError(f"Reward 缺少字段: {key}")
    return _to_batch_array(info[key], batch_size, key)


def _extract_reward_inputs(info: Dict[str, Any], batch_size: int) -> Dict[str, np.ndarray]:
    values = {name: _read_info(info, name, batch_size) for name in REWARD_INFO_KEYS}
    for name, value in values.items():
        _require_finite(name, value)
    return values


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        print('Warning: 数值异常检测到 - ', name, value)
        # raise FloatingPointError(f"{name} 含 NaN 或 Inf: {value}")


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
) -> np.ndarray:
    positive = w_pos * np.exp(-0.5 * ((x - target) / sigma) ** 2)
    low_penalty = -w_low * _softplus((low - x) / low_scale)
    high_penalty = -w_high * _softplus((x - high) / high_scale)
    return positive + low_penalty + high_penalty


def evaluate_unsafe_conditions(
    info: Dict[str, Any],
    batch_size: int,
    values: Dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = values or _extract_reward_inputs(info, batch_size)
    unsafe_mask = np.zeros((batch_size,), dtype=bool)
    reasons = np.full((batch_size,), "", dtype=object)

    def _mark(mask: np.ndarray, reason: str) -> None:
        nonlocal unsafe_mask, reasons
        mask = np.asarray(mask, dtype=bool).reshape(batch_size)
        new_hits = mask & ~unsafe_mask
        reasons[new_hits] = reason
        unsafe_mask |= mask

    nonfinite_mask = np.zeros((batch_size,), dtype=bool)
    for value in values.values():
        nonfinite_mask |= ~np.isfinite(value)
    _mark(nonfinite_mask, "nonfinite_diagnostics")

    _mark((values["n_e_volume_avg"] <= 0.0) | (values["n_e_core"] <= 0.0), "nonpositive_density")
    _mark((values["T_e_volume_avg"] <= 0.0) | (values["T_i_volume_avg"] <= 0.0), "nonpositive_volume_temperature")
    _mark((values["T_e_core"] <= 0.0) | (values["T_i_core"] <= 0.0), "nonpositive_core_temperature")
    _mark(values["P_fusion"] < 0.0, "negative_fusion_power")
    _mark(values["P_external_total"] <= 0.0, "nonpositive_external_power")
    _mark(values["tau_E"] <= 0.0, "nonpositive_tau_E")
    _mark(values["Q_fusion"] < 0.0, "negative_q_fusion")
    _mark(values["S_pellet"] < 0.0, "negative_pellet_source")
    _mark(values["fgw_n_e_volume_avg"] < 0.0, "negative_greenwald_fraction")
    _mark(values["fgw_n_e_volume_avg"] >= FGW_ABORT, "greenwald_abort")
    _mark(
        (values["T_e_core"] <= TE_CORE_ABORT_LOW) | (values["T_i_core"] <= TI_CORE_ABORT_LOW),
        "core_temperature_too_low",
    )
    _mark(
        (values["T_e_core"] >= TE_CORE_ABORT_HIGH) | (values["T_i_core"] >= TI_CORE_ABORT_HIGH),
        "core_temperature_too_high",
    )

    return unsafe_mask, reasons

def compute_reward(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
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

    values = _extract_reward_inputs(info, batch_size)
    unsafe_mask, _unsafe_reasons = evaluate_unsafe_conditions(info, batch_size, values=values)

    # ── clip to physical bounds ──
    fgw_n_e_volume_avg = np.clip(values["fgw_n_e_volume_avg"], 0.0, None)
    tau_E = np.clip(values["tau_E"], 1e-6, None)
    n_e_volume_avg = np.clip(values["n_e_volume_avg"], 1e-12, None)
    T_e_volume_avg = np.clip(values["T_e_volume_avg"], 1e-6, None)
    T_i_volume_avg = np.clip(values["T_i_volume_avg"], 1e-6, None)

    # ════════════════════════════════════════════════════════════
    #  1. Triple product  (weight ≈ 0.40)   — 核心聚变性能指标
    # ════════════════════════════════════════════════════════════
    T_volume_avg = 0.5 * (T_e_volume_avg + T_i_volume_avg)
    triple_product = np.clip(n_e_volume_avg * T_volume_avg * tau_E, 0.0, None)
    _require_finite("triple_product", triple_product)
    triple_reward = 0.40 * np.log1p(triple_product / TRIPLE_PRODUCT_REF)

    # ════════════════════════════════════════════════════════════
    #  2. Density band — fGW centering  (weight ≈ 0.18)
    # ════════════════════════════════════════════════════════════
    density_reward = _band_reward(
        fgw_n_e_volume_avg,
        target=FGW_TARGET,
        sigma=FGW_SIGMA,
        low=FGW_LOW,
        high=FGW_HIGH,
        low_scale=FGW_LOW_SCALE,
        high_scale=FGW_HIGH_SCALE,
        w_pos=0.18,
        w_low=0.25,
        w_high=0.15,
    )

    # ════════════════════════════════════════════════════════════
    #  3. Electron temperature band  (weight ≈ 0.14)
    # ════════════════════════════════════════════════════════════
    te_reward = _band_reward(
        T_e_volume_avg,
        target=TE_VOL_TARGET,
        sigma=TE_VOL_SIGMA,
        low=TE_VOL_LOW,
        high=TE_VOL_HIGH,
        low_scale=TE_VOL_LOW_SCALE,
        high_scale=TE_VOL_HIGH_SCALE,
        w_pos=0.14,
        w_low=0.20,
        w_high=0.16,
    )

    # ════════════════════════════════════════════════════════════
    #  4. Ion temperature band  (weight ≈ 0.14)
    # ════════════════════════════════════════════════════════════
    ti_reward = _band_reward(
        T_i_volume_avg,
        target=TI_VOL_TARGET,
        sigma=TI_VOL_SIGMA,
        low=TI_VOL_LOW,
        high=TI_VOL_HIGH,
        low_scale=TI_VOL_LOW_SCALE,
        high_scale=TI_VOL_HIGH_SCALE,
        w_pos=0.14,
        w_low=0.20,
        w_high=0.16,
    )

    # ════════════════════════════════════════════════════════════
    #  5. Greenwald safety margin  (weight ≈ 0.14)
    #     Gaussian 正奖励 @ 0.72 + 强 softplus 惩罚 > 0.90
    # ════════════════════════════════════════════════════════════
    greenwald_reward = (
        0.14 * np.exp(-0.5 * ((fgw_n_e_volume_avg - 0.72) / 0.10) ** 2)
        - 0.40 * _softplus((fgw_n_e_volume_avg - 0.90) / 0.03)
    )

    # ── total ──
    reward = (
        triple_reward
        + density_reward
        + te_reward
        + ti_reward
        + greenwald_reward
    )

    _require_finite("reward", reward)
    reward = np.nan_to_num(
        reward,
        nan=UNSAFE_TERMINATION_PENALTY,
        posinf=UNSAFE_TERMINATION_PENALTY,
        neginf=UNSAFE_TERMINATION_PENALTY,
    )
    reward[unsafe_mask] = UNSAFE_TERMINATION_PENALTY

    reward = reward.astype(np.float32)
    return reward[0] if single_input else reward
