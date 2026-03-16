import dataclasses
import torch
import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util
from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack
from torax._src.config import config_loader, build_runtime_params
from simulator.run_loop_sim import prepare_simulation

KEV_TO_JOULES = 1.60217663e-16

# --- Helper: Data Conversion ---
# simulator/torax_simulator.py

def _torch_to_jax_sharded(tensor, num_devices, batch_per_device, devices):
    """
    [Robust Version] Torch (B, ...) -> CPU -> JAX Sharded (4, B/4, ...)
    通过 CPU 中转彻底规避 CUDA Context Device Index 冲突问题。
    """
    # 1. 【安全通道】先转到 CPU Numpy
    # 这步操作彻底切断了 GPU 3 和 GPU 0 之间的直接上下文纠葛
    # 对于 Batch=4096 的标量数据，这只需要几微秒
    if torch.is_tensor(tensor):
        # detach() 确保不带梯度，cpu() 搬运，numpy() 转格式
        host_array = tensor.detach().cpu().numpy()
    else:
        host_array = tensor

    # 2. 转为 JAX Array (Host端)
    # 这一步是在 CPU 内存中进行的，非常安全
    jax_host_array = jax.numpy.array(host_array)
    
    # 3. Reshape 为 (4, B/4, ...)
    new_shape = (num_devices, batch_per_device) + jax_host_array.shape[1:]
    reshaped_view = jax_host_array.reshape(new_shape)
    
    # 4. 物理分发 (Scatter)
    # jax.device_put_sharded 会自动处理从 CPU 到 4 张 GPU 的并行搬运
    return jax.device_put_sharded(list(reshaped_view), devices)

def _torch_to_jax_single(tensor, device):
    """Torch (...,) -> single-device JAX array without sharding overhead."""
    if torch.is_tensor(tensor):
        tensor = tensor.detach()
        try:
            if tensor.is_cuda:
                return jax_dlpack.from_dlpack(
                    torch_dlpack.to_dlpack(tensor.contiguous()),
                    device=device,
                    copy=False,
                )
        except Exception:
            # DLPack fast-path is best effort. Fallback keeps behavior robust.
            pass
        host_array = tensor.cpu().numpy()
    else:
        host_array = tensor

    return jax.device_put(jnp.asarray(host_array), device)

def _jax_sharded_to_torch(jax_array):
    """
    JAX Sharded (4, B/4, ...) -> Torch (B, ...)
    """
    # 1. 收集到主设备 (通常是 GPU 0)
    # device_put 会处理数据的 gather
    jax_on_host = jax.device_put(jax_array, jax.devices()[0])
    
    # 2. Flatten 前两维 (4, 32, ...) -> (128, ...)
    if jax_on_host.ndim > 1:
        new_shape = (-1,) + jax_on_host.shape[2:]
        jax_on_host = jax_on_host.reshape(new_shape)
        
    # 3. 转为 Torch
    return torch_dlpack.from_dlpack(jax_on_host)

def _jax_array_to_torch(jax_array, device):
    """Single-device JAX array -> Torch tensor."""
    jax_on_device = jax.device_put(jax_array, device)
    return torch_dlpack.from_dlpack(jax_on_device)


# --- JAX Wrappers (Internal) ---
@jax.tree_util.register_pytree_node_class
class JAXParamsWrapper:
    def __init__(self, params):
        self.params = params
    def __call__(self, t=None):
        return self.params
    def __getattr__(self, name):
        return getattr(self.params, name)
    def tree_flatten(self):
        return (self.params,), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclasses.dataclass(frozen=True)
class BatchedInjectionParams:
    triggered: jnp.ndarray 
    pellet_deposition_location: jnp.ndarray
    pellet_width: jnp.ndarray
    S_total: jnp.ndarray

class TransportSimulator:
    def __init__(self, base_config_path: str, total_batch_size: int = 128):
        all_devices = jax.local_devices()
        self.num_available_devices = len(all_devices)

        # Multi-device sharding only pays off when the batch can be evenly split.
        # Otherwise fallback to a single-device fast path instead of failing.
        self.use_sharded_path = (
            self.num_available_devices > 1
            and total_batch_size >= self.num_available_devices
            and total_batch_size % self.num_available_devices == 0
        )

        if self.use_sharded_path:
            self.devices = all_devices
            self.num_devices = len(self.devices)
        else:
            self.devices = [all_devices[0]]
            self.num_devices = 1

        self.total_batch_size = total_batch_size
        self.batch_per_device = total_batch_size // self.num_devices if self.use_sharded_path else total_batch_size
        path_label = "sharded" if self.use_sharded_path else "single-device"
        print(
            f"[Torax] Initialized on {self.num_devices} device(s) "
            f"({path_label}; {self.num_available_devices} available). "
            f"Total Batch: {total_batch_size}"
        )

        self.base_config = config_loader.build_torax_config_from_file(base_config_path)
        initial_state, initial_outputs, step_fn = prepare_simulation(self.base_config)
        self._initial_state = initial_state
        self._initial_outputs = initial_outputs
        
        param_provider = build_runtime_params.RuntimeParamsProvider.from_config(self.base_config)
        self.default_params = param_provider(0.0)

        def init_batch(leaf):
            return jnp.stack([leaf] * total_batch_size)

        def init_shard(leaf):
            full_batch = init_batch(leaf)
            return full_batch.reshape((self.num_devices, self.batch_per_device) + leaf.shape)

        self._init_state = init_shard if self.use_sharded_path else init_batch
        self.current_states = jax.tree_util.tree_map(self._init_state, initial_state)
        self.last_outputs = jax.tree_util.tree_map(self._init_state, initial_outputs)

        # --- JAX Kernel (Pmap) ---
        def single_env_step(st, out, trig, l, w, s, bp):
            actual_S = jnp.where(trig, s, 0.0)
            old_pellet = bp.sources['pellet']
            new_pellet = dataclasses.replace(old_pellet, pellet_deposition_location=l, pellet_width=w, S_total=actual_S)
            new_sources = bp.sources.copy()
            new_sources['pellet'] = new_pellet
            current_step_params = dataclasses.replace(bp, sources=new_sources)
            provider = JAXParamsWrapper(current_step_params)
            return step_fn(st, out, runtime_params_overrides=provider)

        def batch_step_fn(state, outputs, triggered, loc, width, S_val, base_params):
            return jax.vmap(single_env_step, in_axes=(0, 0, 0, 0, 0, 0, None))(
                state, outputs, triggered, loc, width, S_val, base_params
            )

        if self.use_sharded_path:
            # 这里的 None 表示 base_params 广播，其他 0 表示分片
            self._parallel_step = jax.pmap(
                batch_step_fn,
                axis_name='devices',
                in_axes=(0, 0, 0, 0, 0, 0, None)
            )
            self._single_step = None
        else:
            self._single_step = jax.jit(lambda state, outputs, triggered, loc, width, S_val: batch_step_fn(
                state, outputs, triggered, loc, width, S_val, self.default_params
            ))
            self._parallel_step = None
        self.step_count = 0

    def reset(self):
        """Reset simulator state without rebuilding the full JAX graph."""
        self.current_states = jax.tree_util.tree_map(self._init_state, self._initial_state)
        self.last_outputs = jax.tree_util.tree_map(self._init_state, self._initial_outputs)
        self.step_count = 0
        return self

    def step(self, triggers: torch.Tensor, locs: torch.Tensor, widths: torch.Tensor, rates: torch.Tensor):
        """
        [Standard Torch Interface]
        输入: (Batch,) 形状的 Torch Tensor
        输出: state, output (这里为了简单，暂时返回内部状态对象，或者你可以选择返回特定的 Tensor)
        """
        if self.use_sharded_path:
            j_trig = _torch_to_jax_sharded(triggers, self.num_devices, self.batch_per_device, self.devices)
            j_loc = _torch_to_jax_sharded(locs, self.num_devices, self.batch_per_device, self.devices)
            j_width = _torch_to_jax_sharded(widths, self.num_devices, self.batch_per_device, self.devices)
            j_rate = _torch_to_jax_sharded(rates, self.num_devices, self.batch_per_device, self.devices)

            next_states, next_outputs = self._parallel_step(
                self.current_states,
                self.last_outputs,
                j_trig, j_loc, j_width, j_rate,
                self.default_params
            )
        else:
            device = self.devices[0]
            j_trig = _torch_to_jax_single(triggers, device)
            j_loc = _torch_to_jax_single(locs, device)
            j_width = _torch_to_jax_single(widths, device)
            j_rate = _torch_to_jax_single(rates, device)

            next_states, next_outputs = self._single_step(
                self.current_states,
                self.last_outputs,
                j_trig,
                j_loc,
                j_width,
                j_rate,
            )

        self.current_states = next_states
        self.last_outputs = next_outputs
        self.step_count += 1
        
        # 返回 self 以支持链式调用，或者返回 None，数据存在内部
        return self

    def get_plasma_tensor(self):
        """
        [Standard Torch Interface]
        返回: (Te, ne, Pe, Ti, ni, Pi, D_tuple, T_tuple) 全部为 Torch Tensor
        形状: (Batch, Grid_Size)
        """
        # 1. 提取 JAX 数据 (Sharded)
        s = self.current_states
        T_e = s.core_profiles.T_e.value
        n_e = s.core_profiles.n_e.value
        T_i = s.core_profiles.T_i.value
        n_i = s.core_profiles.n_i.value
        
        # 2. 计算 (JAX 内部计算)
        P_e = n_e * T_e * KEV_TO_JOULES
        P_i = n_i * T_i * KEV_TO_JOULES
        
        to_torch = _jax_sharded_to_torch if self.use_sharded_path else (lambda x: _jax_array_to_torch(x, self.devices[0]))

        # 3. 转换回 Torch
        t_Te = to_torch(T_e) * 1000.0 # keV -> eV
        t_ne = to_torch(n_e)
        t_Pe = to_torch(P_e)
        t_Ti = to_torch(T_i) * 1000.0 # keV -> eV
        t_ni = to_torch(n_i)
        t_Pi = to_torch(P_i)

        # Species (假设 50-50)
        t_nD = t_ni * 0.5
        t_nT = t_ni * 0.5
        t_PD = t_nD * t_Ti * KEV_TO_JOULES
        t_PT = t_nT * t_Ti * KEV_TO_JOULES # Ti is in eV here, formula needs check but keeping structure
        
        return t_Te, t_ne, t_Pe, t_Ti, t_ni, t_Pi, (t_nD, t_Ti, t_PD), (t_nT, t_Ti, t_PT)

    def get_diagnostics(self):
        """返回 reward 所需的标量诊断信息。"""
        last_outs = self.last_outputs
        core_profiles = self.current_states.core_profiles
        to_torch = _jax_sharded_to_torch if self.use_sharded_path else (lambda x: _jax_array_to_torch(x, self.devices[0]))

        return {
            "fgw_n_e_volume_avg": to_torch(last_outs.fgw_n_e_volume_avg).squeeze(),
            "P_fusion": to_torch(last_outs.P_fusion).squeeze(),
            "tau_E": to_torch(last_outs.tau_E).squeeze(),
            "Q_fusion": to_torch(last_outs.Q_fusion).squeeze(),
            "P_external_total": to_torch(last_outs.P_external_total).squeeze(),
            "n_e_volume_avg": to_torch(last_outs.n_e_volume_avg).squeeze(),
            "T_e_volume_avg": to_torch(last_outs.T_e_volume_avg).squeeze(),
            "T_i_volume_avg": to_torch(last_outs.T_i_volume_avg).squeeze(),
            "S_pellet": to_torch(last_outs.S_pellet).squeeze(),
            "n_e_core": to_torch(core_profiles.n_e.value[..., 0]).squeeze(),
            "T_e_core": to_torch(core_profiles.T_e.value[..., 0]).squeeze(),
            "T_i_core": to_torch(core_profiles.T_i.value[..., 0]).squeeze(),
        }
