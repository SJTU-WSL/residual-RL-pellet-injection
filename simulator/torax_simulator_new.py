import os
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

# 修改后的 torax_simulator.py

def _torch_to_jax_sharded(tensor, num_devices, batch_per_device, devices):
    """
    [High Performance Version] Torch (GPU) -> JAX (GPU) -> JAX Sharded
    全程不回 CPU，利用 GPU 显存直接通信 (DLPack)。
    """
    # 1. 确保 Tensor 是连续的 (DLPack 要求)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # 2. 【关键】零拷贝转换：Torch GPU -> JAX GPU (通常是 GPU 0)
    # 这步操作是微秒级的，不涉及数据搬运，只传递指针
    jax_array = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor))
    
    # 3. Reshape 为 (4, B/4, ...)
    # 这一步发生在 JAX 的 GPU 显存上
    new_shape = (num_devices, batch_per_device) + jax_array.shape[1:]
    reshaped_view = jax_array.reshape(new_shape)
    
    # 4. 物理分发 (Shard)
    # 使用 jax.device_put 将数据从 GPU 0 快速广播/切分到 GPU 0-3
    # JAX 会利用 NVLink 或 PCIe P2P 直接传输，比回 CPU 快几十倍
    return jax.device_put(reshaped_view, devices)

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
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        if total_batch_size % self.num_devices != 0:
            raise ValueError(f"Batch {total_batch_size} not divisible by {self.num_devices}")
        
        self.total_batch_size = total_batch_size
        self.batch_per_device = total_batch_size // self.num_devices
        print(f"[Torax] Initialized on {self.num_devices} devices. Total Batch: {total_batch_size}")

        self.base_config = config_loader.build_torax_config_from_file(base_config_path)
        initial_state, initial_outputs, step_fn = prepare_simulation(self.base_config)
        
        param_provider = build_runtime_params.RuntimeParamsProvider.from_config(self.base_config)
        self.default_params = param_provider(0.0)

        # 初始化状态并分片
        def init_shard(leaf):
            full_batch = jnp.stack([leaf] * total_batch_size)
            return full_batch.reshape((self.num_devices, self.batch_per_device) + leaf.shape)

        self.current_states = jax.tree_util.tree_map(init_shard, initial_state)
        self.last_outputs = jax.tree_util.tree_map(init_shard, initial_outputs)

        # --- JAX Kernel (Pmap) ---
        def device_step_fn(state, outputs, triggered, loc, width, S_val, base_params):
            def single_env_step(st, out, trig, l, w, s, bp):
                actual_S = jnp.where(trig, s, 0.0)
                old_pellet = bp.sources['pellet']
                new_pellet = dataclasses.replace(old_pellet, pellet_deposition_location=l, pellet_width=w, S_total=actual_S)
                new_sources = bp.sources.copy()
                new_sources['pellet'] = new_pellet
                current_step_params = dataclasses.replace(bp, sources=new_sources)
                provider = JAXParamsWrapper(current_step_params)
                return step_fn(st, out, runtime_params_overrides=provider)

            return jax.vmap(single_env_step, in_axes=(0,0,0,0,0,0, None))(
                state, outputs, triggered, loc, width, S_val, base_params
            )

        # 这里的 None 表示 base_params 广播，其他 0 表示分片
        self._parallel_step = jax.pmap(
            device_step_fn, 
            axis_name='devices',
            in_axes=(0, 0, 0, 0, 0, 0, None) 
        )
        self.step_count = 0

    def step(self, triggers: torch.Tensor, locs: torch.Tensor, widths: torch.Tensor, rates: torch.Tensor):
        """
        [Standard Torch Interface]
        输入: (Batch,) 形状的 Torch Tensor
        输出: state, output (这里为了简单，暂时返回内部状态对象，或者你可以选择返回特定的 Tensor)
        """
        # 1. Torch -> Sharded JAX
        j_trig = _torch_to_jax_sharded(triggers, self.num_devices, self.batch_per_device, self.devices)
        j_loc = _torch_to_jax_sharded(locs, self.num_devices, self.batch_per_device, self.devices)
        j_width = _torch_to_jax_sharded(widths, self.num_devices, self.batch_per_device, self.devices)
        j_rate = _torch_to_jax_sharded(rates, self.num_devices, self.batch_per_device, self.devices)

        # 2. Run Step
        next_states, next_outputs = self._parallel_step(
            self.current_states,
            self.last_outputs,
            j_trig, j_loc, j_width, j_rate,
            self.default_params
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
        
        # 3. 转换回 Torch
        # 注意：这里会发生 Gather，把 4 张卡的数据拼回来
        t_Te = _jax_sharded_to_torch(T_e) * 1000.0 # keV -> eV
        t_ne = _jax_sharded_to_torch(n_e)
        t_Pe = _jax_sharded_to_torch(P_e)
        t_Ti = _jax_sharded_to_torch(T_i) * 1000.0 # keV -> eV
        t_ni = _jax_sharded_to_torch(n_i)
        t_Pi = _jax_sharded_to_torch(P_i)

        # Species (假设 50-50)
        t_nD = t_ni * 0.5
        t_nT = t_ni * 0.5
        t_PD = t_nD * t_Ti * KEV_TO_JOULES
        t_PT = t_nT * t_Ti * KEV_TO_JOULES # Ti is in eV here, formula needs check but keeping structure
        
        return t_Te, t_ne, t_Pe, t_Ti, t_ni, t_Pi, (t_nD, t_Ti, t_PD), (t_nT, t_Ti, t_PT)

    def get_diagnostics(self):
        """返回关键标量诊断信息 (Torch Tensor)"""
        # print('T_e', self.current_states.core_profiles.T_e.value[0])
        out = self.last_outputs
        # shape: (Batch,)
        ne_vol = _jax_sharded_to_torch(out.fgw_n_e_volume_avg)
        Te_core = _jax_sharded_to_torch(self.current_states.core_profiles.T_e.value[..., 0]) # core
        return ne_vol.squeeze(), Te_core.squeeze()
    