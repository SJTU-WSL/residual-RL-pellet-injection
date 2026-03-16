import torch
import numpy as np
import dataclasses
from pathlib import Path
from typing import Tuple

from .src.torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
from .src.torch_pellet import TorchPellet, TorchPelletConfig
from .src.torch_layer import Model, TorchLayerConfig

@dataclasses.dataclass
class PelletSimulator:
    velocity: float = 400.0      
    thickness: float = 0.004     
    size: float = 0.004          
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    GFILE_PATH: str = "input/g067590.03300"
    TORAX_DT: float = 1e-3      

    def __post_init__(self):
        eqm_config = TorchEquilibriumConfig(device=self.device)
        self.GFILE_PATH = str(self._resolve_gfile_path(self.GFILE_PATH))
        self.equilibrium = TorchEquilibrium(self.GFILE_PATH, config=eqm_config)
        
        self.pellet_config = TorchPelletConfig(device=self.device)
        self.pellet_config.ode_method = 'euler'
        self.pellet_config.max_time = 15.0
        self._pellet = TorchPellet()
        self._layer_config = TorchLayerConfig(device=self.device)
        self._pellet.add_layer(
            Model.DT,
            0.0,
            {"D": 0.5, "T": 0.5},
            config=self._layer_config,
        )
        self._rho_axis_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _resolve_gfile_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        simulator_root = Path(__file__).resolve().parent
        resolved = simulator_root / candidate
        return resolved.resolve()

    def update_plasma_state(self, T_e, n_e, T_i, n_i, species_D=None, species_T=None):
        """
        同步等离子体剖面数据到平衡态管理器。
        
        Args:
            T_e: (Batch, Grid) 电子温度, 单位 eV
            n_e: (Batch, Grid) 电子密度, 单位 m^-3
            T_i: (Batch, Grid) 离子温度, 单位 eV
            n_i: (Batch, Grid) 离子密度, 单位 m^-3
            species_D, species_T: (可选) 组分数据，如果 T_i 已传入则不需要使用
        """
        # 1. 单位转换 (eV -> keV)
        # 物理模型通常使用 keV 计算 (Parks模型等)
        Te_keV = T_e / 1000.0
        Ti_keV = T_i / 1000.0

        # 2. 密度归一化/单位转换
        # 假设输入 n_e 是 SI 单位 (m^-3, 也就是 ~1e19 或 1e20 量级)
        # 很多 Scaling Law 习惯用 10^20 m^-3 作为单位 '1'
        # 这里保持你原有的逻辑：转换为 10^20 m^-3 单位
        ne_20 = n_e * 1e-20
        ni_20 = n_i * 1e-20

        # 3. 创建 Rho 网格 (GPU Tensor)
        # 【关键修正】使用 torch.linspace 而不是 np.linspace
        # 确保网格与数据在同一个 Device 上，避免报错
        batch_size, n_grid = T_e.shape
        device = T_e.device
        dtype = T_e.dtype
        
        rho_tensor = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)

        # 4. 更新平衡态 (Batch Update)
        # 确保你的 TorchEquilibrium.update_profiles 能接收 (Batch, Grid) 的输入
        self.equilibrium.update_profiles(
            rho=rho_tensor, 
            Te=Te_keV, 
            ne=ne_20,
            Ti=Ti_keV,
            ni=ni_20
        )
    '''
    def update_plasma_state(self, T_e, n_e, T_i, n_i, species_D, species_T):
        """
        接收的全是 (Batch, Grid) 的 Torch Tensor
        """
        nD, TD, _ = species_D
        nT, TT, _ = species_T
         
        # 输入单位 eV, 转为 keV
        Te_keV = T_e / 1000.0
        ni_total = nD + nT
        Ti_avg_ev = (nD * TD + nT * TT) / (ni_total + 1e-10)
        Ti_keV = Ti_avg_ev / 1000.0
        ne_converted = n_e * 1e-20

        # Batch Update
        self.equilibrium.update_profiles(
            rho=np.linspace(0, 1, Te_keV.shape[1]), 
            Te=Te_keV, 
            ne=ne_converted,
            Ti=Ti_keV,
            ni=ni_total * 1e-20
        )
    '''

    def simulate_pellet_injection(self, batch_size, velocity, thickness):
        """
        返回: loc, width, rate 均为 (Batch,) 的 Tensor
        """
        velocity = -velocity
        thk_cm = thickness * 100.0
        
        R_launch = float(self.equilibrium.R_max) + 0.2
        pos = torch.zeros((batch_size, 2), device=self.device)
        pos[:, 0] = R_launch
        pos[:, 1] = float(getattr(self.equilibrium, "Z_mid", 0.0)) 
        with torch.no_grad():
            results = self._pellet.inject_batch(pos, velocity, self.equilibrium, self.pellet_config, initial_radius=thk_cm)

        locs = results['avg_dep_rho'].float() # Ensure float32
   
        rho_dep = results['rho_deposition'] # (Batch, Steps)
        K = rho_dep.shape[-1]
        rho_axis_key = (K, rho_dep.device, rho_dep.dtype)
        rho_axis = self._rho_axis_cache.get(rho_axis_key)
        if rho_axis is None:
            rho_axis = torch.linspace(0.0, 1.0, K, device=rho_dep.device, dtype=rho_dep.dtype)
            self._rho_axis_cache[rho_axis_key] = rho_axis

        # 权重（沉积剖面），避免负数
        w = torch.clamp(rho_dep, min=0.0)  # (B, K)
        wsum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,1)

        # 加权均值: mu = sum(w*rho)/sum(w)
        mu = (w * rho_axis).sum(dim=-1, keepdim=True) / wsum  # (B,1)

        # 加权方差: var = sum(w*(rho-mu)^2)/sum(w)
        var = (w * (rho_axis - mu).pow(2)).sum(dim=-1) / wsum.squeeze(-1)  # (B,)

        widths = torch.sqrt(torch.clamp(var, min=0.0)).to(locs.dtype)  # (B,)
        # Rate 计算
        total_particles = self._calculate_total_particles_batch(thk_cm)
        dep_ratios = results['dep_ratio'].float() # (Batch,)
        rates = (total_particles * dep_ratios) / self.TORAX_DT
        
        return locs, widths, rates
    
    def _calculate_total_particles_batch(self, radius_cm: torch.Tensor) -> torch.Tensor:
        """
        向量化计算 Batch 弹丸的粒子总数。
        
        Args:
            radius_cm: (Batch,) 的 Tensor，单位 cm。
            
        Returns:
            (Batch,) 的 Tensor，表示粒子总数。
        """
        _solid_density = {'D': 0.2, 'T': 0.318, 'Li': 0.534, 'Ne': 1.44, 'C': 3.3}   # g/cm^3
        _atomic_weight = {'D': 2.014, 'T': 3.016, 'Li': 6.9, 'Ne': 20.183, 'C': 12.011}  # g/mol
        NA = 6.02214076e23
        
        ratio_dict = {"D": 0.5, "T": 0.5}

        total_ratio = sum(ratio_dict.values())
        normalized_ratio = {k: float(v)/total_ratio for k, v in ratio_dict.items()}
        
        molar_vol_mix = 0.0
        for comp, x in normalized_ratio.items():
            if comp not in _solid_density:
                raise ValueError(f"Unknown component: {comp}")
            
            molar_vol_i = _atomic_weight[comp] / _solid_density[comp]
            molar_vol_mix += x * molar_vol_i
            
        volume = (4.0 / 3.0) * torch.pi * torch.pow(radius_cm, 3)
        
        total_particles = (volume / molar_vol_mix) * NA
        
        return total_particles.to(dtype=torch.float32)
    
    def _calculate_total_particles(self, pellet):
        _solid_density = {'D': 0.2, 'T': 0.318, 'Li': 0.534, 'Ne': 1.44, 'C': 3.3}   # g/cm^3
        _atomic_weight = {'D': 2.014, 'T': 3.016, 'Li': 6.9, 'Ne': 20.183, 'C': 12.011}  # g/mol
        NA = 6.02214076e23
        def layer_particle_num(r_low_cm, r_high_cm, ratio_dict):
            # 体积：球壳体积 (cm^3)
            vol = (4.0/3.0) * np.pi * (r_high_cm**3 - r_low_cm**3)

            # 比例归一化
            s = sum(ratio_dict.values())
            ratio = {k: float(v)/s for k, v in ratio_dict.items()}

            # denom = Σ x_i * A_i / rho_i   (cm^3/mol)
            denom = 0.0
            for comp, x in ratio.items():
                denom += x * (_atomic_weight[comp] / _solid_density[comp])

            # 粒子数
            return vol / denom * NA

        if hasattr(pellet, "layers") and pellet.layers is not None and len(pellet.layers) > 0:
            total = 0.0
            for lyr in pellet.layers:
                # 兼容不同字段名（你 TorchLayer 里一般会有 r_low/r_high/ratio）
                r_low = float(getattr(lyr, "r_low", 0.0))
                r_high = float(getattr(lyr, "r_high", r_low + float(getattr(lyr, "thickness", 0.0))))
                ratio = getattr(lyr, "ratio", None)
                if ratio is None:
                    raise AttributeError("pellet.layers 里的 layer 没有 ratio 字段，无法算粒子数")

                total += layer_particle_num(r_low, r_high, ratio)
            return torch.tensor(total, device=self.device, dtype=torch.float32)
        
        radius_cm = float(pellet.radius)
        ratio_dt = {"D": 0.5, "T": 0.5}  # 你当前就是这么 add_layer 的
        total = layer_particle_num(0.0, radius_cm, ratio_dt)
        return torch.tensor(total, device=self.device, dtype=torch.float32)
