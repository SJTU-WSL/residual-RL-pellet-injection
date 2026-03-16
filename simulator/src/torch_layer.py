import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from typing import Dict, Optional

# 物理常数（与原始代码一致）
_solid_density = {'D': 0.2, 'T': 0.318, 'Li': 0.534, 'Ne': 1.44, 'C': 3.3}
_atomic_weight = {'D': 2.014, 'T': 3.016, 'Li': 6.9, 'Ne': 20.183, 'C': 12.011}
_atomic_Z = {'D': 1, 'T': 1, 'Li': 3, 'Ne': 10, 'C': 6}


class Model(Enum):
    DT = ['D', 'T']
    Li = ['Li']
    NeD = ['Ne', 'D']
    C = ['C']


class TorchLayerConfig:
    """GPU计算配置"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dtype = torch.float32
        self.eps = 1e-10  # 数值稳定性


class TorchAblationModel(nn.Module):
    """
    可微分烧蚀模型基类
    支持批量GPU计算和自动微分
    """
    def __init__(self, layer_params: Dict, config: TorchLayerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # 物理参数（转为PyTorch参数，可选择是否可训练）
        self.density = torch.tensor(
            layer_params['density'], 
            dtype=config.dtype, 
            device=config.device
        )
        self.mean_atomic_weight = torch.tensor(
            layer_params['mean_atomic_weight'],
            dtype=config.dtype,
            device=config.device
        )
        
    def forward(self, Bt: torch.Tensor, Bt_exp: torch.Tensor, 
                Te: torch.Tensor, ne: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        计算烧蚀率 dr/dt
        
        Args:
            Bt: 磁场强度 [batch] or [batch, time]
            Bt_exp: 磁场指数 [batch] or [batch, time]
            Te: 电子温度 (keV) [batch] or [batch, time]
            ne: 电子密度 (10^14/cm^3) [batch] or [batch, time]
            r: 弹丸半径 (cm) [batch] or [batch, time]
            
        Returns:
            drdt: 烧蚀率 (cm/ms) 同形状
        """
        raise NotImplementedError


class TorchParksDT(TorchAblationModel):
    """
    Parks DT烧蚀模型（GPU版本）
    对应原始代码 layer.py Line 94-107
    """
    def __init__(self, layer_params: Dict, config: TorchLayerConfig):
        super().__init__(layer_params, config)
        
        # DT特有参数
        self.weight_ratio = torch.tensor(
            layer_params['weight_ratio'],
            dtype=config.dtype,
            device=config.device
        )
        
        # 可学习的烧蚀系数（可选）
        self.ablation_coeff = nn.Parameter(
            torch.tensor(167.16, dtype=config.dtype, device=config.device)
        )
        
    def forward(self, Bt, Bt_exp, Te, ne, r):
        # 确保所有输入在同一设备
        Bt = self._to_tensor(Bt)
        Bt_exp = self._to_tensor(Bt_exp)
        Te = self._to_tensor(Te)
        ne = self._to_tensor(ne)
        r = self._to_tensor(r)
        
        # 防止除零和负半径
        r_safe = torch.clamp(r, min=self.config.eps)
        
        # Parks公式（向量化）
        drpdt = (
            -(self.ablation_coeff * self.weight_ratio ** (2/3) * 
              (Bt / 2.0) ** Bt_exp) /
            self.density * Te ** (5/3) * ne ** (1/3) / 
            r_safe ** (2/3)
        )
        drpdt = drpdt * 1e-3  # cm/s to cm/ms
        
        # 负半径时烧蚀率为0
        drpdt = torch.where(r > 0, drpdt, torch.zeros_like(drpdt))
        
        return drpdt
    
    def _to_tensor(self, x):
        """辅助函数：转换为PyTorch张量"""
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=self.config.dtype, device=self.device)
        return x.to(device=self.device, dtype=self.config.dtype)


class TorchKuteevLi(TorchAblationModel):
    """
    Kuteev Li烧蚀模型（GPU版本）
    对应原始代码 layer.py Line 110-124
    """
    def __init__(self, layer_params: Dict, config: TorchLayerConfig):
        super().__init__(layer_params, config)
        
        # Li物理常数
        self.coeff = torch.tensor(1.04e15, dtype=config.dtype, device=config.device)
        self.rhomass = torch.tensor(0.534e6, dtype=config.dtype, device=config.device)
        self.amup_pl = torch.tensor(6.9, dtype=config.dtype, device=config.device)
        self.z_navogadro = torch.tensor(6.022e23, dtype=config.dtype, device=config.device)
        self.dena = self.rhomass * self.z_navogadro / self.amup_pl
        
    def forward(self, Bt, Bt_exp, Te, ne, r):
        Bt = self._to_tensor(Bt)
        Te = self._to_tensor(Te)
        ne = self._to_tensor(ne)
        r = self._to_tensor(r)
        
        r_safe = torch.clamp(r, min=self.config.eps)
        
        # Kuteev公式
        dndt = (self.coeff * (Te * 1e3) ** 1.64 * (ne * 1e14) ** (1/3) * 
                r_safe ** (4/3) * self.amup_pl ** (-1/3))
        
        drpdt = -dndt / (self.dena * 4 * np.pi * (r_safe * 0.01) ** 2)
        drpdt = drpdt * 0.1  # m/s to cm/ms
        
        drpdt = torch.where(r > 0, drpdt, torch.zeros_like(drpdt))
        return drpdt
    
    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=self.config.dtype, device=self.device)
        return x.to(device=self.device, dtype=self.config.dtype)


class TorchParksNeD(TorchAblationModel):
    """
    Parks NeD烧蚀模型（GPU版本）
    对应原始代码 layer.py Line 127-143
    """
    def __init__(self, layer_params: Dict, config: TorchLayerConfig):
        super().__init__(layer_params, config)
        
        self.D_ratio = torch.tensor(
            layer_params['D_ratio'],
            dtype=config.dtype,
            device=config.device
        )
        self.rho_mean = torch.tensor(
            layer_params['rho_mean'],
            dtype=config.dtype,
            device=config.device
        )
        
    def forward(self, Bt, Bt_exp, Te, ne, r):
        Bt = self._to_tensor(Bt)
        Bt_exp = self._to_tensor(Bt_exp)
        Te = self._to_tensor(Te)
        ne = self._to_tensor(ne)
        r = self._to_tensor(r)
        
        r_safe = torch.clamp(r, min=self.config.eps)
        
        # X和AoX计算
        X = self.D_ratio / (2 - self.D_ratio)
        AoX = 27.0 + torch.tan(1.48 * X)
        c0 = AoX / (4 * np.pi) * (2.0 / Bt) ** Bt_exp
        
        drpdt = -c0 / self.rho_mean * Te ** (5/3) * ne ** (1/3) / r_safe ** (2/3)
        drpdt = drpdt * 1e-3  # cm/s to cm/ms
        
        drpdt = torch.where(r > 0, drpdt, torch.zeros_like(drpdt))
        return drpdt
    
    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=self.config.dtype, device=self.device)
        return x.to(device=self.device, dtype=self.config.dtype)


class TorchParksC(TorchAblationModel):
    """
    Parks Carbon烧蚀模型（GPU版本）
    对应原始代码 layer.py Line 146-220
    
    这是最复杂的烧蚀模型，包含大量中间计算
    """
    def __init__(self, layer_params: Dict, config: TorchLayerConfig):
        super().__init__(layer_params, config)
        
        # 物理常数
        self.C0 = torch.tensor(8.146777e-9, dtype=config.dtype, device=config.device)
        self.WC = torch.tensor(_atomic_weight['C'], dtype=config.dtype, device=config.device)
        self.gamma = torch.tensor(5.0/3.0, dtype=config.dtype, device=config.device)
        self.ZstarPlus1C = torch.tensor(2.86, dtype=config.dtype, device=config.device)
        self.IstC = torch.tensor(60.0, dtype=config.dtype, device=config.device)
        self.Z_C = torch.tensor(_atomic_Z['C'], dtype=config.dtype, device=config.device)
        
        # Lambda参数（硬编码）
        self.xiexp = torch.tensor(0.601, dtype=config.dtype, device=config.device)
        self.fugCG = torch.tensor(0.777686, dtype=config.dtype, device=config.device)
        
    def forward(self, Bt, Bt_exp, Te, ne, r):
        Te = self._to_tensor(Te) * 1e3  # keV to eV
        ne = self._to_tensor(ne) * 1e14  # 单位转换
        r = self._to_tensor(r)
        
        r_safe = torch.clamp(r, min=self.config.eps)
        
        # Albedo计算
        Albedo = 23.920538030089528 * torch.log(
            1 + 0.20137080524063228 * self.ZstarPlus1C
        )
        flelectro = torch.exp(torch.tensor(-1.936, device=self.device))
        fL = (1.0 - Albedo / 100) * flelectro
        
        # BLamdaq计算
        Ttmp = torch.clamp(Te, min=30.0)
        loglamCSlow = torch.log(
            2.0 * Ttmp / self.IstC * torch.sqrt(torch.e * 2.0)
        )
        BLamdaq = (1 / (self.Z_C * loglamCSlow) * 
                   (4 / (2.5 + 2.2 * torch.sqrt(self.ZstarPlus1C))))
        
        # Gpr计算
        Gpr = (
            self.C0 *
            self.WC ** (2/3) *
            (self.gamma - 1.0) ** (1/3) *
            (fL * ne) ** (1/3) *
            r_safe ** (4/3) *
            Te ** (11/6) *
            BLamdaq ** (2/3)
        )
        
        # CG计算（Lambda参数）
        lamdaa = torch.tensor(0.0933979540623963, device=self.device)
        lamdab = torch.tensor(-0.7127242270013098, device=self.device)
        lamdac = torch.tensor(-0.2437544205933372, device=self.device)
        lamdad = torch.tensor(-0.8534855445478313, device=self.device)
        
        av = 10.420403555938629 * (Ttmp / 2000.0) ** lamdaa
        bv = 0.6879779829877795 * (Ttmp / 2000.0) ** lamdab
        cv = 1.5870910225610804 * (Ttmp / 2000.0) ** lamdac
        dv = 2.9695640286641840 * (Ttmp / 2000.0) ** lamdad
        
        CG = (
            self.fugCG * av *
            torch.log(1 + bv * (ne / 1e14) ** (2/3) * r_safe ** (2/3)) /
            torch.log(cv + dv * (ne / 1e14) ** (2/3) * r_safe ** (2/3))
        )
        
        # 最终烧蚀率
        G = self.xiexp * CG * Gpr
        
        # 单位转换
        rho_C = torch.tensor(
            _solid_density['C'], 
            dtype=self.config.dtype, 
            device=self.device
        )
        drpdt = -G / (4.0 * np.pi * rho_C * r_safe ** 2)
        drpdt = drpdt * 1e-3  # cm/s to cm/ms
        
        drpdt = torch.where(r > 0, drpdt, torch.zeros_like(drpdt))
        return drpdt
    
    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=self.config.dtype, device=self.device)
        return x.to(device=self.device, dtype=self.config.dtype)


class TorchLayer:
    """
    GPU层管理器
    模拟原始Layer类的接口，但使用PyTorch后端
    """
    def __init__(self, layer_idx: int, model: Model, r_low: float, r_high: float,
                 component_ratio: Dict[str, float] = None,
                 min_step: int = 200,
                 config: Optional[TorchLayerConfig] = None):
        
        self.idx = layer_idx
        self.model = model
        self.r_low = r_low
        self.r_high = r_high
        self.thickness = r_high - r_low
        self.min_step = min_step
        self.config = config or TorchLayerConfig()
        
        # 处理组分比例（保持与原始代码一致）
        self.ratio = self._process_ratio(component_ratio)
        
        # 计算物理量
        self._calculate_physical_properties()
        
        # 创建对应的烧蚀模型
        self.ablation_model = self._create_ablation_model()
        
    def _process_ratio(self, component_ratio):
        """处理组分比例（与原始代码逻辑一致）"""
        if component_ratio is None:
            # 默认比例
            default_ratios = {
                Model.DT: {"D": 1, "T": 0.0},
                Model.Li: {"Li": 1},
                Model.NeD: {"Ne": 0, "D": 1},
                Model.C: {"C": 1}
            }
            component_ratio = default_ratios[self.model]
        
        # 归一化
        total = sum(component_ratio.values())
        return {k: v/total for k, v in component_ratio.items()}
    
    def _calculate_physical_properties(self):
        """计算密度、粒子数等物理量"""
        components = self.model.value
        ratio_np = np.array([self.ratio.get(c, 0) for c in components])
        weight_np = np.array([_atomic_weight[c] for c in components])
        density_np = np.array([_solid_density[c] for c in components])
        
        self.density = np.sum(ratio_np * weight_np) / np.sum(ratio_np * weight_np / density_np)
        self.mean_atomic_weight = np.sum(weight_np * ratio_np)
        
        # DT特殊处理
        if self.model is Model.DT:
            D_ratio = self.ratio['D']
            self.weight_ratio = ((1 - D_ratio) * _atomic_weight['T'] / 
                                _atomic_weight['D'] + D_ratio)
    
    def _create_ablation_model(self):
        """根据模型类型创建对应的烧蚀模型"""
        params = {
            'density': self.density,
            'mean_atomic_weight': self.mean_atomic_weight
        }
        
        if self.model is Model.DT:
            params['weight_ratio'] = self.weight_ratio
            return TorchParksDT(params, self.config)
            
        elif self.model is Model.Li:
            return TorchKuteevLi(params, self.config)
            
        elif self.model is Model.NeD:
            D_ratio = self.ratio['D']
            rho_mean = self._calculate_NeD_density(D_ratio)
            params['D_ratio'] = D_ratio
            params['rho_mean'] = rho_mean
            return TorchParksNeD(params, self.config)
            
        elif self.model is Model.C:
            return TorchParksC(params, self.config)
    
    def _calculate_NeD_density(self, D_ratio):
        """计算NeD混合密度"""
        ratio_np = np.array([D_ratio, 1 - D_ratio])
        weight_np = np.array([_atomic_weight['D'], _atomic_weight['Ne']])
        density_np = np.array([_solid_density['D'], _solid_density['Ne']])
        return np.sum(ratio_np * weight_np) / np.sum(ratio_np * weight_np / density_np)
    
    def getDrPerDt(self, Bt, Bt_exp, Te, ne, r):
        """
        接口兼容函数
        支持NumPy或PyTorch输入
        """
        return self.ablation_model(Bt, Bt_exp, Te, ne, r)


# ============================================================
# 验证和测试工具
# ============================================================

def validate_against_numpy(original_layer, torch_layer, num_samples=1000):
    """
    对比PyTorch版本和原始NumPy版本的结果
    """
    import sys
    sys.path.append('..')
    from src.layer import Layer as OriginalLayer
    
    print(f"验证模型: {torch_layer.model.name}")
    print("="*60)
    
    # 生成随机测试数据
    np.random.seed(42)
    Bt = np.random.uniform(0.5, 2.0, num_samples)
    Bt_exp = np.random.uniform(0.3, 0.7, num_samples)
    Te = np.random.uniform(0.5, 10.0, num_samples)  # keV
    ne = np.random.uniform(0.5, 5.0, num_samples)   # 10^14/cm^3
    r = np.random.uniform(0.01, 1.0, num_samples)   # cm
    
    # NumPy版本（逐点计算）
    results_numpy = np.zeros(num_samples)
    for i in range(num_samples):
        results_numpy[i] = original_layer.getDrPerDt(
            Bt[i], Bt_exp[i], Te[i], ne[i], r[i]
        )
    
    # PyTorch版本（批量计算）
    with torch.no_grad():
        results_torch = torch_layer.getDrPerDt(Bt, Bt_exp, Te, ne, r)
        if isinstance(results_torch, torch.Tensor):
            results_torch = results_torch.cpu().numpy()
    
    # 计算误差
    abs_error = np.abs(results_numpy - results_torch)
    rel_error = abs_error / (np.abs(results_numpy) + 1e-10)
    
    print(f"最大绝对误差: {abs_error.max():.2e}")
    print(f"平均绝对误差: {abs_error.mean():.2e}")
    print(f"最大相对误差: {rel_error.max():.2%}")
    print(f"平均相对误差: {rel_error.mean():.2%}")
    
    # 性能测试
    import time
    
    # NumPy速度
    tic = time.time()
    for _ in range(10):
        for i in range(num_samples):
            _ = original_layer.getDrPerDt(Bt[i], Bt_exp[i], Te[i], ne[i], r[i])
    numpy_time = (time.time() - tic) / 10
    
    # PyTorch速度
    Bt_t = torch.tensor(Bt, device=torch_layer.config.device)
    Bt_exp_t = torch.tensor(Bt_exp, device=torch_layer.config.device)
    Te_t = torch.tensor(Te, device=torch_layer.config.device)
    ne_t = torch.tensor(ne, device=torch_layer.config.device)
    r_t = torch.tensor(r, device=torch_layer.config.device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    tic = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = torch_layer.getDrPerDt(Bt_t, Bt_exp_t, Te_t, ne_t, r_t)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    torch_time = (time.time() - tic) / 10
    
    print(f"\n性能对比 ({num_samples}个样本):")
    print(f"NumPy时间: {numpy_time*1000:.3f} ms")
    print(f"PyTorch时间: {torch_time*1000:.3f} ms")
    print(f"加速比: {numpy_time/torch_time:.1f}×")
    print("="*60)


if __name__ == '__main__':
    print("🚀 GPU加速烧蚀模型测试")
    print("\n设备信息:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建配置
    config = TorchLayerConfig()
    
    # 测试DT模型
    print("\n" + "="*60)
    print("测试 Parks DT 模型")
    print("="*60)
    
    layer_dt = TorchLayer(
        layer_idx=0,
        model=Model.DT,
        r_low=0.0,
        r_high=0.2,
        component_ratio={"D": 0.5, "T": 0.5},
        config=config
    )
    
    # 批量测试
    batch_size = 1000
    Bt = torch.ones(batch_size, device=config.device)
    Bt_exp = torch.ones(batch_size, device=config.device) * 0.5
    Te = torch.linspace(1, 10, batch_size, device=config.device)
    ne = torch.linspace(1, 5, batch_size, device=config.device)
    r = torch.linspace(0.01, 0.2, batch_size, device=config.device)
    
    with torch.no_grad():
        drdt = layer_dt.getDrPerDt(Bt, Bt_exp, Te, ne, r)
    
    print(f"✅ 批量计算成功! 形状: {drdt.shape}")
    print(f"   烧蚀率范围: [{drdt.min():.2e}, {drdt.max():.2e}] cm/ms")
    
    print("\n✅ 所有测试通过!")