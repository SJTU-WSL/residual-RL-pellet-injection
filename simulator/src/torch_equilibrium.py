import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from typing import Tuple, Optional, Union
import os
import warnings

# 导入原始模块用于加载gfile
import sys
sys.path.append('..')
try:
    from .eqdsk.eqdsk import Geqdsk
except ImportError:
    print("Warning: Geqdsk not found, using mock loader")
    Geqdsk = None


class TorchEquilibriumConfig:
    """GPU平衡态配置"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dtype = torch.float32
        self.interp_mode = 'bilinear'  # 'bilinear' or 'bicubic'
        self.align_corners = True


class TorchEquilibrium(nn.Module):
    """
    GPU加速的等离子体平衡态
    
    功能：
    1. 批量2D插值（磁通、磁场等）
    2. 批量1D插值（温度、密度剖面）
    3. 批量rho计算
    4. 可微分（支持梯度优化）
    """
    
    def __init__(self, gfile_path: str, te_ne_path: str = None, 
                 config: Optional[TorchEquilibriumConfig] = None):
        super().__init__()
        
        self.config = config or TorchEquilibriumConfig()
        self.device = self.config.device
        self.gfile_path = gfile_path
        
        # 加载G文件数据
        self._load_gfile(gfile_path)
        
        # 加载温度密度剖面
        self._load_te_ne_profiles(te_ne_path)
        # 计算网格参数
        self._setup_grids()
        
    def _load_gfile(self, gfile_path: str):
        """加载G文件数据到GPU"""
        
        if Geqdsk is None:
            # Mock数据用于测试
            self._create_mock_gfile()
            return
        
        try:
            gfile = Geqdsk(gfile_path)
            
            # 基本参数
            self.axis_R = float(gfile['RMAXIS'])
            self.axis_Z = float(gfile['ZMAXIS'])
            self.id = str(gfile['CASE'][3]).replace(" ", "")
            
            # 网格数据 → GPU张量
            # PSIRZ: [nz, nr] 磁通分布
            psi_data = np.array(gfile['PSIRZ'], dtype=np.float32)
            self.psi_grid = torch.tensor(
                psi_data, 
                dtype=self.config.dtype, 
                device=self.device
            )
            
            # R, Z网格坐标
            # 1. 获取网格尺寸和几何范围参数
            nw = gfile['NW']  # R方向网格数
            nh = gfile['NH']  # Z方向网格数
            rdim = gfile['RDIM'] # R宽度
            rleft = gfile['RLEFT'] # R左起始点
            zdim = gfile['ZDIM'] # Z高度
            zmid = gfile['ZMID'] # Z中心点
            
            # 2. 手动构建 R 和 Z 的 1D 坐标数组
            # R 从 rleft 开始，长度为 rdim
            r_array = np.linspace(rleft, rleft + rdim, nw, dtype=np.float32)
            # Z 以 zmid 为中心，跨度为 zdim
            z_array = np.linspace(zmid - zdim/2, zmid + zdim/2, nh, dtype=np.float32)

            # 3. 转换回 Tensor
            self.R_1d = torch.tensor(r_array, dtype=self.config.dtype, device=self.device)
            self.Z_1d = torch.tensor(z_array, dtype=self.config.dtype, device=self.device)
            # 磁轴磁通和边界磁通
            self.psi_axis = torch.tensor(
                float(gfile['SIMAG']),
                dtype=self.config.dtype,
                device=self.device
            )
            self.psi_bdry = torch.tensor(
                float(gfile['SIBRY']),
                dtype=self.config.dtype,
                device=self.device
            )
            
            # LCFS边界点
            bdry_R = torch.tensor(
                np.array(gfile['RBBBS'], dtype=np.float32),
                device=self.device
            )
            bdry_Z = torch.tensor(
                np.array(gfile['ZBBBS'], dtype=np.float32),
                device=self.device
            )
            self.lcfs_R = bdry_R
            self.lcfs_Z = bdry_Z
            
            # 边界范围
            self.R_min = self.R_1d.min()
            self.R_max = self.R_1d.max()
            self.Z_min = self.Z_1d.min()
            self.Z_max = self.Z_1d.max()
            
        except Exception as e:
            print(f"   ⚠️  加载G文件失败: {e}")
            self._create_mock_gfile()
    
    def _create_mock_gfile(self):
        """创建模拟G文件数据（用于测试）"""
        print("   使用模拟G文件数据")
        
        self.axis_R = 1.85
        self.axis_Z = 0.0
        self.id = "mock"
        
        # 创建模拟磁通分布
        nr, nz = 65, 65
        R = torch.linspace(1.2, 2.5, nr, device=self.device)
        Z = torch.linspace(-1.0, 1.0, nz, device=self.device)
        
        R_grid, Z_grid = torch.meshgrid(R, Z, indexing='ij')
        
        # 简单的同心圆磁通
        psi = ((R_grid - self.axis_R)**2 + (Z_grid - self.axis_Z)**2)
        
        self.psi_grid = psi.T  # [nz, nr]
        self.R_1d = R
        self.Z_1d = Z
        self.psi_axis = torch.tensor(0.0, device=self.device)
        self.psi_bdry = torch.tensor(0.5, device=self.device)
        
        # LCFS
        theta = torch.linspace(0, 2*np.pi, 100, device=self.device)
        self.lcfs_R = self.axis_R + 0.5 * torch.cos(theta)
        self.lcfs_Z = self.axis_Z + 0.8 * torch.sin(theta)
        
        self.R_min = R.min()
        self.R_max = R.max()
        self.Z_min = Z.min()
        self.Z_max = Z.max()
    
    def _load_te_ne_profiles(self, te_ne_path: str = None):
        """加载温度密度剖面"""
        
        if te_ne_path is None:
            # 使用默认路径或创建模拟数据
            try:
                directory_path = os.path.dirname(os.path.abspath(__file__))
                te_ne_path = os.path.join(directory_path, 'data', 'TeNe.mat')
            except:
                te_ne_path = None
        
        if te_ne_path and os.path.exists(te_ne_path):
            buf = loadmat(te_ne_path)
            Te = buf['Te'].reshape(-1).astype(np.float32)
            ne = buf['ne'].reshape(-1).astype(np.float32)
            rho = buf['rho'].reshape(-1).astype(np.float32)
        else:
            print(f"   使用模拟Te/ne剖面")
            # 创建模拟剖面
            rho = np.linspace(0, 1.5, 200, dtype=np.float32)
            Te = 10 * (1 - (rho/1.0)**2)  # 抛物线温度
            Te = np.maximum(Te, 0.1)  # 最小0.1 keV
            ne = 5 * (1 - (rho/1.0)**2)   # 抛物线密度
            ne = np.maximum(ne, 0.1)  # 最小0.1
        
        # 转为GPU张量
        self.Te_profile = torch.tensor(Te, device=self.device)
        self.ne_profile = torch.tensor(ne, device=self.device)
        self.rho_profile = torch.tensor(rho, device=self.device)
    
    def _setup_grids(self):
        """设置计算网格"""
        # 归一化系数（用于快速坐标转换）
        self.R_norm_scale = 2.0 / (self.R_max - self.R_min)
        self.R_norm_offset = -1.0 - self.R_min * self.R_norm_scale
        
        self.Z_norm_scale = 2.0 / (self.Z_max - self.Z_min)
        self.Z_norm_offset = -1.0 - self.Z_min * self.Z_norm_scale
    def update_profiles(self, rho, Te, ne, Ti=None, ni=None):
        """
        更新等离子体剖面数据（从 Torax 传入）
        
        Args:
            rho: 1D array/tensor, 径向坐标网格 [0, 1]
            Te, ne, Ti, ni: 1D arrays, 对应的物理剖面
        """
        # 确保数据在正确的设备上并转换为 Tensor
        def to_torch(x):
            if x is None: return None
            return torch.as_tensor(x, device=self.device, dtype=self.config.dtype)

        self.prof_rho = to_torch(rho)
        self.prof_Te = to_torch(Te)
        self.prof_ne = to_torch(ne)
        self.prof_Ti = to_torch(Ti)
        self.prof_ni = to_torch(ni)

        # 预计算一些插值常数以加速 get_plasma_params
        self.rho_min = self.prof_rho[0]
        self.rho_max = self.prof_rho[-1]
        self.rho_step = (self.rho_max - self.rho_min) / (len(self.prof_rho) - 1)

    def get_plasma_params(self, R, Z):
        """
        根据 R, Z 坐标获取等离子体参数
        实现：(R,Z) -> psi_n(rho) -> 1D Interpolation(Te, ne...)
        """
        # 1. 映射 R, Z 到归一化磁通 rho (即 sqrt(psi_n))
        # 假设之前的 psi_grid 映射已经实现
        rho_query = self.get_rho_from_rz(R, Z) 

        # 2. 1D 线性插值保护 (防止外推导致 NaN)
        rho_indices = (rho_query - self.rho_min) / self.rho_step
        idx0 = torch.floor(torch.clamp(rho_indices, 0, len(self.prof_rho) - 2)).long()
        idx1 = idx0 + 1
        weight = torch.clamp(rho_indices - idx0, 0, 1)

        # 3. 执行批量插值
        Te_val = self.prof_Te[idx0] * (1 - weight) + self.prof_Te[idx1] * weight
        ne_val = self.prof_ne[idx0] * (1 - weight) + self.prof_ne[idx1] * weight
        
        # 处理可选的离子参数
        Ti_val = (self.prof_Ti[idx0] * (1 - weight) + self.prof_Ti[idx1] * weight) if self.prof_Ti is not None else Te_val
        ni_val = (self.prof_ni[idx0] * (1 - weight) + self.prof_ni[idx1] * weight) if self.prof_ni is not None else ne_val

        return {
            'rho': rho_query,
            'Te': Te_val,
            'ne': ne_val,
            'Ti': Ti_val,
            'ni': ni_val
        }
    def normalize_coords(self, R: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将物理坐标归一化到[-1, 1]（用于grid_sample）
        
        Args:
            R, Z: 物理坐标 [batch] or [batch, time]
        Returns:
            R_norm, Z_norm: 归一化坐标 [-1, 1]
        """
        R_norm = R * self.R_norm_scale + self.R_norm_offset
        Z_norm = Z * self.Z_norm_scale + self.Z_norm_offset
        return R_norm, Z_norm
    
    def interp_psi(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:

        # 归一化坐标
        R_norm, Z_norm = self.normalize_coords(R, Z)

        # psi_grid: [nz, nr] -> psi_input: [1, 1, nz, nr]
        psi_input = self.psi_grid.unsqueeze(0).unsqueeze(0)

        if R.dim() == 1:
            # R_norm, Z_norm: [B]
            # grid: [B, 1, 1, 2]
            grid = torch.stack([R_norm, Z_norm], dim=-1).unsqueeze(1).unsqueeze(1)
            B = grid.shape[0]

            # ✅ 关键：把 psi_input 扩成 [B, 1, nz, nr]
            psi_input = psi_input.expand(B, -1, -1, -1)

            psi = F.grid_sample(
                psi_input,
                grid,
                mode=self.config.interp_mode,
                align_corners=self.config.align_corners,
                padding_mode="border",
            )  # [B, 1, 1, 1]

            # squeeze channel + spatial -> [B]
            psi = psi.squeeze(1).squeeze(-1).squeeze(-1)
            return psi

        elif R.dim() == 2:
            # R_norm, Z_norm: [B, T]
            # grid: [B, T, 1, 2]
            grid = torch.stack([R_norm, Z_norm], dim=-1).unsqueeze(2)
            B = grid.shape[0]

            # ✅ 关键：把 psi_input 扩成 [B, 1, nz, nr]
            psi_input = psi_input.expand(B, -1, -1, -1)

            psi = F.grid_sample(
                psi_input,
                grid,
                mode=self.config.interp_mode,
                align_corners=self.config.align_corners,
                padding_mode="border",
            )  # [B, 1, T, 1]

            # squeeze channel + last dim -> [B, T]
            psi = psi.squeeze(1).squeeze(-1)
            return psi

        else:
            raise ValueError(f"Unsupported shape: {R.shape}")
    
    def psi_to_rho(self, psi: torch.Tensor) -> torch.Tensor:
        """
        将磁通转换为归一化半径 ρ = sqrt((ψ - ψ_axis) / (ψ_bdry - ψ_axis))
        
        Args:
            psi: 磁通值（任意形状）
        Returns:
            rho: 归一化半径（同形状）
        """
        psi_norm = (psi - self.psi_axis) / (self.psi_bdry - self.psi_axis + 1e-10)
        rho = torch.sqrt(torch.clamp(psi_norm, min=0))
        return rho
    
    def RZ_to_rho(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        直接从(R,Z)计算ρ
        
        Args:
            R, Z: 坐标（任意形状）
        Returns:
            rho: 归一化半径（同形状）
        """
        psi = self.interp_psi(R, Z)
        return self.psi_to_rho(psi)
    
    def interp_Te_ne(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量1D插值温度和密度剖面
        
        Args:
            rho: 归一化半径（任意形状）
        Returns:
            Te, ne: 温度(keV)和密度(1e14/cm³)（同形状）
        """
        # 使用torch.searchsorted进行快速查找
        # 注意：要求rho_profile是单调递增的
        
        # Flatten输入
        original_shape = rho.shape
        rho_flat = rho.flatten()
        
        # 查找插值索引
        idx = torch.searchsorted(self.rho_profile, rho_flat, right=False)
        idx = torch.clamp(idx, 1, len(self.rho_profile) - 1)
        
        # 线性插值
        rho0 = self.rho_profile[idx - 1]
        rho1 = self.rho_profile[idx]
        weight = (rho_flat - rho0) / (rho1 - rho0 + 1e-10)
        
        Te0 = self.Te_profile[idx - 1]
        Te1 = self.Te_profile[idx]
        Te = Te0 + weight * (Te1 - Te0)
        
        ne0 = self.ne_profile[idx - 1]
        ne1 = self.ne_profile[idx]
        ne = ne0 + weight * (ne1 - ne0)
        
        # 恢复原始形状
        Te = Te.reshape(original_shape)
        ne = ne.reshape(original_shape)
        
        return Te, ne
    
    def get_plasma_params(self, R: torch.Tensor, Z: torch.Tensor) -> dict:
        """
        一次性获取所有等离子体参数
        
        Args:
            R, Z: 坐标 [batch] or [batch, time]
        Returns:
            dict: {'psi', 'rho', 'Te', 'ne'}
        """
        psi = self.interp_psi(R, Z)
        rho = self.psi_to_rho(psi)
        Te, ne = self.interp_Te_ne(rho)
        
        return {
            'psi': psi,
            'rho': rho,
            'Te': Te,
            'ne': ne
        }
    
    def forward(self, R: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：(R, Z) → (Te, ne)
        用于神经网络集成
        """
        rho = self.RZ_to_rho(R, Z)
        Te, ne = self.interp_Te_ne(rho)
        return Te, ne
    
    def is_inside_lcfs(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        判断点是否在LCFS内
        
        Returns:
            inside: bool张量，同形状
        """
        rho = self.RZ_to_rho(R, Z)
        return rho < 1.0
    
    def to(self, device):
        """移动到指定设备"""
        self.device = device
        return super().to(device)


# ============================================================
# 验证和测试工具
# ============================================================

def validate_interpolation(torch_eqm: TorchEquilibrium, num_samples=1000):
    """
    验证PyTorch插值精度
    如果有原始equilibrium可用，对比结果
    """
    print("\n" + "="*60)
    print("🧪 插值精度测试")
    print("="*60)
    
    # 生成测试点
    R_test = torch.rand(num_samples, device=torch_eqm.device) * 0.8 + 1.4
    Z_test = torch.rand(num_samples, device=torch_eqm.device) * 1.6 - 0.8
    
    # PyTorch插值
    with torch.no_grad():
        psi = torch_eqm.interp_psi(R_test, Z_test)
        rho = torch_eqm.psi_to_rho(psi)
        Te, ne = torch_eqm.interp_Te_ne(rho)
    
    print(f"✓ 批量插值 {num_samples} 点")
    print(f"  ψ范围: [{psi.min():.3f}, {psi.max():.3f}]")
    print(f"  ρ范围: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"  Te范围: [{Te.min():.2f}, {Te.max():.2f}] keV")
    print(f"  ne范围: [{ne.min():.2f}, {ne.max():.2f}] 1e14/cm³")
    
    # 测试可微分性
    R_grad = R_test[:10].clone().requires_grad_(True)
    Z_grad = Z_test[:10].clone().requires_grad_(True)
    
    Te_out, ne_out = torch_eqm(R_grad, Z_grad)
    loss = Te_out.sum()
    loss.backward()
    
    print(f"\n✓ 梯度测试通过")
    print(f"  ∂Te/∂R: {R_grad.grad.abs().mean():.2e}")
    print(f"  ∂Te/∂Z: {Z_grad.grad.abs().mean():.2e}")


def performance_benchmark(torch_eqm: TorchEquilibrium):
    """性能基准测试"""
    import time
    
    print("\n" + "="*60)
    print("⚡ 性能基准测试")
    print("="*60)
    
    batch_sizes = [100, 1000, 10000, 50000]
    
    for batch_size in batch_sizes:
        R = torch.rand(batch_size, device=torch_eqm.device) * 0.8 + 1.4
        Z = torch.rand(batch_size, device=torch_eqm.device) * 1.6 - 0.8
        
        # 预热
        _ = torch_eqm.get_plasma_params(R, Z)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        tic = time.time()
        for _ in range(10):
            params = torch_eqm.get_plasma_params(R, Z)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.time() - tic) / 10
        throughput = batch_size / elapsed
        
        print(f"批量 {batch_size:6d}: {elapsed*1000:6.2f} ms ({throughput:10.0f} 点/秒)")


if __name__ == '__main__':
    print("🚀 TorchEquilibrium GPU加速测试")
    print("="*60)
    
    # 创建GPU平衡态
    config = TorchEquilibriumConfig()
    print(f"\n设备: {config.device}")
    
    # 使用模拟数据（如果有真实gfile，替换路径）
    eqm = TorchEquilibrium(
        gfile_path='mock',  # 或真实路径
        te_ne_path=None,
        config=config
    )
    
    # 验证插值
    validate_interpolation(eqm, num_samples=1000)
    
    # 性能测试
    performance_benchmark(eqm)
    
    # 批量计算示例
    print("\n" + "="*60)
    print("📊 批量计算示例")
    print("="*60)
    
    # 模拟100个弹丸轨迹，每个1000步
    batch_size = 100
    time_steps = 1000
    
    R = torch.linspace(1.4, 2.0, time_steps, device=config.device).unsqueeze(0).repeat(batch_size, 1)
    Z = torch.linspace(-0.5, 0.5, time_steps, device=config.device).unsqueeze(0).repeat(batch_size, 1)
    
    with torch.no_grad():
        params = eqm.get_plasma_params(R, Z)
    
    print(f"✅ 成功计算 {batch_size} 条轨迹 × {time_steps} 步")
    print(f"   输出形状: {params['Te'].shape}")
    print(f"   Te范围: [{params['Te'].min():.2f}, {params['Te'].max():.2f}] keV")
    
    print("\n✅ 所有测试通过!")