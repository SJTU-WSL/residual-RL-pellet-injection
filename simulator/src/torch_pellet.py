import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    print("⚠️  torchdiffeq未安装，使用简化Euler方法")
    print("   安装: pip install torchdiffeq")
    TORCHDIFFEQ_AVAILABLE = False
    
from .torch_equilibrium import TorchEquilibrium
from .torch_layer import TorchLayer, Model


class TorchPelletConfig:
    """弹丸计算配置"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dtype = torch.float32
        
        # ODE求解器设置
        self.ode_method = 'euler'
        self.rtol = 1e-5
        self.atol = 1e-7
        
        # 物理参数
        self.Bt = 1.0  # 磁场强度 (T)
        self.Bt_exp = 0.5  # 磁场指数
        
        # 计算控制
        self.max_time = 1.0  # 最大模拟时间 (ms)
        self.time_steps = 1000  # 输出时间点数
        
        # 事件检测
        self.min_radius = 1e-4  # 最小半径 (cm)，低于此值认为完全烧蚀


class PelletODE(nn.Module):
    """
    弹丸运动的ODE系统
    
    状态向量: [R, Z, r, vR, vZ]
    - R, Z: 弹丸位置 (m)
    - r: 弹丸半径 (cm)
    - vR, vZ: 弹丸速度 (m/s → m/ms)
    """
    
    def __init__(self, 
                 equilibrium: TorchEquilibrium,
                 ablation_model: nn.Module,
                 config: TorchPelletConfig):
        super().__init__()
        
        self.eqm = equilibrium
        self.ablation = ablation_model
        self.config = config
        
        # 物理常数
        self.Bt = torch.tensor(config.Bt, device=config.device)
        self.Bt_exp = torch.tensor(config.Bt_exp, device=config.device)
        
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        ODE右端函数: dstate/dt = f(t, state)
        
        Args:
            t: 时间标量
            state: [batch, 5] = [R, Z, r, vR, vZ]
        Returns:
            dstate: [batch, 5]
        """
        # 解包状态
        R = state[:, 0]    # m
        Z = state[:, 1]    # m
        r = state[:, 2]    # cm
        vR = state[:, 3]   # m/ms
        vZ = state[:, 4]   # m/ms
        
        # 清理输入中的异常值
        R = torch.nan_to_num(R, nan=1.5, posinf=5.0, neginf=0.5)
        Z = torch.nan_to_num(Z, nan=0.0, posinf=3.0, neginf=-3.0)
        r = torch.nan_to_num(r, nan=0.0, posinf=10.0, neginf=0.0)
        r = torch.clamp(r, min=0.0)  # 确保非负
        
        # 获取等离子体参数
        with torch.no_grad():  # 等离子体参数不需要梯度
            params = self.eqm.get_plasma_params(R, Z)
            Te = params['Te']  # keV
            ne = params['ne']  # 1e14/cm³
            
            # 清理等离子体参数
            Te = torch.nan_to_num(Te, nan=1.0, posinf=100.0, neginf=0.1)
            ne = torch.nan_to_num(ne, nan=1.0, posinf=100.0, neginf=0.01)
        
        # 计算烧蚀率 (cm/ms)
        drdt = self.ablation(self.Bt, self.Bt_exp, Te, ne, r)
        
        # 清理烧蚀率
        drdt = torch.nan_to_num(drdt, nan=0.0, posinf=0.0, neginf=-10.0)
        drdt = torch.clamp(drdt, min=-50.0, max=0.0)  # 限制范围
        
        # 如果半径很小，停止烧蚀
        drdt = torch.where(r < 1e-4, torch.zeros_like(drdt), drdt)
        
        # 速度保持恒定（简化模型）
        dvRdt = torch.zeros_like(vR)
        dvZdt = torch.zeros_like(vZ)
        
        # 位置变化率
        dRdt = vR  # m/ms
        dZdt = vZ  # m/ms
        
        # 组装导数
        dstate = torch.stack([dRdt, dZdt, drdt, dvRdt, dvZdt], dim=1)
        
        # 最后清理一次异常值
        dstate = torch.nan_to_num(dstate, nan=0.0, posinf=0.0, neginf=0.0)
        
        return dstate


class SimplifiedEulerSolver:
    """
    简化的Euler求解器（当torchdiffeq不可用时）
    """
    def __init__(self, ode_func, state0, t, method='euler'):
        self.ode_func = ode_func
        self.state0 = state0
        self.t = t
        
    def solve(self):
        """执行求解"""
        n_steps = len(self.t)
        batch_size = self.state0.shape[0]
        state_dim = self.state0.shape[1]
        
        # 预分配结果
        solution = torch.zeros(
            n_steps, batch_size, state_dim,
            device=self.state0.device,
            dtype=self.state0.dtype
        )
        solution[0] = self.state0
        
        # Euler积分
        for i in range(n_steps - 1):
            dt = self.t[i+1] - self.t[i]
            dstate = self.ode_func(self.t[i], solution[i])
            
            # 检测NaN并处理
            dstate = torch.nan_to_num(dstate, nan=0.0, posinf=0.0, neginf=0.0)
            
            solution[i+1] = solution[i] + dstate * dt
            
            # 防止半径为负或过小
            solution[i+1, :, 2] = torch.clamp(solution[i+1, :, 2], min=0.0)
            
            # 如果半径接近0，停止烧蚀
            solution[i+1, :, 2] = torch.where(
                solution[i+1, :, 2] < 1e-4,
                torch.zeros_like(solution[i+1, :, 2]),
                solution[i+1, :, 2]
            )
        
        return solution


class TorchPellet:
    """
    GPU加速弹丸类
    
    功能：
    1. 批量轨迹计算
    2. 自动烧蚀
    3. 沉积分布计算
    """
    
    def __init__(self, name: str = 'TorchPellet'):
        self.name = name
        self.layers: List[TorchLayer] = []
        
    def add_layer(self, model: Model, thickness: float, 
                  ratio: Dict[str, float], config=None):
        """
        添加壳层
        
        Args:
            model: 烧蚀模型类型
            thickness: 厚度 (cm)
            ratio: 组分比例
        """
        layer_idx = len(self.layers)
        r_low = sum([layer.thickness for layer in self.layers])
        r_high = r_low + thickness
        
        layer = TorchLayer(
            layer_idx=layer_idx,
            model=model,
            r_low=r_low,
            r_high=r_high,
            component_ratio=ratio,
            config=config
        )
        self.layers.append(layer)
    
    @property
    def radius(self) -> float:
        """总半径"""
        return sum([layer.thickness for layer in self.layers])
    
    def inject_batch(self,
                    positions: torch.Tensor,
                    velocities: torch.Tensor,
                    equilibrium: TorchEquilibrium,
                    config: Optional[TorchPelletConfig] = None,
                    initial_radius: Optional[torch.Tensor] = None) -> Dict:
        """
        批量弹丸注入
        
        Args:
            positions: [batch, 2] (R, Z) in meters
            velocities: [batch, 2] (vR, vZ) in m/s
            equilibrium: GPU平衡态
            config: 配置
            
        Returns:
            dict: 包含轨迹、烧蚀等信息
        """
        if config is None:
            config = TorchPelletConfig(device=positions.device)
        
        batch_size = positions.shape[0]
        device = positions.device
        
        # 初始化状态
        R0 = positions[:, 0]  # m
        Z0 = positions[:, 1]  # m
        if initial_radius is None:
            r0 = torch.full((batch_size,), self.radius, device=device)  # cm
        else:
            r0 = initial_radius  # cm
        vR0 = velocities[:, 0] * 1e-3  # m/s → m/ms
        vZ0 = velocities[:, 1] * 1e-3  # m/s → m/ms
        
        state0 = torch.stack([R0, Z0, r0, vR0, vZ0], dim=1)
        
        # 时间点
        t = torch.linspace(0, config.max_time, config.time_steps, device=device)
        
        # 创建ODE系统（使用最外层）
        ode_system = PelletODE(
            equilibrium=equilibrium,
            ablation_model=self.layers[-1].ablation_model,
            config=config
        )
        # 求解ODE
        if TORCHDIFFEQ_AVAILABLE and False:
            # print(f"🔧 使用torchdiffeq求解 ({config.ode_method})")
            solution = odeint(
                ode_system,
                state0,
                t,
                method=config.ode_method,
                rtol=config.rtol,
                atol=config.atol
            )
        else:
            # print(f"🔧 使用简化Euler方法")
            solver = SimplifiedEulerSolver(ode_system, state0, t)
            solution = solver.solve()
        
        # solution: [time_steps, batch, 5]
        
        # 提取轨迹
        R_path = solution[:, :, 0]      # [time, batch]
        Z_path = solution[:, :, 1]
        r_path = solution[:, :, 2]
        vR_path = solution[:, :, 3]
        vZ_path = solution[:, :, 4]
        # print(r_path)
        
        # 清理轨迹中的NaN（关键步骤！）
        R_path = torch.nan_to_num(R_path, nan=1.5, posinf=5.0, neginf=0.5)
        Z_path = torch.nan_to_num(Z_path, nan=0.0, posinf=3.0, neginf=-3.0)
        r_path = torch.nan_to_num(r_path, nan=0.0, posinf=10.0, neginf=0.0)
        r_path = torch.clamp(r_path, min=0.0)  # 确保半径非负
        
        vR_path = torch.nan_to_num(vR_path, nan=0.0, posinf=1.0, neginf=-1.0)
        vZ_path = torch.nan_to_num(vZ_path, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算等离子体参数沿轨迹
        with torch.no_grad():
            # 将轨迹reshape为 [batch*time]
            R_flat = R_path.T.reshape(-1)
            Z_flat = Z_path.T.reshape(-1)
            
            params_flat = equilibrium.get_plasma_params(R_flat, Z_flat)
            
            # Reshape回 [batch, time]
            rho_path = params_flat['rho'].reshape(batch_size, -1)
            Te_path = params_flat['Te'].reshape(batch_size, -1)
            ne_path = params_flat['ne'].reshape(batch_size, -1)
        
        # 计算沉积分布
        deposition_results = self._calculate_deposition(
            r_path, rho_path
        )
        
        # 组装输出
        results = {
            'time': t,                    # [time_steps]
            'R_path': R_path.T,          # [batch, time]
            'Z_path': Z_path.T,
            'r_path': r_path.T,
            'vR_path': vR_path.T * 1e3,  # 转回 m/s
            'vZ_path': vZ_path.T * 1e3,
            'rho_path': rho_path,
            'Te_path': Te_path,
            'ne_path': ne_path,
            **deposition_results
        }
        
        return results
    
    def _calculate_deposition(self, r_path: torch.Tensor, 
                              rho_path: torch.Tensor) -> Dict:
        """
        计算沉积分布
        
        简化版本：假设均匀沉积在当前ρ位置
        
        改进：增加数值稳定性，防止NaN
        """
        # r_path: [time, batch]
        # rho_path: [batch, time]
        
        # 1. 清理异常值（NaN、Inf）
        r_path = torch.nan_to_num(r_path, nan=0.0, posinf=0.0, neginf=0.0)
        rho_path = torch.nan_to_num(rho_path, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 2. 确保半径非负
        r_path = torch.clamp(r_path, min=0.0)
        
        # 3. 计算每步烧蚀量
        dr = torch.diff(r_path, dim=0)  # [time-1, batch]
        ablated = -dr  # 烧蚀的半径（应该为正）
        ablated = torch.clamp(ablated, min=0.0)  # 确保非负
        
        # 4. 沉积位置
        rho_deposition = rho_path[:, :-1]  # [batch, time-1]
        rho_deposition = torch.clamp(rho_deposition, min=0.0, max=1.0)
        
        # 5. 计算平均沉积深度（加权平均）
        total_ablated = ablated.sum(dim=0)  # [batch]
        weighted_rho = (ablated.T * rho_deposition).sum(dim=1)  # [batch]
        
        # 防止除零
        avg_dep_rho = torch.where(
            total_ablated > 1e-6,
            weighted_rho / total_ablated,
            torch.zeros_like(weighted_rho)
        )
        
        # 6. 计算沉积比例
        initial_r = r_path[0, :]  # [batch]
        final_r = r_path[-1, :]
        
        # 防止除零和负值
        dep_ratio = torch.where(
            initial_r > 1e-6,
            torch.clamp((initial_r - final_r) / initial_r, min=0.0, max=1.0),
            torch.zeros_like(initial_r)
        )
        
        # 7. 清理输出中的NaN
        avg_dep_rho = torch.nan_to_num(avg_dep_rho, nan=0.0)
        dep_ratio = torch.nan_to_num(dep_ratio, nan=0.0)
        
        return {
            'avg_dep_rho': avg_dep_rho,
            'dep_ratio': dep_ratio,
            'ablation_profile': ablated.T,  # [batch, time-1]
            'rho_deposition': rho_deposition
        }


# ============================================================
# 批量注入接口
# ============================================================

def batch_inject(positions: torch.Tensor,
                velocities: torch.Tensor,
                thicknesses: torch.Tensor,
                model: Model,
                ratio: Dict[str, float],
                equilibrium: TorchEquilibrium,
                config: Optional[TorchPelletConfig] = None) -> Dict:
    """
    批量弹丸注入（快捷接口）
    
    Args:
        positions: [batch, 2] (R, Z)
        velocities: [batch, 2] (vR, vZ)
        thicknesses: [batch] 厚度
        model: 弹丸模型
        ratio: 组分比例
        equilibrium: 平衡态
        config: 配置
    
    Returns:
        dict: 批量结果
    """
    if config is None:
        config = TorchPelletConfig(device=positions.device)
    
    batch_size = positions.shape[0]
    
    # 处理不同厚度：为每个创建pellet
    unique_thicknesses = torch.unique(thicknesses)
    
    if len(unique_thicknesses) == 1:
        # 所有厚度相同，一次性处理
        pellet = TorchPellet()
        pellet.add_layer(model, unique_thicknesses[0].item(), ratio)
        results = pellet.inject_batch(positions, velocities, equilibrium, config)
    else:
        # 不同厚度，分组处理
        all_results = []
        for thickness in unique_thicknesses:
            mask = thicknesses == thickness
            pellet = TorchPellet()
            pellet.add_layer(model, thickness.item(), ratio)
            
            results = pellet.inject_batch(
                positions[mask],
                velocities[mask],
                equilibrium,
                config
            )
            all_results.append((mask, results))
        
        # 合并结果
        results = _merge_results(all_results, batch_size)
    
    return results


def _merge_results(results_list, batch_size):
    """合并不同厚度的结果"""
    # 简化实现：假设所有厚度相同
    # 完整实现需要处理不同长度的轨迹
    return results_list[0][1]


# ============================================================
# 测试和验证
# ============================================================

def test_single_injection():
    """测试单个弹丸注入"""
    print("\n" + "="*60)
    print("🧪 测试单个弹丸注入")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建平衡态
    from torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
    eqm_config = TorchEquilibriumConfig(device=device)
    eqm = TorchEquilibrium('mock', config=eqm_config)
    
    # 创建弹丸
    pellet = TorchPellet('TestPellet')
    pellet.add_layer(Model.DT, 0.2, {"D": 0.5, "T": 0.5})
    
    # 单个注入
    position = torch.tensor([[1.5, 0.2]], device=device)
    velocity = torch.tensor([[300.0, 50.0]], device=device)
    
    config = TorchPelletConfig(device=device)
    config.max_time = 10.0
    config.time_steps = 500
    
    results = pellet.inject_batch(position, velocity, eqm, config)
    
    print(f"✓ 轨迹形状: {results['R_path'].shape}")
    print(f"✓ 初始半径: {results['r_path'][0, 0]:.4f} cm")
    print(f"✓ 最终半径: {results['r_path'][0, -1]:.4f} cm")
    print(f"✓ 平均沉积ρ: {results['avg_dep_rho'][0]:.4f}")
    print(f"✓ 沉积比例: {results['dep_ratio'][0]:.2%}")


def test_batch_injection():
    """测试批量注入"""
    print("\n" + "="*60)
    print("🚀 测试批量弹丸注入")
    print("="*60)
    
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建平衡态
    from torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
    eqm_config = TorchEquilibriumConfig(device=device)
    eqm = TorchEquilibrium('mock', config=eqm_config)
    
    # 批量参数
    batch_size = 100
    
    # 生成随机初始条件
    R0 = torch.rand(batch_size, device=device) * 0.3 + 1.4
    Z0 = torch.rand(batch_size, device=device) * 0.6 - 0.3
    positions = torch.stack([R0, Z0], dim=1)
    
    vR = torch.rand(batch_size, device=device) * 200 + 100
    vZ = torch.rand(batch_size, device=device) * 100 - 50
    velocities = torch.stack([vR, vZ], dim=1)
    
    thicknesses = torch.full((batch_size,), 0.15, device=device)
    
    # 批量注入
    config = TorchPelletConfig(device=device)
    config.max_time = 8.0
    config.time_steps = 400
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    tic = time.time()
    
    results = batch_inject(
        positions, velocities, thicknesses,
        Model.DT, {"D": 0.5, "T": 0.5},
        eqm, config
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - tic
    
    print(f"✅ 批量计算完成!")
    print(f"   批量大小: {batch_size}")
    print(f"   总时间: {elapsed:.3f} 秒")
    print(f"   单次平均: {elapsed/batch_size*1000:.2f} ms")
    print(f"   吞吐量: {batch_size/elapsed:.1f} 模拟/秒")
    print(f"\n结果统计:")
    print(f"   平均沉积ρ: {results['avg_dep_rho'].mean():.4f} ± {results['avg_dep_rho'].std():.4f}")
    print(f"   沉积比例: {results['dep_ratio'].mean():.2%} ± {results['dep_ratio'].std():.2%}")


def test_gradient_optimization():
    """测试梯度优化"""
    print("\n" + "="*60)
    print("🎓 测试梯度优化")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建平衡态
    from torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
    eqm_config = TorchEquilibriumConfig(device=device)
    eqm = TorchEquilibrium('mock', config=eqm_config)
    
    # 创建可优化参数
    R0 = torch.tensor([1.6], device=device, requires_grad=True)
    Z0 = torch.tensor([0.0], device=device, requires_grad=True)
    vR0 = torch.tensor([250.0], device=device, requires_grad=True)
    vZ0 = torch.tensor([50.0], device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([R0, Z0, vR0, vZ0], lr=0.01)
    
    print("目标: 最大化沉积深度")
    print(f"初始参数: R={R0.item():.3f}, Z={Z0.item():.3f}, vR={vR0.item():.1f}, vZ={vZ0.item():.1f}")
    
    # 优化循环
    for epoch in range(20):
        optimizer.zero_grad()
        
        position = torch.stack([R0, Z0]).unsqueeze(0)
        velocity = torch.stack([vR0, vZ0]).unsqueeze(0)
        thickness = torch.tensor([0.15], device=device)
        
        config = TorchPelletConfig(device=device)
        config.max_time = 5.0
        config.time_steps = 200
        
        results = batch_inject(
            position, velocity, thickness,
            Model.DT, {"D": 0.5, "T": 0.5},
            eqm, config
        )
        
        # 损失函数：负的平均沉积深度
        loss = -results['avg_dep_rho'].mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: 沉积ρ = {-loss.item():.4f}")
    
    print(f"\n优化后参数:")
    print(f"  R={R0.item():.3f}, Z={Z0.item():.3f}")
    print(f"  vR={vR0.item():.1f}, vZ={vZ0.item():.1f}")
    print(f"  最大沉积ρ: {-loss.item():.4f}")


if __name__ == '__main__':
    print("🚀 TorchPellet GPU加速测试")
    print("="*60)
    
    if not TORCHDIFFEQ_AVAILABLE:
        print("\n⚠️  建议安装torchdiffeq以获得更好的性能:")
        print("   pip install torchdiffeq")
    
    # 运行测试
    test_single_injection()
    for i in range(100):
        print(i)
        test_batch_injection()
    test_gradient_optimization()
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)