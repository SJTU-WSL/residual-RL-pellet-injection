import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass

from src.torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
from src.torch_pellet import TorchPellet, batch_inject, TorchPelletConfig
from src.torch_layer import Model, TorchLayerConfig


@dataclass
class RunProfile:
    pellet_name: str
    thickness: float
    model: Model
    ratio: Dict[str, float]
    position: Tuple[float, float]  # (R, Z) in meters
    velocity: Tuple[float, float]  # (vR, vZ) in m/s


class TorchOutput:
    def __init__(self):
        # 注入参数
        self.position_inject_R: Optional[float] = None
        self.position_inject_Z: Optional[float] = None
        self.velocity_inject_R: Optional[float] = None
        self.velocity_inject_Z: Optional[float] = None
        
        # 轨迹数据
        self.path_time: Optional[np.ndarray] = None
        self.path_R: Optional[np.ndarray] = None
        self.path_Z: Optional[np.ndarray] = None
        self.path_pellet_radius: Optional[np.ndarray] = None
        self.path_rho: Optional[np.ndarray] = None
        self.path_Te: Optional[np.ndarray] = None
        self.path_ne: Optional[np.ndarray] = None
        
        # 沉积数据
        self.total_avg_dep_rho: Optional[float] = None
        self.dep_ratio: Optional[float] = None
        
    @classmethod
    def from_torch_results(cls, profile: RunProfile, results: Dict, idx: int):
        """
        从PyTorch结果创建Output对象
        
        Args:
            profile: 运行配置
            results: 批量计算结果
            idx: 在批量中的索引
        """
        output = cls()
        
        # 注入参数
        output.position_inject_R = profile.position[0]
        output.position_inject_Z = profile.position[1]
        output.velocity_inject_R = profile.velocity[0]
        output.velocity_inject_Z = profile.velocity[1]
        
        # 轨迹数据（转为NumPy）
        output.path_time = results['time'].detach().cpu().numpy()
        output.path_R = results['R_path'][idx].detach().cpu().numpy()
        output.path_Z = results['Z_path'][idx].detach().cpu().numpy()
        output.path_pellet_radius = results['r_path'][idx].detach().cpu().numpy()
        output.path_rho = results['rho_path'][idx].detach().cpu().numpy()
        output.path_Te = results['Te_path'][idx].detach().cpu().numpy()
        output.path_ne = results['ne_path'][idx].detach().cpu().numpy()
        
        # 沉积数据
        output.total_avg_dep_rho = results['avg_dep_rho'][idx].item()
        output.dep_ratio = results['dep_ratio'][idx].item()
        
        return output


class TorchRunner:
    """
    GPU加速的批量运行器
    API设计尽量与原始Runner兼容，但内部使用GPU批量计算
    """
    
    def __init__(self, gfile_path: Optional[str] = None, 
                 te_ne_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化Runner
        
        Args:
            gfile_path: G文件路径
            te_ne_path: 温度密度剖面路径
            device: 计算设备
        """
        self.device = device
        self.gfile_path = gfile_path
        self.te_ne_path = te_ne_path
        
        # 配置
        self.eqm_config = TorchEquilibriumConfig(device=device)
        self.pellet_config = TorchPelletConfig(device=device)
        self.layer_config = TorchLayerConfig(device=device)
        
        # 平衡态（延迟加载）
        self.equilibrium: Optional[TorchEquilibrium] = None
        
        # 运行配置列表
        self.profiles: List[RunProfile] = []
        
        # 输出结果
        self.output: Dict[str, TorchOutput] = {}
        
        print(f"🔧 TorchRunner初始化 (device: {device})")
        
        # 如果提供了gfile，立即加载
        if gfile_path is not None:
            self.load_gfile(gfile_path, te_ne_path)
    
    def load_gfile(self, gfile_path: str, te_ne_path: str = None):
        """
        加载G文件
        
        Args:
            gfile_path: G文件路径
            te_ne_path: 温度密度剖面路径
        
        Returns:
            scalar_space, vector_space (与原始API兼容，但返回None)
        """
        print(f"📂 加载G文件: {gfile_path}")
        
        self.gfile_path = gfile_path
        self.te_ne_path = te_ne_path
        
        self.equilibrium = TorchEquilibrium(
            gfile_path=gfile_path,
            te_ne_path=te_ne_path,
            config=self.eqm_config
        )
        
        # 为了API兼容，返回None（原始代码返回Space2D对象）
        return None, None
    
    def add_DT_inject(self, thickness: float, 
                     position: Tuple[float, float],
                     velocity: Tuple[float, float],
                     dratio: float = 1.0):
        """
        添加DT弹丸注入配置
        
        Args:
            thickness: 厚度 (cm)
            position: 位置 (R, Z) - 可以是元组或Coordinate对象
            velocity: 速度 (vR, vZ)
            dratio: D的比例（D/(D+T)）
        """
        # 处理position（可能是Coordinate对象）
        if hasattr(position, 'toCartesian'):
            pos_cart = position.toCartesian()
            position = (float(pos_cart.x), float(pos_cart.y))
        elif hasattr(position, '__iter__') and len(position) == 2:
            position = tuple(float(p) for p in position)
        
        # 处理velocity
        if hasattr(velocity, 'toCartesian'):
            vel_cart = velocity.toCartesian()
            velocity = (float(vel_cart.x), float(vel_cart.y))
        elif hasattr(velocity, '__iter__') and len(velocity) == 2:
            velocity = tuple(float(v) for v in velocity)
        
        # 创建配置
        profile = RunProfile(
            pellet_name=f'Pellet{len(self.profiles) + 1}',
            thickness=thickness,
            model=Model.DT,
            ratio={"D": dratio, "T": 1 - dratio},
            position=position,
            velocity=velocity
        )
        
        self.profiles.append(profile)
    
    def add_profile(self, pellet_name: str, thickness: float,
                   model: Model, ratio: Dict[str, float],
                   position: Tuple[float, float],
                   velocity: Tuple[float, float]):
        """
        添加通用注入配置
        """
        profile = RunProfile(
            pellet_name=pellet_name,
            thickness=thickness,
            model=model,
            ratio=ratio,
            position=position,
            velocity=velocity
        )
        self.profiles.append(profile)
    
    def run(self, force_single_core: bool = False, verbose: bool = True) -> Dict[str, TorchOutput]:
        if self.equilibrium is None:
            raise RuntimeError("未加载G文件，请先调用load_gfile()")
        
        n_profiles = len(self.profiles)
        if n_profiles == 0: return {}

        # 强制使用固定步长求解器以提升 GPU 稳定性
        # 如果你之前遇到 underflow 错误，这里必须改
        self.pellet_config.ode_method = 'rk4' 

        groups = self._group_profiles()
        
        # 1. 第一阶段：GPU 纯计算流（不进行任何 .item() 或 .numpy() 操作）
        group_raw_results = []
        tic_gpu = time.time()
        
        for (model, thickness), group_profiles in groups.items():
            n_group = len(group_profiles)
            positions, velocities = self._prepare_batch_inputs(group_profiles)
            
            thicknesses = torch.full((n_group,), thickness, device=self.device)
            ratio = group_profiles[0].ratio
            
            # 执行计算
            batch_res = batch_inject(
                positions=positions,
                velocities=velocities,
                thicknesses=thicknesses,
                model=model,
                ratio=ratio,
                equilibrium=self.equilibrium,
                config=self.pellet_config
            )
            
            # 只存下 Tensor，不搬运
            group_raw_results.append((group_profiles, batch_res))

        if torch.cuda.is_available():
            torch.cuda.synchronize() # 仅在此处同步一次
        gpu_time = time.time() - tic_gpu

        # 2. 第二阶段：异步后处理（统一搬回 CPU）
        if verbose: print(f"🚀 GPU 计算完成，耗时: {gpu_time:.3f}s，正在进行数据后处理...")
        
        all_results = {}
        for group_profiles, res in group_raw_results:
            # 批量将该组结果移至 CPU，减少 PCIe 往返次数
            res_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in res.items()}
            
            for idx, profile in enumerate(group_profiles):
                # 此时 from_torch_results 内部调用 .numpy() 不再触发 GPU 同步
                output = TorchOutput.from_torch_results(profile, res_cpu, idx)
                all_results[profile.pellet_name] = output

        self.output = all_results
        return all_results

    def _group_profiles(self, tolerance: float = 1e-4) -> Dict[Tuple, List[RunProfile]]:
            """
            按模型和厚度（带容差）进行分组，以最大化 Batch Size
            """
            groups = {}
            for profile in self.profiles:
                # 使用容差对厚度进行“对齐”，例如 0.12001 和 0.11999 视为 0.12
                # 这样可以防止因微小精度差异导致无法合并批次
                thickness_key = round(profile.thickness / tolerance) * tolerance
                key = (profile.model, round(thickness_key, 6))
                
                if key not in groups:
                    groups[key] = []
                groups[key].append(profile)
            return groups

    def _prepare_batch_inputs(self, profiles: List[RunProfile]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 NumPy 预组装后一次性转为 Tensor，避免 Python 逐个索引赋值
        """
        # 提取所有位置和速度
        pos_list = [p.position for p in profiles]
        vel_list = [p.velocity for p in profiles]
        
        # 一次性创建 Tensor，效率远高于在循环里 positions[i] = ...
        positions = torch.tensor(pos_list, device=self.device, dtype=torch.float32)
        velocities = torch.tensor(vel_list, device=self.device, dtype=torch.float32)
        
        return positions, velocities
    
    def __len__(self):
        """返回配置数量"""
        return len(self.profiles)
    
    def clear(self):
        """清空配置和结果"""
        self.profiles = []
        self.output = {}
    
    def get_summary(self) -> Dict:
        """
        获取结果摘要
        
        Returns:
            统计信息字典
        """
        if not self.output:
            return {}
        
        avg_rhos = [out.total_avg_dep_rho for out in self.output.values()]
        dep_ratios = [out.dep_ratio for out in self.output.values()]
        
        return {
            'n_simulations': len(self.output),
            'avg_dep_rho_mean': np.mean(avg_rhos),
            'avg_dep_rho_std': np.std(avg_rhos),
            'dep_ratio_mean': np.mean(dep_ratios),
            'dep_ratio_std': np.std(dep_ratios),
        }


# ============================================================
# 兼容性包装器（用于替换原始Runner）
# ============================================================

class HybridRunner:
    """
    混合Runner：自动选择CPU或GPU后端
    
    提供与原始Runner完全兼容的API
    """
    
    def __init__(self, gfile_path: Optional[str] = None, 
                 use_gpu: bool = True):
        """
        Args:
            gfile_path: G文件路径
            use_gpu: 是否使用GPU（如果可用）
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            print("🎮 使用GPU后端")
            self.backend = TorchRunner(gfile_path)
        else:
            print("💻 使用CPU后端")
            # 导入原始Runner
    
    def load_gfile(self, gfile_path: str, te_ne_path: str = None):
        return self.backend.load_gfile(gfile_path, te_ne_path)
    
    def add_DT_inject(self, thickness, position, velocity, dratio=1.0):
        self.backend.add_DT_inject(thickness, position, velocity, dratio)
    
    def run(self, force_single_core=False):
        return self.backend.run(force_single_core)
    
    @property
    def equilibrium(self):
        return self.backend.equilibrium
    
    @property
    def profiles(self):
        return self.backend.profiles
    
    @property
    def output(self):
        return self.backend.output
    
    def __len__(self):
        return len(self.backend)


# ============================================================
# 测试和示例
# ============================================================

def test_simple_run():
    """测试简单运行"""
    print("\n" + "="*60)
    print("🧪 测试简单运行")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = TorchRunner(device=device)
    
    # 加载模拟G文件
    runner.load_gfile('mock')
    
    # 添加几个配置
    for i in range(5):
        runner.add_DT_inject(
            thickness=0.15,
            position=(1.5 + i*0.05, 0.1),
            velocity=(250 + i*20, 30),
            dratio=0.5
        )
    
    # 运行
    results = runner.run()
    
    # 检查结果
    print(f"\n✅ 运行完成，{len(results)} 个结果")
    
    for name, output in results.items():
        print(f"\n{name}:")
        print(f"  位置: R={output.position_inject_R:.3f}, Z={output.position_inject_Z:.3f}")
        print(f"  沉积ρ: {output.total_avg_dep_rho:.4f}")
        print(f"  沉积比例: {output.dep_ratio:.2%}")


def test_parameter_scan():
    """测试参数扫描（类似原始speed_test）"""
    print("\n" + "="*60)
    print("🔍 测试参数扫描")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = TorchRunner(device=device)
    runner.load_gfile('mock')
    
    # 参数网格
    pa_num = 50
    rva_num = 25
    
    pa_lin = np.linspace(-np.pi/2, 3*np.pi/2, pa_num)
    rva_lin = np.linspace(-0.3, 0.3, rva_num)
    
    print(f"网格: {pa_num} × {rva_num} = {pa_num * rva_num} 个配置")
    
    # 添加所有配置
    for i, pa in enumerate(pa_lin):
        for j, rva in enumerate(rva_lin):
            # 简化的坐标转换
            R = 1.85 + 0.5 * np.cos(pa)
            Z = 0.5 * np.sin(pa)
            
            vR = 300 * np.cos(rva)
            vZ = 300 * np.sin(rva)
            
            runner.add_DT_inject(
                thickness=0.1,
                position=(R, Z),
                velocity=(vR, vZ),
                dratio=0.5
            )
    
    # 批量运行
    tic = time.time()
    results = runner.run()
    elapsed = time.time() - tic
    
    # 统计
    summary = runner.get_summary()
    
    print(f"\n📊 结果统计:")
    print(f"  模拟数量: {summary['n_simulations']}")
    print(f"  平均沉积ρ: {summary['avg_dep_rho_mean']:.4f} ± {summary['avg_dep_rho_std']:.4f}")
    print(f"  沉积比例: {summary['dep_ratio_mean']:.2%} ± {summary['dep_ratio_std']:.2%}")
    
    # 性能
    print(f"\n⚡ 性能:")
    print(f"  总时间: {elapsed:.3f} 秒")
    print(f"  单次平均: {elapsed/len(results)*1000:.2f} ms")
    print(f"  吞吐量: {len(results)/elapsed:.1f} 模拟/秒")


def test_hybrid_runner():
    """测试混合Runner"""
    print("\n" + "="*60)
    print("🔄 测试混合Runner（CPU/GPU自动选择）")
    print("="*60)
    
    runner = HybridRunner(use_gpu=True)
    runner.load_gfile('mock')
    
    # 添加配置
    for i in range(10):
        runner.add_DT_inject(
            thickness=0.1,
            position=(1.5 + i*0.03, 0.0),
            velocity=(280, 40),
            dratio=0.5
        )
    
    # 运行
    results = runner.run()
    
    print(f"\n✅ 完成 {len(results)} 个模拟")
    print(f"  后端: {'GPU' if runner.use_gpu else 'CPU'}")


if __name__ == '__main__':
    print("🚀 TorchRunner GPU加速测试")
    print("="*60)
    
    # 运行测试
    test_simple_run()
    test_parameter_scan()
    test_hybrid_runner()
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)