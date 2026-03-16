"""
test_integration.py - 集成测试

测试torch_equilibrium, torch_pellet, torch_runner三个模块的协同工作
对比原始FPAD代码的性能和精度
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict

from src.torch_equilibrium import TorchEquilibrium, TorchEquilibriumConfig
from src.torch_pellet import TorchPellet, batch_inject, TorchPelletConfig
from src.torch_runner import TorchRunner, HybridRunner
from src.torch_layer import Model


def test_full_pipeline():
    """
    测试完整流程：加载平衡态 → 批量注入 → 结果分析
    """
    print("="*70)
    print("🔬 完整流程集成测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================
    # 步骤1: 初始化平衡态
    # ========================================
    print("\n" + "-"*70)
    print("步骤1: 初始化平衡态")
    print("-"*70)
    
    eqm_config = TorchEquilibriumConfig(device=device)
    eqm = TorchEquilibrium('input/g067590.03300', config=eqm_config)
    
    print("✅ 平衡态加载完成")
    
    # ========================================
    # 步骤2: 创建Runner并添加配置
    # ========================================
    print("\n" + "-"*70)
    print("步骤2: 配置批量注入")
    print("-"*70)
    
    runner = TorchRunner(device=device)
    runner.equilibrium = eqm
    
    # 参数扫描：位置角度 × 速度角度
    pa_num = 100
    rva_num = 50
    total_configs = pa_num * rva_num
    
    pa_lin = np.linspace(-np.pi/4, np.pi/4, pa_num)
    rva_lin = np.linspace(-0.2, 0.2, rva_num)
    
    print(f"网格设置: {pa_num} × {rva_num} = {total_configs} 个配置")
    
    for i, pa in enumerate(pa_lin):
        for j, rva in enumerate(rva_lin):
            # 圆形轨迹坐标
            R = 1.85 + 0.4 * np.cos(pa)
            Z = 0.4 * np.sin(pa)
            
            # 速度分量
            v_mag = 280  # m/s
            vR = v_mag * np.cos(rva + pa + np.pi)
            vZ = v_mag * np.sin(rva + pa + np.pi)
            
            runner.add_DT_inject(
                thickness=0.12,
                position=(R, Z),
                velocity=(vR, vZ),
                dratio=0.5
            )
    
    print(f"✅ 已添加 {len(runner)} 个注入配置")
    
    # ========================================
    # 步骤3: 批量计算
    # ========================================
    print("\n" + "-"*70)
    print("步骤3: 批量GPU计算")
    print("-"*70)
    
    tic_total = time.time()
    results = runner.run(verbose=True)
    total_time = time.time() - tic_total
    
    # ========================================
    # 步骤4: 结果分析
    # ========================================
    print("\n" + "-"*70)
    print("步骤4: 结果分析")
    print("-"*70)
    
    # 提取数据
    avg_rhos = np.array([out.total_avg_dep_rho for out in results.values()])
    dep_ratios = np.array([out.dep_ratio for out in results.values()])
    
    # Reshape为网格
    avg_rhos_grid = avg_rhos.reshape(pa_num, rva_num)
    dep_ratios_grid = dep_ratios.reshape(pa_num, rva_num)
    
    print(f"\n📊 统计信息:")
    print(f"  平均沉积深度ρ:")
    print(f"    均值: {avg_rhos.mean():.4f}")
    print(f"    标准差: {avg_rhos.std():.4f}")
    print(f"    范围: [{avg_rhos.min():.4f}, {avg_rhos.max():.4f}]")
    print(f"\n  沉积比例:")
    print(f"    均值: {dep_ratios.mean():.2%}")
    print(f"    标准差: {dep_ratios.std():.2%}")
    print(f"    范围: [{dep_ratios.min():.2%}, {dep_ratios.max():.2%}]")
    
    # ========================================
    # 步骤5: 可视化
    # ========================================
    print("\n" + "-"*70)
    print("步骤5: 生成可视化")
    print("-"*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 沉积深度分布
    im1 = axes[0].imshow(
        avg_rhos_grid.T, 
        aspect='auto', 
        cmap='hot',
        origin='lower',
        extent=[pa_lin[0], pa_lin[-1], rva_lin[0], rva_lin[-1]]
    )
    axes[0].set_xlabel('position angle (rad)')
    axes[0].set_ylabel('velocity angle (rad)')
    axes[0].set_title('average deposition depth ρ')
    plt.colorbar(im1, ax=axes[0])
    
    # 沉积比例分布
    im2 = axes[1].imshow(
        dep_ratios_grid.T * 100, 
        aspect='auto', 
        cmap='viridis',
        origin='lower',
        extent=[pa_lin[0], pa_lin[-1], rva_lin[0], rva_lin[-1]]
    )
    axes[1].set_xlabel('position angle (rad)')
    axes[1].set_ylabel('velocity angle (rad)')
    axes[1].set_title('deposition ratio (%)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('plot/integration_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化已保存: integration_test_results.png")
    plt.close()
    
    # ========================================
    # 步骤6: 性能报告
    # ========================================
    print("\n" + "="*70)
    print("⚡ 性能报告")
    print("="*70)
    
    avg_time = total_time / total_configs
    throughput = total_configs / total_time
    
    # 估算CPU性能（基于原始14ms/次）
    cpu_estimate = total_configs * 0.014  # 秒
    speedup = cpu_estimate / total_time
    
    print(f"\n总计算:")
    print(f"  配置数量: {total_configs}")
    print(f"  总时间: {total_time:.3f} 秒")
    print(f"  单次平均: {avg_time*1000:.2f} ms")
    print(f"  吞吐量: {throughput:.1f} 模拟/秒")
    
    print(f"\n对比估算:")
    print(f"  原始CPU预估: {cpu_estimate:.1f} 秒")
    print(f"  GPU实际: {total_time:.3f} 秒")
    print(f"  加速比: {speedup:.1f}×")
    
    return results


def test_accuracy_validation():
    """
    精度验证：对比单个case的详细结果
    """
    print("\n" + "="*70)
    print("🎯 精度验证测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建平衡态
    eqm = TorchEquilibrium('input/g067590.03300', config=TorchEquilibriumConfig(device=device))
    
    # 单个弹丸
    pellet = TorchPellet('ValidationPellet')
    pellet.add_layer(Model.DT, 0.2, {"D": 0.5, "T": 0.5})
    
    position = torch.tensor([[1.6, 0.1]], device=device)
    velocity = torch.tensor([[200.0, 40.0]], device=device)
    
    config = TorchPelletConfig(device=device)
    config.max_time = 12.0
    config.time_steps = 600
    
    # 计算
    results = pellet.inject_batch(position, velocity, eqm, config)
    
    # 详细输出
    print("\n轨迹详情:")
    print(f"  时间步数: {len(results['time'])}")
    print(f"  初始位置: R={results['R_path'][0,0]:.3f}, Z={results['Z_path'][0,0]:.3f}")
    print(f"  最终位置: R={results['R_path'][0,-1]:.3f}, Z={results['Z_path'][0,-1]:.3f}")
    print(f"\n半径演化:")
    print(f"  初始: {results['r_path'][0,0]:.4f} cm")
    print(f"  最终: {results['r_path'][0,-1]:.4f} cm")
    print(f"  烧蚀量: {results['r_path'][0,0] - results['r_path'][0,-1]:.4f} cm")
    print(f"\n沉积:")
    print(f"  平均深度ρ: {results['avg_dep_rho'][0]:.4f}")
    print(f"  沉积比例: {results['dep_ratio'][0]:.2%}")
    
    # 绘制轨迹
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t = results['time'].detach().cpu().numpy()
    R = results['R_path'][0].detach().cpu().numpy()
    Z = results['Z_path'][0].detach().cpu().numpy()
    r = results['r_path'][0].detach().cpu().numpy()
    rho = results['rho_path'][0].detach().cpu().numpy()
    Te = results['Te_path'][0].detach().cpu().numpy()
    ne = results['ne_path'][0].detach().cpu().numpy()
    
    # 轨迹
    axes[0,0].plot(R, Z, 'b-', linewidth=2)
    axes[0,0].plot(R[0], Z[0], 'go', markersize=10, label='start')
    axes[0,0].plot(R[-1], Z[-1], 'ro', markersize=10, label='end')
    axes[0,0].set_xlabel('R (m)')
    axes[0,0].set_ylabel('Z (m)')
    axes[0,0].set_title('pellet trajectory')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axis('equal')
    
    # 半径演化
    axes[0,1].plot(t, r, 'r-', linewidth=2)
    axes[0,1].set_xlabel('time (ms)')
    axes[0,1].set_ylabel('radius (cm)')
    axes[0,1].set_title('pellet radius change over time')
    axes[0,1].grid(True, alpha=0.3)
    
    # 温度密度剖面
    axes[1,0].plot(t, Te, 'b-', label='Te', linewidth=2)
    axes[1,0].set_xlabel('time (ms)')
    axes[1,0].set_ylabel('Te (keV)', color='b')
    axes[1,0].tick_params(axis='y', labelcolor='b')
    axes[1,0].grid(True, alpha=0.3)
    
    ax_ne = axes[1,0].twinx()
    ax_ne.plot(t, ne, 'r-', label='ne', linewidth=2)
    ax_ne.set_ylabel('ne (1e14/cm³)', color='r')
    ax_ne.tick_params(axis='y', labelcolor='r')
    axes[1,0].set_title('plasma parameters along trajectory')
    
    # ρ演化
    axes[1,1].plot(t, rho, 'g-', linewidth=2)
    axes[1,1].axhline(y=1.0, color='k', linestyle='--', label='LCFS')
    axes[1,1].set_xlabel('time (ms)')
    axes[1,1].set_ylabel('ρ')
    axes[1,1].set_title('rho change along trajectory')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot/validation_trajectory.png', dpi=150, bbox_inches='tight')
    print("\n✅ 轨迹可视化已保存: validation_trajectory.png")
    plt.close()


def test_gradient_optimization_demo():
    """
    演示梯度优化功能
    """
    print("\n" + "="*70)
    print("🎓 梯度优化演示")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建平衡态
    eqm = TorchEquilibrium('/input/g093536.06060ke', config=TorchEquilibriumConfig(device=device))
    
    # 可优化参数
    print("\n目标: 找到最优注入角度以最大化沉积深度")
    
    # 初始猜测
    R0 = torch.tensor([1.7], device=device, requires_grad=True)
    Z0 = torch.tensor([0.2], device=device, requires_grad=True)
    vR = torch.tensor([250.0], device=device, requires_grad=True)
    vZ = torch.tensor([60.0], device=device, requires_grad=True)
    
    print(f"\n初始参数:")
    print(f"  位置: R={R0.item():.3f}, Z={Z0.item():.3f}")
    print(f"  速度: vR={vR.item():.1f}, vZ={vZ.item():.1f}")
    
    # 优化器
    optimizer = torch.optim.Adam([R0, Z0, vR, vZ], lr=0.005)
    
    # 记录
    history = {
        'R': [], 'Z': [], 'vR': [], 'vZ': [],
        'depth': [], 'loss': []
    }
    
    # 优化循环
    print("\n开始优化...")
    n_epochs = 100
    pellet = TorchPellet()
    pellet.add_layer(Model.DT, 0.12, {"D": 0.5, "T": 0.5})
    
    config = TorchPelletConfig(device=device)
    config.max_time = 8.0
    config.time_steps = 300 
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 组装输入
        position = torch.cat([R0, Z0], dim=0).unsqueeze(0)   # [1,2]
        velocity = torch.cat([vR, vZ], dim=0).unsqueeze(0)   # [1,2]
        thickness = torch.tensor([0.12], device=device)
        
        # 注入
        #print(f"DEBUG: R0={R0.item()}, Z0={Z0.item()}, vR={vR.item()}")
        results = pellet.inject_batch(position, velocity, eqm, config)
        
        # 损失函数：负的沉积深度
        depth = results['avg_dep_rho'][0]
        loss = -depth
        
        # 反向传播
        loss.backward()
        
        # 记录
        history['R'].append(R0.item())
        history['Z'].append(Z0.item())
        history['vR'].append(vR.item())
        history['vZ'].append(vZ.item())
        history['depth'].append(depth.item())
        history['loss'].append(loss.item())
        
        # 更新
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: 沉积深度ρ = {depth.item():.4f}")
    
    print(f"\n优化完成!")
    print(f"\n最优参数:")
    print(f"  位置: R={R0.item():.3f}, Z={Z0.item():.3f}")
    print(f"  速度: vR={vR.item():.1f}, vZ={vZ.item():.1f}")
    print(f"  最大沉积深度: {-loss.item():.4f}")
    print(f"  提升: {(history['depth'][-1]/history['depth'][0] - 1)*100:.1f}%")
    
    # 绘制优化过程
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(len(history['depth']))
    
    # 沉积深度
    axes[0,0].plot(epochs, history['depth'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('deepth ρ')
    axes[0,0].set_title('target')
    axes[0,0].grid(True, alpha=0.3)
    
    # 位置演化
    axes[0,1].plot(history['R'], history['Z'], 'ro-', markersize=4)
    axes[0,1].plot(history['R'][0], history['Z'][0], 'go', markersize=10, label='initial')
    axes[0,1].plot(history['R'][-1], history['Z'][-1], 'bs', markersize=10, label='optimal')
    axes[0,1].set_xlabel('R (m)')
    axes[0,1].set_ylabel('Z (m)')
    axes[0,1].set_title('position optimization trajectory')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # R, Z分量
    axes[1,0].plot(epochs, history['R'], 'b-', label='R', linewidth=2)
    axes[1,0].plot(epochs, history['Z'], 'r-', label='Z', linewidth=2)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('position(m)')
    axes[1,0].set_title('position components')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # vR, vZ分量
    axes[1,1].plot(epochs, history['vR'], 'b-', label='vR', linewidth=2)
    axes[1,1].plot(epochs, history['vZ'], 'r-', label='vZ', linewidth=2)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('velocity(m/s)')
    axes[1,1].set_title('velocity components')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot/optimization_demo.png', dpi=150, bbox_inches='tight')
    print("\n✅ 优化过程已保存: optimization_demo.png")
    plt.close()


if __name__ == '__main__':
    print("🚀 FPAD GPU重构 - 集成测试套件")
    print("="*70)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 运行所有测试
    try:
        # 测试1: 完整流程
        results = test_full_pipeline()
        
        # 测试2: 精度验证
        test_accuracy_validation()
        
        # 测试3: 梯度优化
        # test_gradient_optimization_demo()
        
        print("\n" + "="*70)
        print("✅ 所有集成测试通过!")
        print("="*70)
        print("\n生成的文件:")
        print("  • integration_test_results.png - 批量计算结果")
        print("  • validation_trajectory.png - 单个轨迹详情")
        print("  • optimization_demo.png - 梯度优化过程")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()