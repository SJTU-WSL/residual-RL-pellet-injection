import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

# ==========================================
# 1. Mock 类 (模拟你的物理环境)
# 实际使用时，请删除这些，导入你真实的库
# ==========================================

class MockProfileValue:
    def __init__(self, val):
        self.value = [val]

class MockCoreProfiles:
    def __init__(self):
        self.T_e = MockProfileValue(20.0)
        self.T_i = MockProfileValue(10.0)
        self.n_e = MockProfileValue(1e19)
        self.n_i = MockProfileValue(1e19)

class MockState:
    def __init__(self):
        self.core_profiles = MockCoreProfiles()

class MockOutput:
    def __init__(self):
        self.fgw_n_e_volume_avg = 0.5
        self.fgw_n_e_line_avg = 0.5
        self.Q_fusion = 1.0

class MockAction:
    def __init__(self):
        self.triggered = False
        self.S_total = 0
        self.pellet_deposition_location = 0
        self.pellet_width = 0

class MockSimulator:
    def step(self, action):
        # 模拟物理演化
        state = MockState()
        output = MockOutput()
        
        # 制造一些随机波动
        base_ne = 6e19 if not action.triggered else 8e19
        state.core_profiles.n_e.value[0] = base_ne + np.random.randn() * 1e18
        state.core_profiles.n_i.value[0] = state.core_profiles.n_e.value[0] * 0.9
        
        state.core_profiles.T_e.value[0] = 30.0 + np.random.randn() * 0.5
        state.core_profiles.T_i.value[0] = 15.0 + np.random.randn() * 0.5
        
        output.fgw_n_e_volume_avg = 0.4 + np.random.rand() * 0.1
        output.fgw_n_e_line_avg = 0.45 + np.random.rand() * 0.1
        output.Q_fusion = 2.0 + (1.0 if action.triggered else 0) + np.random.randn() * 0.1
        
        # 简单的冷却逻辑，模拟弹丸注入后的 flag 重置 (仅作演示)
        if action.triggered and np.random.rand() > 0.95:
             action.triggered = False
             
        return state, output, action.triggered

# ==========================================
# 2. 工作线程 (运行你的 while 循环)
# ==========================================

class SimulationThread(QThread):
    # 定义信号，用于将数据发送回主界面
    sig_log = pyqtSignal(str)          # 发送日志文本
    sig_data = pyqtSignal(dict)        # 发送绘图数据
    sig_finished = pyqtSignal()        # 结束信号

    def __init__(self):
        super().__init__()
        self.is_running = True
        
        # 初始化物理对象 (这里换成你真实的初始化)
        self.transport_simulator = MockSimulator()
        self.action = MockAction()
        self.t = 0

    def run(self):
        # 对应你的: while t < 8000
        while self.t < 8000 and self.is_running:
            
            # 1. 物理步进
            state, output, triggered = self.transport_simulator.step(self.action)
            self.action.triggered = triggered # 更新状态
            
            # 2. 收集数据 (用于绘图)
            # 为了界面流畅，我们每一帧都发送数据，或者降低频率
            data_packet = {
                't': self.t,
                'T_e': state.core_profiles.T_e.value[0],
                'T_i': state.core_profiles.T_i.value[0],
                'n_e': state.core_profiles.n_e.value[0],
                'n_i': state.core_profiles.n_i.value[0],
                'fgw_vol': output.fgw_n_e_volume_avg,
                'fgw_line': output.fgw_n_e_line_avg,
                'q_fusion': output.Q_fusion,
                'injection_active': 1 if self.action.triggered else 0
            }
            self.sig_data.emit(data_packet)

            # 3. 日志打印逻辑 (对应你的 print)
            # 这里的频率可以跟 print 保持一致，或者为了看清设小一点
            if self.t % 50 == 0:  # 原代码是 1000，为了演示效果我改快了
                log_msg = ""
                if self.action.triggered:
                    log_msg += '--- Injection ACTIVE ---\n'
                
                log_msg += (
                    f"time: {self.t} ms, "
                    f"fgw_avg: {output.fgw_n_e_volume_avg:.2f}, "
                    f"T_e: {state.core_profiles.T_e.value[0]:.2f}, "
                    f"n_e: {state.core_profiles.n_e.value[0]:.2e}, "
                    f"q_fusion: {output.Q_fusion:.2f}"
                )
                self.sig_log.emit(log_msg)

            # 4. 你的控制策略逻辑
            self.t += 1
            if self.t % 20 == 0 and not self.action.triggered:
                # 触发注入
                self.action.triggered = True
                self.action.S_total = np.array([np.random.rand() * 1e25 for _ in range(3)])
                self.action.pellet_deposition_location = 0.79582
                self.action.pellet_width = 0.05903
                self.sig_log.emit(f">>> Triggering Pellet at t={self.t}")

            # 模拟计算耗时 (实际物理计算自带耗时，不需要这个)
            time.sleep(0.005) 

        self.sig_finished.emit()

    def stop(self):
        self.is_running = False

# ==========================================
# 3. 主界面 GUI
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Torax Pellet Injection Controller")
        self.resize(1200, 800)

        # 数据存储用于绘图
        self.history = {
            't': [], 'T_e': [], 'T_i': [], 'n_e': [], 'n_i': [],
            'fgw_vol': [], 'q_fusion': [], 'injection': []
        }

        self.init_ui()

    def init_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：绘图区 (垂直排列3个图)
        plot_layout = QVBoxLayout()
        
        # 1. 聚变性能图 (Q & Greenwald)
        self.plot_perf = pg.PlotWidget(title="Fusion Performance (Q & Greenwald)")
        self.plot_perf.addLegend()
        self.curve_q = self.plot_perf.plot(pen=pg.mkPen('y', width=2), name="Q_fusion")
        self.curve_fgw = self.plot_perf.plot(pen=pg.mkPen('g', width=2, style=Qt.DashLine), name="f_GW")
        plot_layout.addWidget(self.plot_perf)

        # 2. 温度图 (Te & Ti)
        self.plot_temp = pg.PlotWidget(title="Temperature [keV]")
        self.plot_temp.addLegend()
        self.curve_te = self.plot_temp.plot(pen=pg.mkPen('r', width=2), name="T_e")
        self.curve_ti = self.plot_temp.plot(pen=pg.mkPen('m', width=2), name="T_i")
        plot_layout.addWidget(self.plot_temp)

        # 3. 密度图 (ne & ni)
        self.plot_dens = pg.PlotWidget(title="Density [m^-3]")
        self.plot_dens.addLegend()
        self.curve_ne = self.plot_dens.plot(pen=pg.mkPen('c', width=2), name="n_e")
        self.curve_ni = self.plot_dens.plot(pen=pg.mkPen('b', width=2), name="n_i")
        # 注入标记 (用柱状条表示)
        self.curve_inj = self.plot_dens.plot(pen=pg.mkPen('w', width=0), fillLevel=0, brush=(255, 255, 255, 50), name="Injection")
        plot_layout.addWidget(self.plot_dens)

        # 右侧：控制与日志区
        ctrl_layout = QVBoxLayout()
        
        # 控制面板
        box_ctrl = QGroupBox("Simulation Control")
        vbox_ctrl = QVBoxLayout()
        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_simulation)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False)

        vbox_ctrl.addWidget(self.btn_start)
        vbox_ctrl.addWidget(self.btn_stop)
        box_ctrl.setLayout(vbox_ctrl)
        
        # 日志面板
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        ctrl_layout.addWidget(box_ctrl)
        ctrl_layout.addWidget(QLabel("Simulation Log:"))
        ctrl_layout.addWidget(self.log_view)
        
        # 设置左右比例
        main_layout.addLayout(plot_layout, 7) # 左侧占70%
        main_layout.addLayout(ctrl_layout, 3) # 右侧占30%

    def start_simulation(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_view.clear()
        
        # 清空旧数据
        for key in self.history:
            self.history[key] = []
            
        # 启动线程
        self.thread = SimulationThread()
        self.thread.sig_data.connect(self.update_plots)
        self.thread.sig_log.connect(self.update_log)
        self.thread.sig_finished.connect(self.on_finished)
        self.thread.start()

    def stop_simulation(self):
        if self.thread:
            self.thread.stop()
            self.log_view.append(">>> Stopping simulation...")

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_view.append(">>> Simulation Finished.")

    def update_log(self, text):
        self.log_view.append(text)
        # 自动滚动到底部
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_plots(self, data):
        # 1. 追加数据
        self.history['t'].append(data['t'])
        self.history['T_e'].append(data['T_e'])
        self.history['T_i'].append(data['T_i'])
        self.history['n_e'].append(data['n_e'])
        self.history['n_i'].append(data['n_i'])
        self.history['fgw_vol'].append(data['fgw_vol'])
        self.history['q_fusion'].append(data['q_fusion'])
        
        # 注入状态可视化：如果激活，给一个很高的值用于画阴影，否则为0
        inj_val = 1.5e20 if data['injection_active'] else 0
        self.history['injection'].append(inj_val)

        # 2. 刷新曲线
        t_axis = self.history['t']
        
        self.curve_q.setData(t_axis, self.history['q_fusion'])
        self.curve_fgw.setData(t_axis, self.history['fgw_vol'])
        
        self.curve_te.setData(t_axis, self.history['T_e'])
        self.curve_ti.setData(t_axis, self.history['T_i'])
        
        self.curve_ne.setData(t_axis, self.history['n_e'])
        self.curve_ni.setData(t_axis, self.history['n_i'])
        self.curve_inj.setData(t_axis, self.history['injection'])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置黑色主题
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'd')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())