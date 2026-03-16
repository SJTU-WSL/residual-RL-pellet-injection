import os
import numpy as np # pyright: ignore[reportMissingImports]
import dataclasses
import copy
from torax._src.config import config_loader, build_runtime_params # pyright: ignore[reportMissingImports]
from run_loop_sim import prepare_simulation # type: ignore

KEV_TO_JOULES = 1.60217663e-16
@dataclasses.dataclass
class InjectionParams:
    """策略模块传来的指令 (Action)"""
    triggered: bool = False
    pellet_deposition_location: float = 0.95
    pellet_width: float = 0.05
    S_total: float = 0.0 # 修改点：这是一个标量 float，代表当前 1ms 内注入的总粒子数

class TransportSimulator:
    def __init__(self, base_config_path: str):
        self.base_config = config_loader.build_torax_config_from_file(base_config_path)
        self.override_config = copy.deepcopy(self.base_config)

        self.initial_state, self.initial_outputs, self.step_fn = prepare_simulation(self.base_config)
        
        self.current_state = self.initial_state
        self.last_outputs = self.initial_outputs
        self.step_count = 0
        
        # 移除游标状态，因为现在是瞬时注入
        self._active_params = None   

    def reset(self):
        self.current_state = self.initial_state
        self.last_outputs = self.initial_outputs
        self.step_count = 0
        self._active_params = None
        return self.current_state

    def step(self, action: InjectionParams):
        """
        执行一步仿真。
        如果 action.triggered 为 True，则在本步立即一次性注入 S_total，
        并在下一步自动关闭触发器（瞬时注入）。
        """
        # 默认不注入
        current_override = None
        
        # 只要触发，就立即构建 override 参数
        if action.triggered:
            current_S_value = action.S_total 

            target_width = self.override_config.sources.pellet.pellet_width
            object.__setattr__(target_width, "value", np.array([action.pellet_width]))
            
            target_loc = self.override_config.sources.pellet.pellet_deposition_location
            object.__setattr__(target_loc, "value", np.array([action.pellet_deposition_location]))
            
            target_S = self.override_config.sources.pellet.S_total
            object.__setattr__(target_S, "value", np.array([current_S_value]))
            
            current_override = build_runtime_params.RuntimeParamsProvider.from_config(self.override_config)
            
            next_trigger = False
        else:
            target_S = self.override_config.sources.pellet.S_total
            object.__setattr__(target_S, "value", np.array([0.0]))
            current_override = build_runtime_params.RuntimeParamsProvider.from_config(self.override_config)
            next_trigger = False

        next_state, next_outputs = self.step_fn(
            self.current_state,
            self.last_outputs,
            runtime_params_overrides=current_override
        )

        self.current_state = next_state
        self.last_outputs = next_outputs
        self.step_count += 1

        return next_state, next_outputs, next_trigger

    def interpolate_profiles(self, state):
        rho_old = np.linspace(0, 1, 25) 
        rho_new = np.linspace(0, 1, 201)

        T_e_old = np.array(state.core_profiles.T_e.value).flatten()
        n_e_old = np.array(state.core_profiles.n_e.value).flatten()
        T_i_old = np.array(state.core_profiles.T_i.value).flatten()
        n_i_old = np.array(state.core_profiles.n_i.value).flatten()

        T_e_new = np.interp(rho_new, rho_old, T_e_old)
        n_e_new = np.interp(rho_new, rho_old, n_e_old)
        T_i_new = np.interp(rho_new, rho_old, T_i_old)
        n_i_new = np.interp(rho_new, rho_old, n_i_old)

        P_e_new = n_e_new * T_e_new * KEV_TO_JOULES
        P_i_new = n_i_new * T_i_new * KEV_TO_JOULES

        return T_e_new * 1000, n_e_new, P_e_new, T_i_new * 1000, n_i_new, P_i_new

    def calculate_species_profiles(self, n_i_total, T_i_total):
        T_D = T_i_total.copy()
        T_T = T_i_total.copy()
        
        n_D = n_i_total * 0.5
        n_T = n_i_total * 0.5
        
        P_D = n_D * T_D * KEV_TO_JOULES
        P_T = n_T * T_T * KEV_TO_JOULES
        
        return (n_D, T_D, P_D), (n_T, T_T, P_T)
