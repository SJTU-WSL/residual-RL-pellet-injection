CONFIG = {
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': 'Ne',
        'Z_eff': 1.6,
    },

    'profile_conditions': {
        'Ip': 15.0e6,

        # 1) 初始温度直接设到 10 keV 级别
        'T_i': {0.0: {0.0: 10.0, 1.0: 0.2}},
        'T_e': {0.0: {0.0: 10.0, 1.0: 0.2}},

        # 2) 右边界本来就很低，可以保留；也可略提高到 0.5 避免过陡梯度
        'T_i_right_bc': 0.2,
        'T_e_right_bc': 0.2,

        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,

        # 3) 为了更“压温度”，可以把 nbar 稍微提高（更高密度 -> 更难飙温）
        #    你之前输出里 fGW 掉得很快，nbar 适当提高也能缓解“低密度高Te”趋势
        'nbar': 0.35,  # 原 0.8 -> 0.35（基线密度明显偏低）
        'n_e': {0: {0.0: 1.1, 1.0: 0.8}},  # 稍微降核心与边界归一化剖面
        'n_e_right_bc': 5.0e18,            # 右边界也降一点，避免边界把密度托住
    },

    'numerics': {
        't_final': 100,
        'fixed_dt': 0.001,
        'resistivity_multiplier': 1,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        'chi_timestep_prefactor': 30,
        'dt_reduction_factor': 3,
    },

    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
    },

    'neoclassical': {
        'bootstrap_current': {'bootstrap_multiplier': 1.0},
    },

    'sources': {
        'generic_current': {
            'fraction_of_total_current': 0.15,
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },

        'generic_particle': {
            'S_total': 0.0,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'gas_puff': {'S_total': 0.0, 'puff_decay_length': 0.3},
        'pellet': {'S_total': 0.0, 'pellet_width': 0.05, 'pellet_deposition_location': 0.8},

        # 4) 把外加热从 73 MW 明显降下来，并且不要 100% 给电子
        #    经验上要把 Te 压到 <=10 keV，这里先给 15–25 MW 的量级更合适
        'generic_heat': {
            'gaussian_location': 0.1274,
            'gaussian_width': 0.0728,
            'P_total': 2.0e7,              # 原来 7.3e7 (73 MW) -> 2.0e7 (20 MW)
            'electron_heat_fraction': 0.5, # 原来 1.0 -> 0.5（电子/离子各一半）
        },

        'fusion': {},

        'ei_exchange': {'Qei_multiplier': 1.0},
    },

    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,

        # ped 本来 4.5 keV 就 <=10，不必须改；但如果你想更“稳压”，可以降到 3
        'T_i_ped': 3.0,   # 原来 4.5
        'T_e_ped': 3.0,   # 原来 4.5

        'n_e_ped': 0.3e20,
        'rho_norm_ped_top': 0.9,
    },

    'transport': {
        'model_name': 'qlknn',

        # 5) 提高核心热输运 patch（尤其 chi_e），避免核心“过好约束”导致 Te runaway
        'apply_inner_patch': True,
        'D_e_inner': 0.25,
        'V_e_inner': 0.0,
        'chi_i_inner': 2.5,  # 原来 1.0
        'chi_e_inner': 3.5,  # 原来 1.0（重点加大电子热扩散）
        'rho_inner': 0.2,

        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.9,

        # 6) 把 chi_min 提高一点，让最低输运别太小（这对“压温度”很有效）
        'chi_min': 0.3,   # 原来 0.05
        'chi_max': 100,
        'D_e_min': 0.05,

        'DV_effective': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'avoid_big_negative_s': True,
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },

    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'n_corrector_steps': 1,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },

    'time_step_calculator': {'calculator_type': 'fixed'},
}
