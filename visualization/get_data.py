from simulator.torax_simulator import TransportSimulator, InjectionParams
from simulator.PAM_simulator import PelletSimulator
import pandas as pd
import numpy as np

transport_simulator = TransportSimulator('config/test_iter.py')
pellet_sim = PelletSimulator(velocity=400.0, thickness=0.004, size=0.004)

action = InjectionParams()
t = 0
def get_vector(val):
    if val is None: return np.zeros(25) # 占位
    # 如果是 CellVariable (有 value 属性)，取 value
    if hasattr(val, 'value'):
        return np.array(val.value)
    # 否则直接转 numpy
    return np.array(val)

def get_scalar(val):
    if val is None: return 0.0
    # 如果是 JAX 数组，转为 float；如果是 python float，保持不变
    return val.item() if hasattr(val, 'item') else val

simulation_history = []
post_processing_history = []
# Initial ramp-up phase 
while t < 1000:
    state, output, action.triggered = transport_simulator.step(action)
    t += 1

    row_data = {
        't': t,

        'pellet': False,
        'Q_fusion': get_scalar(output.Q_fusion),
        'H98': get_scalar(output.H98),
        'W_thermal_total': get_scalar(output.W_thermal_total),
        'q95': get_scalar(output.q95),
        'q_min': get_scalar(output.q_min),
        'fgw_n_e_volume_avg': get_scalar(output.fgw_n_e_volume_avg),
        'fgw_n_e_line_avg': get_scalar(output.fgw_n_e_line_avg),


        'q_face': get_vector(state.core_profiles.q_face),
        's_face': get_vector(state.core_profiles.s_face),
        'T_e': get_vector(state.core_profiles.T_e),
        'n_e': get_vector(state.core_profiles.n_e),
        'T_i': get_vector(state.core_profiles.T_i),
        'n_i': get_vector(state.core_profiles.n_i)
    }
    simulation_history.append(row_data)
    post_processing_history.append(output)


# Main simulation loop with pellet injections every 300 ms
while t < 20000:
    state, output, action.triggered = transport_simulator.step(action)
    
    if t % 100 == 0:
        print('time:', t, 'ms', end=', ')
        print('fgw_n_e_volume_avg:{:.2f}'.format(output.fgw_n_e_volume_avg), end=', ')
        print('fgw_n_e_line_avg:{:.2f}'.format(output.fgw_n_e_line_avg), end=', ')
        print('T_e[0]:{:.2f}'.format(state.core_profiles.T_e.value[0]), end=', ')
        print('T_i[0]:{:.2f}'.format (state.core_profiles.T_i.value[0]), end=', ')
        print('n_e[0]:{:.2e}'.format(state.core_profiles.n_e.value[0]), end=', ')
        print('n_i[0]:{:.2e}'.format(state.core_profiles.n_i.value[0]), end=', ')
        print('q_fusion:{:.2f}'.format(output.Q_fusion))

    t += 1
    
    if np.random.rand() < 0.005:
        T_e, n_e, P_e, T_i, n_i, P_i = transport_simulator.interpolate_profiles(state)
        species_D, species_T = transport_simulator.calculate_species_profiles(n_i, T_i)
    
        pellet_sim.update_plasma_state(
            T_e, n_e, P_e, T_i, n_i, P_i, 
            species_D, species_T
        )
        
        loc, width, rate = pellet_sim.simulate_pellet_injection(
            velocity=300.0, 
            size=0.002,
            thickness=0.002
        )
        
        action.triggered = True
        action.pellet_deposition_location = loc
        action.pellet_width = width
        action.S_total = rate

        print(f"--- Triggering Injection at t={t} ---> [{loc:.4f}, {width:.4f}, {rate:.3e}]")

    row_data = {
        't': t,

        'pellet': action.triggered,
        'Q_fusion': get_scalar(output.Q_fusion),
        'H98': get_scalar(output.H98),
        'W_thermal_total': get_scalar(output.W_thermal_total),
        'q95': get_scalar(output.q95),
        'q_min': get_scalar(output.q_min),
        'fgw_n_e_volume_avg': get_scalar(output.fgw_n_e_volume_avg),
        'fgw_n_e_line_avg': get_scalar(output.fgw_n_e_line_avg),


        'q_face': get_vector(state.core_profiles.q_face),
        's_face': get_vector(state.core_profiles.s_face),
        'T_e': get_vector(state.core_profiles.T_e),
        'n_e': get_vector(state.core_profiles.n_e),
        'T_i': get_vector(state.core_profiles.T_i),
        'n_i': get_vector(state.core_profiles.n_i)
    }
    # print(row_data['pellet'])
    simulation_history.append(row_data)
    post_processing_history.append(output)

runtime_data = pd.DataFrame(simulation_history)
runtime_data.to_pickle("simulation data/simulation_results.pkl")
print("Data saved to simulation_results.pkl (Recommended)")

csv_df = runtime_data.copy()
vector_cols = ['q_face', 's_face', 'T_e', 'n_e']
for col in vector_cols:
    csv_df[col] = csv_df[col].apply(lambda x: x.tolist())

csv_df.to_csv("simulation data/simulation_results.csv", index=False)
print("Data saved to simulation_results.csv")
