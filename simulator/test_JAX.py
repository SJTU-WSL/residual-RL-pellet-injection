from JAX_torax_simulator import MultiGPUTSimulator, BatchedInjectionParams
import jax, jax.numpy as jnp

# 1. 初始化 (Batch Size = 128)
batch_size = 6144

env = MultiGPUTSimulator("config/test_iter.py", total_batch_size=batch_size)
states = env.reset()

# 2. 构造并行随机动作
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

# 随机决定哪些环境触发注入 (5% 概率)
trigger_probs = jax.random.uniform(subkey, (batch_size,))
triggers = trigger_probs < 0.05

# 构造动作 Batch
actions = BatchedInjectionParams(
    triggered=triggers,
    pellet_deposition_location=jnp.ones(batch_size) * 0.95, # 所有环境位置相同
    pellet_width=jnp.ones(batch_size) * 0.05,
    S_total=jnp.ones(batch_size) * 2e20 # 假设注入量
)

for i in range(50000):
    # 3. 极速并行 Step (此时在 GPU 上并行跑 batch_size 个仿真)
    next_states, next_outputs = env.step(actions)
    print('t = ', i)

    print(f"T_e shape: {next_states.core_profiles.T_e.value.shape}")

    # 2. 查看离子密度 n_i 的形状
    print(f"n_i shape: {next_states.core_profiles.n_i.value.shape}")

    # 3. 查看时间 t 的形状
    # 预期形状: (128,)  -> 每个环境都有自己的时间
    print(f"Time t shape: {next_states.t.shape}")


# 4. 获取所有环境的插值结果
results = env.get_interpolated_profiles()
# results 中的每个数组形状都是 (batch_size, 201)