import gym
import time
import yaml

from stable_baselines3 import DQN # type: ignore
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack # type: ignore
from stable_baselines3.common.callbacks import EvalCallback

USING_FRAME_STACK = True


# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

def make_train_env():
    return Monitor(
        gym.make(
            "scripts:airsim-env-v0",
            ip_address="127.0.0.1",
            image_shape=(50, 50, 3),
            env_config=env_config["TrainEnv"]
        )
    )

def make_eval_env():
    return Monitor(
        gym.make(
            "scripts:airsim-env-v0",
            ip_address="127.0.0.1",
            image_shape=(50, 50, 3),
            env_config=env_config["TrainEnv"]
        )
    )

# VecEnv (n_envs=1)
env = DummyVecEnv([make_train_env])
eval_env = DummyVecEnv([make_eval_env])
if USING_FRAME_STACK:
    env = VecFrameStack(env, n_stack=4, channels_order="last")  # (3,50,50) -> (12,50,50)
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")  # (3,50,50) -> (12,50,50)

env =      VecTransposeImage(env)  
eval_env = VecTransposeImage(eval_env)

print("shape: ", env.observation_space)
# DQN
model = DQN(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    seed=42,
    device="cuda",
    tensorboard_log="./tb_logs/",

    buffer_size=100000,          # replay buffer size :contentReference[oaicite:5]{index=5}
    learning_starts=5000,        # 先收集一些数据再开始学 :contentReference[oaicite:6]{index=6}
    batch_size=32,                # minibatch :contentReference[oaicite:7]{index=7}
    gamma=0.99,
    train_freq=4,                 # 每 4 step 更新一次网络 :contentReference[oaicite:8]{index=8}
    gradient_steps=1,             # 每次更新做几步梯度 :contentReference[oaicite:9]{index=9}
    target_update_interval=1_000, # target network 更新间隔 :contentReference[oaicite:10]{index=10}
    exploration_fraction=0.30,    # 多久把 eps 从初值降到终值 :contentReference[oaicite:11]{index=11}
    exploration_initial_eps=1.0,  # 初始随机动作概率 :contentReference[oaicite:12]{index=12}
    exploration_final_eps=0.05,   # 最终随机动作概率 :contentReference[oaicite:13]{index=13}
    optimize_memory_usage=True,
)

# Eval callback（建议 eval_env 单独建，不要复用 train env）
eval_callback = EvalCallback(
    eval_env,
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,  # DQN 通常评估频率可以拉大点
)

log_name = "dqn_run_" + str(time.time())

model.learn(
    total_timesteps=150000,
    tb_log_name=log_name,
    callback=[eval_callback],
)

model.save("dqn_navigation_policy")
