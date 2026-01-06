import gym
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

USING_FRAME_STACK = True
N_STACK = 4

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

def make_train_env():
    return Monitor(
        gym.make(
            "scripts:airsim-env-v0",
            ip_address="127.0.0.1",
            image_shape=(50, 50, 3),          # HWC
            env_config=env_config["TrainEnv"]
        )
    )

def make_eval_env():
    return Monitor(
        gym.make(
            "scripts:airsim-env-v0",
            ip_address="127.0.0.1",
            image_shape=(50, 50, 3),          # HWC
            env_config=env_config["TrainEnv"]
        )
    )

env = DummyVecEnv([make_train_env])
eval_env = DummyVecEnv([make_eval_env])

if USING_FRAME_STACK:
    env = VecFrameStack(env, n_stack=N_STACK, channels_order="last")
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")

env = VecTransposeImage(env)
eval_env = VecTransposeImage(eval_env)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    seed=42,
    device="cuda",
    tensorboard_log="./tb_logs/",
    # 可选：如果你 env 输出已经是 float 且范围 0~1，可能需要加：policy_kwargs=dict(normalize_images=False)
)

eval_callback = EvalCallback(
    eval_env,              
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
)

log_name = "ppo_run_" + str(time.time())

model.learn(
    total_timesteps=150000,
    tb_log_name=log_name,
    callback=[eval_callback],
)

model.save("ppo_frameStack")
