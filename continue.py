import gym
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

# 1. 重新创建环境（保持和第一次训练完全一致）
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0",
        ip_address="127.0.0.1",
        image_shape=(50, 50, 3),
        env_config=env_config["TrainEnv"]
    )
)])
env = VecTransposeImage(env)
env.reset()

# 2. 从已保存的模型加载，并绑定新环境
model = PPO.load(
    "ppo_navigation_policy",   # 你之前保存的模型文件名
    env=env,
    device="cuda"
)

# 3. 重新设置 EvalCallback（可选，也可以不评估）
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
)
callbacks.append(eval_callback)

# 4. 继续训练 130k 步
log_name = "ppo_run_continue_" + str(time.time())

model.learn(
    total_timesteps=130000,      # 追加的步数
    tb_log_name=log_name,
    callback=callbacks,
    reset_num_timesteps=False    # 关键：继续之前的计数，而不是从 0 重新算
)

# 5. 覆盖保存或另存为新模型
model.save("ppo_navigation_policy_continued")
