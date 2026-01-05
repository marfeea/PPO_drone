# policy_run.py
import argparse
import yaml
import gym

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


CONFIG_PATH = "scripts/config.yml"
CONFIG_KEY = "TrainEnv"

IP_ADDRESS = "127.0.0.1"
IMAGE_SHAPE = (50, 50, 3)

# 训练时用的 env
TRAIN_ENV_ID_MAP = {
    "ppo": "scripts:airsim-env-v0",
    "dqn": "scripts:airsim-env-v0",
    "sac": "scripts:airsim-env-v1",
}

# 测试 env（你目前代码里明确有 test-env-v0）
# 注意：SAC 需要连续动作的 test-env-v1，否则 action space 会不匹配
TEST_ENV_ID_MAP = {
    "ppo": "scripts:test-env-v0",
    "dqn": "scripts:test-env-v0",
    "sac": "scripts:test-env-v1",  # 如果你没注册这个 env，请看下面的报错提示
}

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "dqn": DQN,
}


def make_env(env_id: str, env_config: dict, frame_stack: int) -> DummyVecEnv:
    def _make():
        return Monitor(
            gym.make(
                env_id,
                ip_address=IP_ADDRESS,
                image_shape=IMAGE_SHAPE,
                env_config=env_config[CONFIG_KEY],
            )
        )

    env = DummyVecEnv([_make])

    if frame_stack and frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="last")

    env = VecTransposeImage(env)
    return env


def run_episodes(model, env, run_item: int):
    for ep in range(run_item):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _infos = env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])

        print(f"[Episode {ep+1}/{run_item}] reward={ep_reward:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["ppo", "sac", "dqn"])
    parser.add_argument("--ckpt", required=True, help="path to saved SB3 policy (.zip or name)")
    parser.add_argument("--run_item", required=True, type=int, help="how many episodes to run")
    parser.add_argument("--frame_stack", type=int, default=0, help="0=auto (dqn->4 else->1)")
    parser.add_argument("--test_env", action="store_true", help="use test env (no reset on pass-wall, etc.)")

    args = parser.parse_args()

    with open(CONFIG_PATH, "r") as f:
        env_config = yaml.safe_load(f)

    frame_stack = args.frame_stack
    if frame_stack == 0:
        frame_stack = 4 if args.policy == "dqn" else 1

    env_map = TEST_ENV_ID_MAP if args.test_env else TRAIN_ENV_ID_MAP
    env_id = env_map[args.policy]

    if args.test_env and args.policy == "sac" and env_id == "scripts:test-env-v1":
        try:
            _ = gym.spec(env_id)
        except Exception as e:
            raise RuntimeError(
                "你开启了 --test_env 并选择了 SAC，但当前工程里似乎没有注册 scripts:test-env-v1。\n"
                "SAC 需要连续动作(Box)的测试环境，不能直接用 test-env-v0(通常是离散动作)。\n"
                "请你实现/注册一个连续动作版本的测试环境（例如继承 AirSimDroneEnvSAC），并注册为 scripts:test-env-v1。"
            ) from e

    env = make_env(env_id, env_config, frame_stack)

    Algo = ALGOS[args.policy]
    model = Algo.load(args.ckpt, env=env)

    mode = "TEST" if args.test_env else "TRAIN"
    print(f"[{mode}] Loaded {args.policy.upper()} from: {args.ckpt}")
    print(f"[{mode}] Env: {env_id}, frame_stack={frame_stack}, obs_space={env.observation_space}")

    run_episodes(model, env, args.run_item)


if __name__ == "__main__":
    main()
