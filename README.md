# 项目结构说明（Statement）

## 根目录
- main.py: 使用 PPO 训练离散 9 动作的导航策略，50x50x3 RGB 输入，经 VecTransposeImage 转置，TensorBoard 日志在 `./tb_logs/`，最佳模型保存在当前目录。
- continue.py: 在相同环境上从 `ppo_navigation_policy` 续训 130k 步，输出 `ppo_navigation_policy_continued.zip`。
- train_sac.py: 使用 SAC（连续 2 维动作）训练，较大 replay buffer 与训练频率配置，输出 `sac_navigation_policy`，日志同样写入 `./tb_logs/`。
- policy_run.py: 在测试环境加载 `saved_policy/ppo_navigation_policy.zip` 运行 1000 步推理。
- requirements.txt: 依赖列表；README.md、LICENSE 为项目说明与许可证。
- 检查点: `ppo_navigation_policy*.zip`、`best_model.zip`、`third_best.zip` 等模型文件；`evaluations.npz` 为评估结果。

## scripts 目录
- scripts/config.yml: 走廊 sections 配置，定义 9 个洞口位置与 x 轴偏移。
- scripts/airsim_env.py: 自定义 Gym 环境定义。
  - `AirSimDroneEnv`: 离散 9 动作，基于 RGB 的对准/穿洞奖励与碰撞惩罚。
  - `TestEnv`: 评估用，固定起点，统计前进距离和穿洞数。
  - `AirSimDroneEnvSAC`: 连续动作版本，动作空间 (vy, vz) ∈ [-1,1]^2，步长 dt=0.5。
- scripts/airsim/: AirSim Python 客户端副本（client、types、utils、pfm 等）。

## saved_policy 目录
- saved_policy/ppo_navigation_policy.zip: 预训练策略权重，policy_run.py 默认加载。

## tb_logs 目录
- tb_logs/*: TensorBoard 日志，多次 PPO/SAC 运行记录（部分子目录为中文实验名）。

## TrainEnv 目录（Unreal 环境，该项目内不包含）
- TrainEnv/WindowsNoEditor/TrainEnv.exe: UE4 打包的走廊模拟器（仅 Windows）。
- TrainEnv/WindowsNoEditor/TrainEnv/Saved/Logs: 运行日志与备份。
- TrainEnv/WindowsNoEditor/TrainEnv/Saved/Crashes: UE4 崩溃转储。
- 其余 Engine/Content/Paks 等为 UE4 运行所需资源。


## 快速运行
- 训练 PPO: `python main.py`
- 续训 PPO: `python continue.py`
- 训练 SAC: `python train_sac.py`
- 运行推理: 启动 `TrainEnv.exe` 后执行 `python policy_run.py`

## 依赖与设置
- 需要在 `Documents/AirSim/settings.json` 按 README 配置 (50x50 RGB, ClockSpeed 20, SimpleFlight)。
- 建议 Python 3.8，使用 `pip install -r requirements.txt` 安装 Stable Baselines3、Gym、AirSim 客户端等依赖。
