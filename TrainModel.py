import os.path

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from VizDoomEnv import VizDoomEnv

from RewardShaping import RewardShaping

scenario = "deathmatch"

learning_rate = 1e-4
steps = 8 * 1024
batch_size = 64
total_timesteps = 1000000
clip_range = 0.15
gae_lambda = 0.90

frame_skip = 4

is_gray_observation = True

CHECKPOINT_DIR = os.path.join(os.path.curdir, "model", scenario)
LOG_DIR = os.path.join(os.path.curdir, "logs", scenario)


def main():
    env = VizDoomEnv(
        scenario,
        frame_skip=frame_skip,
        is_converting_to_gray=is_gray_observation,
        doom_skill=5,
        reward_shaping=RewardShaping(),
    )
    check_env(env)

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO(
        "CnnPolicy",
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=steps,
        gae_lambda=gae_lambda,
        clip_range=lambda _: clip_range,
        batch_size=batch_size,
    )

    # model = PPO.load("model/deadly_corridor/best_model_300000L4.zip")
    model.set_env(env)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
