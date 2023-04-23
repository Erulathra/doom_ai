import os.path

from VizDoomEnv import VizDoomEnv

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

scenario = 'defend_the_center'

learning_rate = 0.0001
steps = 4096
total_timesteps = 100000
clip_range = 0.2
gae_lambda = 0.95

frame_skip = 4

CHECKPOINT_DIR = os.path.join(os.path.curdir, 'model', scenario)
LOG_DIR = os.path.join(os.path.curdir, 'logs', scenario)


def main():
    env = VizDoomEnv(scenario, frame_skip=frame_skip)
    check_env(env)

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=learning_rate, n_steps=steps)
    model.clip_range = lambda _: clip_range
    # model.gae_lambda = gae_lambda
    model.learn(total_timesteps=total_timesteps, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
