from VizDoomEnv import VizDoomEnv

from rich import print
from rich.progress import track

from matplotlib import pyplot as plt

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO

CHECKPOINT_DIR = './model/basic'
LOG_DIR = './logs/basic'

learning_rate = 0.0001
steps = 256


def main():
    env = VizDoomEnv()
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=learning_rate, n_steps=steps)
    model.learn(total_timesteps=100000, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
