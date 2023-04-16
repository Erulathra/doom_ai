from VizDoomEnv import VizDoomEnv

from rich import print
from rich.progress import track

from matplotlib import pyplot as plt

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO

CHECKPOINT_DIR = './model/basic'
LOG_DIR = './logs/basic'


def main():
    env = VizDoomEnv()
    env.reset()

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    env.close()


if __name__ == "__main__":
    main()
