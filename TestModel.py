import time

import numpy as np

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO

MODEL_DIR = 'model/basic/best_model_100000.zip'


def main():
    model = PPO.load(MODEL_DIR)
    env = VizDoomEnv(True)
    env.frame_skip = 1

    for episode in track(range(10)):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, _, info = env.step(action)
            time.sleep(1./30.)
            total_reward += reward

        print(f"{episode}. Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
