import os.path
import time

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO

from TrainModel import scenario
from TrainModel import total_timesteps

MODEL_DIR = os.path.join('model', scenario, 'best_model_' + str(total_timesteps) + '.zip')


def main():
    model = PPO.load(MODEL_DIR)
    env = VizDoomEnv(scenario, is_window_visible=True)
    env.frame_skip = 1

    for episode in track(range(10)):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, _, info = env.step(action)
            time.sleep(1. / 30.)
            total_reward += reward

        print(f"{episode}. Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
