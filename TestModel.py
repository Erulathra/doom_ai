import os.path
import time

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO

from TrainModel import scenario
from TrainModel import total_timesteps
from TrainModel import is_gray_observation

from RewardShaping import RewardShaping

# MODEL_DIR = os.path.join('model', scenario, 'best_model_' + str(total_timesteps) + '.zip')
MODEL_DIR = "model/deathmatch/best_model_2000000.zip"


def main():
    model = PPO.load(MODEL_DIR)

    reward_shaping = RewardShaping()

    env = VizDoomEnv(
        scenario,
        is_window_visible=True,
        is_converting_to_gray=is_gray_observation,
        doom_skill=3,
        reward_shaping=reward_shaping,
    )
    env.frame_skip = 1

    for episode in track(range(10)):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        print(f"Episode: {episode}")
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, _, info = env.step(action)
            time.sleep(1.0 / 30.0)
            total_reward += reward

            # reward_shaping.get_statistics()
            print(f"{episode}. Total Reward: {total_reward}")


    env.close()


if __name__ == "__main__":
    main()
