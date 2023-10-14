import os.path
import time
from EventBuffer import EventBuffer

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO, A2C

from TrainModel import scenario, memory_size

from RewardShaping import RewardShaping

# MODEL_DIR = os.path.join('model', scenario, 'best_model_' + str(total_timesteps) + '.zip')
MODEL_DIR = "model/simple_deathmatch/mem_5/best_model_2500000.zip"


def main():
    model = A2C.load(MODEL_DIR)

    event_buffer = EventBuffer(7)
    reward_shaping = RewardShaping(event_buffer)

    env = VizDoomEnv(
        scenario,
        is_window_visible=True,
        doom_skill=3,
        reward_shaping=reward_shaping,
        memory_size=5,
        advanced_actions=True
    )
    env.frame_skip = 1

    for episode in track(range(101)):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        print(f"Episode: {episode}")
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, _, info = env.step(action)
            time.sleep(1.0 / (60.0 * 2))
            total_reward += reward

        print(f"{episode}. Total Reward: {total_reward}")
        print(reward_shaping.get_statistics())

    env.close()


if __name__ == "__main__":
    main()
