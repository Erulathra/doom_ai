import os
import time
from EventBuffer import EventBuffer

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO, A2C

from TrainModel import scenario, memory_size

from RewardShaping import RewardShaping

import imageio

GIF_DIR = os.path.join('./', 'giphy', scenario)
GIF_PATH = os.path.join(GIF_DIR, f'mem_{memory_size}.gif')
MODEL_DIR = "model/simple_deathmatch/mem_5/best_model_2500000.zip"

def main():
    model = A2C.load(MODEL_DIR)

    event_buffer = EventBuffer(7)
    reward_shaping = RewardShaping(event_buffer)

    env = VizDoomEnv(
        scenario,
        is_window_visible=False,
        doom_skill=2,
        reward_shaping=reward_shaping,
        memory_size=memory_size
    )
    env.frame_skip = 1

    runs = []
    rewards = []

    for _ in track(range(50)):
        images = []

        obs, _ = env.reset()
        terminated = False

        action, _ = model.predict(obs)

        reward_sum = 0

        i = 0
        while not terminated:
            if i % 4 == 0:
                action, _ = model.predict(obs)

            obs, reward, terminated, _, info = env.step(action)

            if not terminated and i % 3 == 0:
                images.append(env.game.get_state().screen_buffer)

            reward_sum += reward
            i += 1

        runs.append(images)
        rewards.append(reward_sum)

    index_max = max(range(len(rewards)), key=rewards.__getitem__)

    os.makedirs(GIF_DIR, exist_ok=True)
    imageio.mimsave(GIF_PATH, runs[index_max], fps=30)

    env.close()


if __name__ == "__main__":
    main()
