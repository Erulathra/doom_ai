import os
from EventBuffer import EventBuffer
from VizDoomEnv import VizDoomEnv
from rich.progress import track
from stable_baselines3 import PPO, A2C
from TrainModel import scenario, memory_size
from ROERewardShaping import ROERewardShaping, EVENTS_TYPES_NUMBER
import imageio

scenario = 'deathmatch_multiple_buttons'

GIF_DIR = os.path.join('./', 'giphy', scenario)
GIF_PATH = os.path.join(GIF_DIR, f'ROE_16_mem_{memory_size}_3.gif')
MODEL_DIR = "model/deathmatch_multiple_buttons/mem_1/16_best_model_530000.zip"


def main():
    model = A2C.load(MODEL_DIR)

    env = VizDoomEnv(
        scenario,
        is_window_visible=False,
        doom_skill=1,
        memory_size=1,
        advanced_actions=True,

        reward_shaping_class=ROERewardShaping,
        reward_shaping_kwargs={
            'event_buffer_class': EventBuffer,
            'event_buffer_kwargs': {'n': EVENTS_TYPES_NUMBER}
        }
    )
    env.frame_skip = 1

    runs = []
    rewards = []

    for _ in track(range(1)):
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

            if not terminated and i % 2 == 0:
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
