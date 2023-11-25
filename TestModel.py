import os.path
import time
from EventBuffer import EventBuffer
from VizDoomBotsEnv import VizDoomBotsEnv

from VizDoomEnv import VizDoomEnv
from rich import print
from rich.progress import track

from stable_baselines3 import PPO, A2C

from TrainModel import scenario, memory_size

from ROERewardShaping import ROERewardShaping, EVENTS_TYPES_NUMBER, BotsAdditionalRewardShaping

scenario = 'deathmatch_multiple_buttons'

# MODEL_DIR = os.path.join('model', scenario, 'best_model_' + str(total_timesteps) + '.zip')
MODEL_DIR = "model/deathmatch_multiple_buttons/mem_1/16_best_model_530000.zip"

def main():
    model = A2C.load(MODEL_DIR)

    env = VizDoomEnv(
        scenario,
        is_window_visible=True,
        doom_skill=5,
        memory_size=1,
        advanced_actions=True,

        reward_shaping_class=ROERewardShaping,
        reward_shaping_kwargs={
            'event_buffer_class': EventBuffer,
            'event_buffer_kwargs': {'n': EVENTS_TYPES_NUMBER},
            'additional_reward_shaping_class': BotsAdditionalRewardShaping
        },
        # n_bots=10
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
        print(env.get_statistics())

    env.close()


if __name__ == "__main__":
    main()
