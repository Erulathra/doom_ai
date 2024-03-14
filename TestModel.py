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

scenario = 'deathmatch'
# MODEL_DIR = "model/final/A2C_ADV/sep_buffer/adv_action/mem_1/deathmatch/best_model_600000.zip"
MODEL_DIR = "model/final/MEM_TEST/sep_buffer/adv_action/mem_10/deathmatch/best_model_600000.zip"

def main():
    model = A2C.load(MODEL_DIR)

    env = VizDoomEnv(
        scenario,
        is_window_visible=True,
        doom_skill=3,
        memory_size=10,
        advanced_actions=True,

        reward_shaping_class=ROERewardShaping,
        reward_shaping_kwargs={
            'event_buffer_class': EventBuffer,
            'event_buffer_kwargs': {'n': EVENTS_TYPES_NUMBER}
        }
    )
    env.frame_skip = 1

    episode_results = []

    for episode in range(100):
        obs, _ = env.reset()
        action, _ = model.predict(obs)

        terminated = False

        step = 0
        while not terminated:
            if step % 4 == 0:
                action, _ = model.predict(obs)

            obs, reward, terminated, _, info = env.step(action)

            step += 1
            time.sleep(1 / (30. * 2))

    env.close()


if __name__ == "__main__":
    main()
