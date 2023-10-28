import os.path

import wakepy
from wakepy import keep

from EventBuffer import EventBuffer

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from VizDoomEnv import VizDoomEnv

from ROERewardShaping import ROERewardShaping, SimpleRewardShaping

scenario = "simple_deathmatch"
frame_skip = 4
memory_size = 1
advanced_actions = False

CHECKPOINT_DIR = os.path.join(os.path.curdir, "model", scenario, f"mem_{memory_size}")

if advanced_actions:
    advanced_actions_str = 'adv_action_space'
else:
    advanced_actions_str = 'basic_action_space'

LOG_DIR = os.path.join(os.path.curdir, "logs", "sep_buff", scenario, f"mem_{memory_size}", f"{advanced_actions_str}")


def main():
    env = make_vec_env(
        VizDoomEnv,
        n_envs=4,
        env_kwargs={
            "scenario": scenario,
            "is_window_visible": False,
            "frame_skip": frame_skip,
            "doom_skill": 3,
            "memory_size": memory_size,
            "advanced_actions": advanced_actions,

            'reward_shaping_class': ROERewardShaping,
            'reward_shaping_kwargs': {
                'event_buffer_class': EventBuffer,
                'event_buffer_kwargs': {'n': 7}
            }
        },

    )

    callback = TrainAndLoggingCallback(
        check_freq=10000, save_path=CHECKPOINT_DIR
    )

    # A2C
    model = A2C(
        "CnnPolicy",
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-5
    )

    # model = PPO.load("model/deathmatch/best_model_1240000.zip")
    # model.set_env(env)

    with keep.running() as m:
        if not m.success:
            print("Cannot prevent sleep")

        model.learn(total_timesteps=1e7, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
