import os.path
from EventBuffer import EventBuffer

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from VizDoomEnv import VizDoomEnv

from RewardShaping import RewardShaping, SimpleRewardShaping

scenario = "deadly_corridor"
frame_skip = 4
memory_size = 1
advanced_actions = False

CHECKPOINT_DIR = os.path.join(os.path.curdir, "model", scenario, f"mem_{memory_size}")

if advanced_actions:
    advanced_actions_str = 'adv_action_space'
else:
    advanced_actions_str = 'basic_action_space'

LOG_DIR = os.path.join(os.path.curdir, "logs", "debug", scenario, f"mem_{memory_size}", f"{advanced_actions_str}")


def main():
    event_buffer = EventBuffer(7)
    reward_shaping = RewardShaping(event_buffer)
    # reward_shaping = SimpleRewardShaping(event_buffer)

    env = make_vec_env(
        VizDoomEnv,
        n_envs=4,
        env_kwargs={
            "scenario": scenario,
            "is_window_visible": False,
            "frame_skip": frame_skip,
            "doom_skill": 4,
            "reward_shaping": reward_shaping,
            "memory_size": memory_size,
            "advanced_actions": advanced_actions
        },
    )

    callback = TrainAndLoggingCallback(
        check_freq=10000, save_path=CHECKPOINT_DIR, reward_shaping=reward_shaping
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
        # use_rms_prop=True,
        # rms_prop_eps=1e-5
    )

    # model = PPO.load("model/deathmatch/best_model_1240000.zip")
    # model.set_env(env)

    model.learn(total_timesteps=1e7, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
