import os.path
from EventBuffer import EventBuffer

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from VizDoomEnv import VizDoomEnv

from RewardShaping import RewardShaping

scenario = "my_way_home"

learning_rate = 7e-4
steps = 32
batch_size = 64
total_timesteps = int(1e7)
clip_range = 0.5
gae_lambda = 0.99

frame_skip = 4

is_gray_observation = True

CHECKPOINT_DIR = os.path.join(os.path.curdir, "model", scenario)
LOG_DIR = os.path.join(os.path.curdir, "logs", scenario)


def main():
    event_buffer = EventBuffer(7)
    reward_shaping = RewardShaping(event_buffer)

    env = make_vec_env(
        VizDoomEnv,
        n_envs=4,
        env_kwargs={
            "scenario": scenario,
            "frame_skip": frame_skip,
            "is_converting_to_gray": is_gray_observation,
            "doom_skill": 4,
            "reward_shaping": reward_shaping,
        },
    )

    callback = TrainAndLoggingCallback(
        check_freq=10000, save_path=CHECKPOINT_DIR, reward_shaping=reward_shaping
    )

    model = A2C(
        "CnnPolicy",
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=steps,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # model = PPO.load("model/deathmatch/best_model_1240000.zip")
    # model.set_env(env)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    env.close()


if __name__ == "__main__":
    main()
