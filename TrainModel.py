import os.path

from wakepy import keep

from EventBuffer import EventBuffer

from TrainAndLoggingCallback import TrainAndLoggingCallback

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from VizDoomBotsEnv import VizDoomBotsEnv
from VizDoomEnv import VizDoomEnv

from ROERewardShaping import ROERewardShaping, SimpleRewardShaping, EVENTS_TYPES_NUMBER, BotsAdditionalRewardShaping

from stable_baselines3.common.logger import configure

scenario = "simple_deathmatch"
frame_skip = 4
memory_size = 1
advanced_actions = True


if advanced_actions:
    advanced_actions_str = 'adv_action'
else:
    advanced_actions_str = 'basic_action'

buffer_str = 'sep_buffer'

static_additional_reward = 0.


def main():
    # scenarios_to_learn = ['health_gathering', 'health_gathering_supreme', 'my_way_home']
    scenarios_to_learn = ['simple_deathmatch', 'deathmatch']

    for scenario_name in scenarios_to_learn:
    # if True:
    #     scenario_name =
        PATH = os.path.join("final", 'MEM_TEST', buffer_str,
                            f"{advanced_actions_str}", f"mem_{memory_size}",
                            scenario_name)

        LOG_DIR = os.path.join(os.path.curdir, 'logs', PATH)

        CHECKPOINT_DIR = os.path.join(os.path.curdir, "model", PATH)

        n_envs = 4
        if scenario_name == 'deathmatch':
            n_envs = 16

        env = make_vec_env(
            VizDoomEnv,
            n_envs=n_envs,
            env_kwargs={
                "scenario": scenario_name,
                "is_window_visible": False,
                "frame_skip": frame_skip,
                "doom_skill": 3,
                "memory_size": memory_size,
                "advanced_actions": advanced_actions,

                'reward_shaping_class': ROERewardShaping,
                'reward_shaping_kwargs': {
                    'event_buffer_class': EventBuffer,
                    'event_buffer_kwargs': {'n': EVENTS_TYPES_NUMBER},
                    'additional_reward_shaping_class': None
                },
            },

        )

        callback = TrainAndLoggingCallback(
            check_freq=100000, save_path=CHECKPOINT_DIR
        )

        # A2C
        model = A2C(
            "CnnPolicy",
            env,
            # tensorboard_log=LOG_DIR,
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

        # model = PPO(
        #     "CnnPolicy",
        #     env,
        #     tensorboard_log=LOG_DIR,
        #     verbose=1,
        #     learning_rate=7e-4,
        #     n_steps=4096,
        #     gae_lambda=0.95,
        #     clip_range=0.1,
        #     batch_size=64,
        # )

        train_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
        model.set_logger(train_logger)

        with keep.running() as m:
            if not m.success:
                print("Cannot prevent sleep")

            model.learn(total_timesteps=1e7, callback=callback)

        env.close()


if __name__ == "__main__":
    main()
