import os
import time

from rich.progress import Progress

from EventBuffer import EventBuffer
from VizDoomEnv import VizDoomEnv
from stable_baselines3 import PPO, A2C
from ROERewardShaping import ROERewardShaping, EVENTS_TYPES_NUMBER

import numpy as np

baseline_models = {
    'health_gathering': 'model/final/baseline/sep_buffer/basic_action/mem_1/health_gathering/best_model_2500000.zip',
    'health_gathering_supreme': 'model/final/baseline/sep_buffer/basic_action/mem_1/health_gathering_supreme/best_model_2500000.zip',
    'my_way_home': 'model/final/baseline/sep_buffer/basic_action/mem_1/my_way_home/best_model_2500000.zip',
    'deadly_corridor': 'model/final/baseline/sep_buffer/basic_action/mem_1/deadly_corridor/best_model_2500000.zip',
    'simple_deathmatch': 'model/final/baseline/sep_buffer/basic_action/mem_1/simple_deathmatch/best_model_2300000.zip',
    'deathmatch': 'model/final/baseline/sep_buffer/basic_action/mem_1/deathmatch/best_model_600000.zip'
}

roe_models = {
    'health_gathering': 'model/final/ROE/sep_buffer/basic_action/mem_1/health_gathering/best_model_2500000.zip',
    'health_gathering_supreme': 'model/final/ROE/sep_buffer/basic_action/mem_1/health_gathering_supreme/best_model_2500000.zip',
    'my_way_home': 'model/final/ROE/sep_buffer/basic_action/mem_1/my_way_home/best_model_2500000.zip',
    'deadly_corridor': 'model/final/ROE/sep_buffer/basic_action/mem_1/deadly_corridor/best_model_2500000.zip',
    'simple_deathmatch': 'model/final/ROE/sep_buffer/basic_action/mem_1/simple_deathmatch/best_model_2500000.zip',
    'deathmatch': 'model/final/ROE/sep_buffer/basic_action/mem_1/deathmatch/best_model_600000.zip'
}

scenarios = [
    ('health_gathering', 'Health gathering'),
    ('health_gathering_supreme', 'Health gathering supreme'),
    ('my_way_home', 'My way home'),
    ('deadly_corridor', 'Deadly corridor'),
    ('simple_deathmatch', 'Simple deathmatch'),
    ('deathmatch', 'Deathmatch'),
]


def main():
    results = [{}, {}]

    with Progress() as progress:
        whole_task = progress.add_task('[green]Whole...', total=len(scenarios) * 2)

        for i, models in enumerate([baseline_models, roe_models]):
            for scenario, scenario_name in scenarios:
                model_dir = models[scenario]

                if model_dir is None:
                    results[i][scenario] = (None, None)
                    continue

                model = A2C.load(model_dir)

                env = VizDoomEnv(
                    scenario,
                    is_window_visible=False,
                    doom_skill=3,
                    memory_size=1,
                    advanced_actions=False,

                    reward_shaping_class=ROERewardShaping,
                    reward_shaping_kwargs={
                        'event_buffer_class': EventBuffer,
                        'event_buffer_kwargs': {'n': EVENTS_TYPES_NUMBER}
                    }
                )
                env.frame_skip = 1

                sub_task = progress.add_task(f'[blue]{scenario}...', total=10)
                episode_results = []

                for episode in range(10):
                    progress.update(sub_task, advance=1)
                    progress.refresh()

                    obs, _ = env.reset()
                    action, _ = model.predict(obs)

                    terminated = False

                    step = 0
                    while not terminated:
                        if step % 4 == 0:
                            action, _ = model.predict(obs)

                        obs, reward, terminated, _, info = env.step(action)

                        step += 1
                        # time.sleep(1 / (30. * 4))

                    stats = env.get_statistics()
                    episode_results.append(stats['extrinsic_reward'])

                env.close()
                model = None

                progress.advance(whole_task, advance=1)

                print(episode_results)
                results[i][scenario] = (np.mean(episode_results), np.std(episode_results))

    print('\\begin{table}[H]')
    print('\t\\begin{tabular}{|| c | c | c ||}')
    print('\t\t\\hline')
    print('\t\tscenario & VizDoom & ROE \\\\')
    print('\t\t\\hline\\hline')
    for scenario, scenario_name in scenarios:
        print(f'\t\t{scenario_name} & ${results[0][scenario][0]:.3g} \\pm {results[0][scenario][1]:.3g}$ &'
              f' ${results[1][scenario][0]:.3g} \\pm {results[1][scenario][1]:.3g}$ \\\\')
        print('\t\t\\hline')

    print('\t\\end{tabular}')
    print('\\end{table}')


if __name__ == "__main__":
    main()
