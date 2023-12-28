import math
import os

import matplotlib.pyplot as plt
import numpy as np

from sympy.physics.control.control_plots import matplotlib


SMOOTHING_VALUE = 0.95


def read_csf_to_dict(file_path: str):
    return np.genfromtxt(file_path, delimiter=',', names=True)


def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed




def generate_compare_plots(data_dirs, data_name, data_human_readable_name):
    scenarios = os.listdir(data_dirs[0])
    for scenario in scenarios:
        datas = []
        for data_dir in data_dirs:
            scenario_path_one = os.path.join(data_dir, scenario, 'progress.csv')
            datas.append(read_csf_to_dict(scenario_path_one))

        output_dir = os.path.join(OUT, data_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f'{scenario}.png')

        generate_compare_plot(datas, data_name, data_human_readable_name, output_path)


def show_one_value_plot(data, save_path=None):
    x_points = data["timetotal_timesteps"]

    y_points_raw = data["rolloutep_len_mean"].tolist()
    y_points_smooth = smooth(data["rolloutep_len_mean"].tolist(), SMOOTHING_VALUE)

    plt.plot(x_points, y_points_raw, color='#FF000022')
    plt.plot(x_points, y_points_smooth, color='#FF0000FF')

    plt.xlabel("liczba kroków nauki")
    plt.ylabel("Długość scenariusza")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()

def generate_compare_plot(datas, data_name, data_human_readable_name, save_path=None):
    colors = ['#0000FF', '#FF0000', '#00FF00']
    array_color = [(datas[i], colors[i]) for i in range(0, len(datas))]

    for (data, color) in array_color:
        x_points = data["timetotal_timesteps"]

        y_points_raw = data[data_name].tolist()
        y_points_smooth = smooth(data[data_name].tolist(), SMOOTHING_VALUE)

        plt.plot(x_points, y_points_raw, color=f'{color}22')
        plt.plot(x_points, y_points_smooth, color=f'{color}FF')

        plt.xlabel("liczba kroków nauki")
        plt.ylabel(data_human_readable_name)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()


def show_roe_plot(data):
    x_points = data["timetotal_timesteps"]

    events_names = [
        "emoPICKUP_AMMO",
        "emoMOVEMENT",
        "emoPICKUP_HEALTH",
        "emoKILL_MONSTER",
        "emoDAMAGE_MONSTER",
        "emoepisode_length",
        "emoSHOOTING",
        "emoPICKUP_ARMOUR"
    ]

    colors_names = list(matplotlib.colors.TABLEAU_COLORS)

    for event_id in range(len(events_names)):
        event = events_names[event_id]

        y_points_raw = data[event].tolist()
        y_points_smooth = smooth(y_points_raw, SMOOTHING_VALUE)

        color = colors_names[event_id]

        plt.plot(x_points, y_points_raw, color=color, alpha=0.2)
        plt.plot(x_points, y_points_smooth, color=color, alpha=1.0, label=event)

    plt.xlabel("Liczba kroków nauki")
    plt.ylabel("Liczba wystąpień zdarzenia")
    # plt.yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.show()


FILE_ONE = 'logs/final/baseline/sep_buffer/adv_action/mem_1/health_gathering/progress.csv'
DIR_ONE = 'logs/final/baseline/sep_buffer/basic_action/mem_1'
DIR_TWO = 'logs/final/ROE/sep_buffer/basic_action/mem_1'

OUT = 'plots/baselines_vs_roe'
matplotlib.use('qtAgg')

# ROE vs Baselines
generate_compare_plots([DIR_ONE, DIR_TWO], "emoepisode_length", "długość epizodu")
generate_compare_plots([DIR_ONE, DIR_TWO], "emoextrinsic_reward", "zewnętrzna nagroda")

generate_compare_plots([DIR_ONE, DIR_TWO], "emoPICKUP_AMMO", "podniesienie amunicji")
generate_compare_plots([DIR_ONE, DIR_TWO], "emoPICKUP_HEALTH", "podniesienie apteczki")

# A2C vs PPO
DIR_ONE = 'logs/final/PPO/sep_buffer/basic_action/mem_1'
OUT = 'plots/a2c_vs_ppo'
generate_compare_plots([DIR_ONE, DIR_TWO], "emoepisode_length", "długość epizodu")
generate_compare_plots([DIR_ONE, DIR_TWO], "emoextrinsic_reward", "zewnętrzna nagroda")

# Buffers
DIR_ONE = 'logs/final/SAME_BUF/sep_buffer/adv_action/mem_1'
DIR_TWO = 'logs/final/A2C_ADV/sep_buffer/adv_action/mem_1'
OUT = 'plots/buffers'
generate_compare_plots([DIR_ONE, DIR_TWO], "emoextrinsic_reward", "zewnętrzna nagroda")

# Action space
DIR_ONE = 'logs/final/A2C_ADV/sep_buffer/adv_action/mem_1'
DIR_TWO = 'logs/final/ROE/sep_buffer/basic_action/mem_1'
OUT = 'plots/act_space'
generate_compare_plots([DIR_ONE, DIR_TWO], "emoextrinsic_reward", "zewnętrzna nagroda")

# MEM
DIR_ONE = 'logs/final/A2C_ADV/sep_buffer/adv_action/mem_1'
DIR_TWO = 'logs/final/MEM_TEST/sep_buffer/adv_action/mem_5'
DIR_THREE = 'logs/final/MEM_TEST/sep_buffer/adv_action/mem_10'
OUT = 'plots/mem'
generate_compare_plots([DIR_ONE, DIR_TWO, DIR_THREE], "emoextrinsic_reward", "zewnętrzna nagroda")
