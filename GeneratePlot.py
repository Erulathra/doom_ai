import math

import matplotlib.pyplot as plt
import numpy as np

from sympy.physics.control.control_plots import matplotlib

FILE_ONE = 'logs/debug/health_gathering/mem_1/adv_action_space/TEST/progress.csv'

SMOOTHING_VALUE = 0.90


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


def main():
    matplotlib.use('qtAgg')

    data = read_csf_to_dict(FILE_ONE)

    show_one_value_plot(data)
    # show_roe_plot(data)


def show_one_value_plot(data):
    x_points = data["timetotal_timesteps"]

    y_points_raw = data["rolloutep_len_mean"].tolist()
    y_points_smooth = smooth(data["rolloutep_len_mean"].tolist(), SMOOTHING_VALUE)

    plt.plot(x_points, y_points_raw, color='#00FFFF22')
    plt.plot(x_points, y_points_smooth, color='#00FFFFFF')

    plt.xlabel("Liczba kroków nauki")
    plt.ylabel("Długość scenariusza")

    plt.show()


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


if __name__ == "__main__":
    main()
