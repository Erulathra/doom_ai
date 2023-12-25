import csv
import os
from copy import copy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorboard as tf

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

from VizDoomEnv import VizDoomEnv


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

        csv_log_path = os.path.join(save_path, 'logs.csv')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        statistics = self._get_average_statistics()

        for name, value in statistics.items():
            self.logger.record(f"emo/{name}", value)

        self._log_heatmap()

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _log_heatmap(self):
        if self.num_timesteps % 12800 == 0 and self.num_timesteps != 0:
            figure = plt.Figure()

            vec_datas = self.training_env.env_method('get_position_heat_matrix', (64, 64))
            data = sum(vec_datas) / len(vec_datas)

            figure.add_subplot().imshow(data, cmap='hot', interpolation='nearest')
            self.logger.record("position/heatmap", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

    def _get_average_statistics(self) -> Dict[str, float]:
        vec_stats = self.training_env.env_method('get_statistics')
        result = copy(vec_stats[0])

        for stats in vec_stats[1:]:
            for name, value in stats.items():
                result[name] += value

        for name, value in result.items():
            result[name] /= len(vec_stats)

        return result
