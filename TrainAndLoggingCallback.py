import os
from abc import ABC

from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback, ABC):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
