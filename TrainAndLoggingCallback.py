import os
import tensorboard as tb

from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1, reward_shaping=None):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

        self.reward_shaping = reward_shaping

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        # Log ROE
        if self.reward_shaping is not None:
            statistics = self.reward_shaping.get_statistics()

            for name, value in statistics.items():
                self.logger.record(f"emo/{name}", value)

        
        return True


