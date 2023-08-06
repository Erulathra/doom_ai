import math

import numpy as np


class PositionBuffer:
    def __init__(self, buffer_size: int = 12800):
        self.position_history: list[tuple[float, float]] = []
        self.buffer_size = buffer_size
        self.min_position = [0, 0]
        self.max_position = [0, 0]

    def record_position(self, position: tuple[float, float]):
        self.position_history.append(position)

        self.min_position[0] = min(self.min_position[0], position[0])
        self.min_position[1] = min(self.min_position[1], position[1])

        self.max_position[0] = max(self.max_position[0], position[0])
        self.max_position[1] = max(self.max_position[1], position[1])

        if len(self.position_history) > self.buffer_size:
            self.position_history.pop(0)

    def get_position_heat_matrix(self, size: tuple[int, int]) -> np.ndarray:
        position_matrix = np.zeros(size)

        for i, position in enumerate(self.position_history):
            scaled_x = math.floor((position[0] - self.min_position[0]) / (self.max_position[0] - self.min_position[0]) * (size[0] - 1))
            scaled_y = math.floor((position[1] - self.min_position[1]) / (self.max_position[1] - self.min_position[1]) * (size[1] - 1))

            position_matrix[scaled_x, scaled_y] += 1

        result = np.divide(position_matrix, len(self.position_history))
        result = np.clip(result, 0, 0.25)
        return result
