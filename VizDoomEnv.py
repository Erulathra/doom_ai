import os.path
from typing import Any

import numpy as np
import cv2

import vizdoom as vzd
from vizdoom import GameState

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete

from rich import print

config_path = os.path.join(vzd.scenarios_path, "basic.cfg")
number_of_available_actions = 3
is_window_visible = False

frame_skip = 5
resolution = (24, 32)


class VizDoomEnv(Env):
    def __init__(self):
        super().__init__()

        self.game = vzd.DoomGame()
        self._setup_game()
        self._setup_environment()

    def _setup_game(self):
        self.game.load_config(config_path)
        self.game.set_window_visible(is_window_visible)
        self.game.init()

    def _setup_environment(self):
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(resolution[1], resolution[0], 1),
                                     dtype=np.uint8)

        self.action_space = Discrete(number_of_available_actions)

    def step(self, action: ActType):
        available_actions = np.identity(number_of_available_actions, dtype=np.uint8)
        reward = self.game.make_action(available_actions[action], frame_skip)

        # Get State data
        doom_state: GameState = self.game.get_state()
        if doom_state:
            img = doom_state.screen_buffer
            img = self.grey_scale(img)
            info = doom_state.game_variables
        else:
            img = np.zeros(self.observation_space.shape)
            info = 0

        done = self.game.is_episode_finished()

        return img, reward, done, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.game.new_episode()
        img = self.game.get_state().screen_buffer
        img = self.grey_scale(img)
        return img

    def grey_scale(self, observation):
        grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(grey, resolution, interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (resolution[1], resolution[0], 1))

    def close(self):
        self.game.close()
