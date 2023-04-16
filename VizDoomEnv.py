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

frame_skip = 10
resolution = (160, 120)


class VizDoomEnv(Env):
    def __init__(self, is_window_visible=False):
        super().__init__()

        self._is_window_visible = is_window_visible

        self.game = vzd.DoomGame()
        self._setup_game()
        self._setup_environment()
        self.frame_skip = frame_skip

    def _setup_game(self):
        self.game.load_config(config_path)
        self.game.set_window_visible(self._is_window_visible)
        self.game.init()

    def _setup_environment(self):
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(resolution[1], resolution[0], 3),
                                     dtype=np.uint8)

        self.action_space = Discrete(number_of_available_actions)

    def step(self, action: ActType):
        available_actions = np.identity(number_of_available_actions, dtype=np.uint8)
        reward = self.game.make_action(available_actions[action], self.frame_skip)

        # Get State data
        doom_state: GameState = self.game.get_state()
        if doom_state:
            img = doom_state.screen_buffer
            img = self.grey_scale(img)
            ammo = doom_state.game_variables[0]

        else:
            img = np.zeros(self.observation_space.shape)
            ammo = 0

        info = {"ammo": ammo}

        terminated = self.game.is_episode_finished()

        # todo: implemement truncated
        return img, reward, terminated, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.game.new_episode()

        doom_state: GameState = self.game.get_state()

        img = doom_state.screen_buffer
        img = self.grey_scale(img)
        info = {"ammo": doom_state.game_variables[0]}

        return img, info

    def grey_scale(self, observation):
        # grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(np.moveaxis(observation, 0, -1), resolution, interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (resolution[1], resolution[0], 3))

    def close(self):
        self.game.close()
