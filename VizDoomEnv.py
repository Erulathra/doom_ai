import os.path
from typing import Any

import numpy as np
import cv2

import vizdoom as vzd
from vizdoom import GameState

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete

wad_path = 'Test/DOOM2.WAD'

class VizDoomEnv(Env):
    def __init__(self, scenario: str, frame_skip=10, resolution=(160, 120), is_window_visible=False):
        super().__init__()

        self.scenario_path = os.path.join(os.path.curdir, "scenarios", scenario + ".cfg")
        self.resolution = resolution
        self.frame_skip = frame_skip

        self._is_window_visible = is_window_visible

        self.game = vzd.DoomGame()

        if os.path.exists(wad_path):
            self.game.set_doom_game_path(wad_path)

        self._setup_game()
        self._setup_environment()
        self.frame_skip = frame_skip

    def _setup_game(self):
        self.game.load_config(self.scenario_path)
        self.number_of_actions = self.game.get_available_buttons_size()

        self.game.set_window_visible(self._is_window_visible)
        self.game.init()

    def _setup_environment(self):
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(self.resolution[1], self.resolution[0], 3),
                                     dtype=np.uint8)

        self.action_space = Discrete(self.number_of_actions)

    def step(self, action: ActType):
        available_actions = np.identity(self.number_of_actions, dtype=np.uint8)
        reward = self.game.make_action(available_actions[action], self.frame_skip)

        # Get State data
        doom_state: GameState = self.game.get_state()
        if doom_state:
            img = doom_state.screen_buffer
            img = self.prepare_color_buffer(img)
            ammo = doom_state.game_variables[0]

        else:
            img = np.zeros(self.observation_space.shape)
            ammo = 0

        info = {"ammo": ammo}

        terminated = self.game.is_episode_finished()

        return img, reward, terminated, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.game.new_episode()

        doom_state: GameState = self.game.get_state()

        img = doom_state.screen_buffer
        img = self.prepare_color_buffer(img)
        info = {"ammo": doom_state.game_variables[0]}

        return img, info

    def prepare_color_buffer(self, observation):
        # grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(np.moveaxis(observation, 0, -1), self.resolution, interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (self.resolution[1], self.resolution[0], 3))

    def close(self):
        self.game.close()
