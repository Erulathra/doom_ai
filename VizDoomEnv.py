import os.path
from typing import Any

import numpy as np
import cv2

import vizdoom as vzd
from vizdoom import GameState

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete

from  VizDoomActionSpace import get_available_actions

wad_path = "Test/DOOM2.WAD"


class VizDoomEnv(Env):
    def __init__(
        self,
        scenario: str,
        frame_skip=10,
        resolution=(160, 120),
        is_window_visible=False,
        doom_skill=-1,
        reward_shaping=None,
        memory_size=1,
    ):
        super().__init__()

        self.scenario_path = os.path.join(
            os.path.curdir, "scenarios", scenario + ".cfg"
        )
        self.resolution = resolution
        self.frame_skip = frame_skip

        self._is_window_visible = is_window_visible

        self.game = vzd.DoomGame()

        if doom_skill >= 0:
            self.game.set_doom_skill(doom_skill)

        if os.path.exists(wad_path):
            self.game.set_doom_game_path(wad_path)

        self.is_first_step = True

        self.memory_size = memory_size
        self.memory = []

        self._setup_game()
        self._setup_environment()
        self.frame_skip = frame_skip
        self.reward_shaping = reward_shaping

        self.is_first_step = True

    def _setup_game(self):
        self.game.load_config(self.scenario_path)
        self.number_of_actions = self.game.get_available_buttons_size()

        self.game.set_window_visible(self._is_window_visible)
        self.game.init()

    def _setup_environment(self):
        available_buttons = self.game.get_available_buttons()
        self.available_actions = get_available_actions(available_buttons)

        self.observation_space = Box(low=0, high=255, shape=(self.memory_size, self.resolution[1], self.resolution[0]), dtype=np.uint8)
        self.action_space = Discrete(len(self.available_actions))

    def step(self, action: ActType):
        reward = self.game.make_action(self.available_actions[action], self.frame_skip)

        # Get State data
        if self.game.get_state():
            doom_state: GameState = self.game.get_state()
            screen_buffer = doom_state.screen_buffer
            screen_buffer = self.prepare_color_buffer(screen_buffer)

            if self.reward_shaping is not None:
                if self.is_first_step:
                    self.reward_shaping.first_step(doom_state)
                else:
                    self.reward_shaping.step(doom_state)

                reward = self.reward_shaping.get_reward(reward, doom_state)

            self.is_first_step = False

        else:
            screen_buffer = np.zeros((self.resolution[1], self.resolution[0]))

        terminated = self.game.is_episode_finished()
        truncated = False

        self.append_frame_to_memory(screen_buffer)

        if terminated and self.reward_shaping is not None:
            self.reward_shaping.episode_finished()

        return self.get_memory_matrix(), reward, terminated, truncated, {}

    def append_frame_to_memory(self, screen_buffer):
        self.memory.append(screen_buffer)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.game.new_episode()
        
        if self.reward_range is not None:
            self.reward_shaping.new_episode(self.game.get_state())

        doom_state: GameState = self.game.get_state()

        screen_buffer = doom_state.screen_buffer
        screen_buffer = self.prepare_color_buffer(screen_buffer)

        self.append_frame_to_memory(screen_buffer)

        memory_matrix = self.get_memory_matrix()

        self.is_first_step = True

        return memory_matrix, {}

    def get_memory_matrix(self):
        memory_matrix = np.zeros((self.memory_size, self.resolution[1], self.resolution[0]))
        for i, frame in enumerate(self.memory):
            matrix_index = self.memory_size - 1 - i
            memory_matrix[matrix_index] = frame
        return memory_matrix

    def prepare_color_buffer(self, observation):
        resize = cv2.resize(
            np.moveaxis(observation, 0, -1),
            self.resolution,
            interpolation=cv2.INTER_CUBIC,
        )

        return np.reshape(resize, (self.resolution[1], self.resolution[0]))

    def close(self):
        self.game.close()
