import math

import numpy as np
import vizdoom as vzd

from enum import Enum

from vizdoom import GameVariable

from EventBuffer import EventBuffer
from PositionBuffer import PositionBuffer

EVENTS_TYPES_NUMBER = 26


class VizdoomEvent(Enum):
    MOVEMENT = 0
    PICKUP_HEALTH = 1
    PICKUP_ARMOUR = 2
    SHOOTING = 3
    PICKUP_AMMO = 4,
    WEAPON_PICKUP_START = 5
    WEAPON_PICKUP_END = 14
    KILL_MONSTER = 15
    KILL_MONSTER_WEAPON_START = 16
    KILL_MONSTER_WEAPON_END = 24
    DAMAGE_MONSTER = 25


class ROERewardShaping:
    def __init__(
            self,
            event_buffer_class,
            event_buffer_kwargs,
            additional_reward_shaping_class=None
    ) -> None:
        self.event_buffer = event_buffer_class(**event_buffer_kwargs)

        self.distance_moved_squared = 0
        self.intrinsic_reward = 0
        self.extrinsic_reward = 0

        self.additional_reward_shaping_class = additional_reward_shaping_class
        self.additional_reward_shaping = None

        self.position_buffer = PositionBuffer()

    def get_reward(self, reward: float) -> float:
        intrinsic_reward_this_step = self.event_buffer.intrinsic_reward(self.events_this_step)
        self.intrinsic_reward += intrinsic_reward_this_step
        self.extrinsic_reward += reward

        additional_reward = 0

        if self.additional_reward_shaping is not None:
            additional_reward = self.additional_reward_shaping.get_reward(self.current_vars)
            self.extrinsic_reward += additional_reward

        return reward + intrinsic_reward_this_step + additional_reward

    def step(self, doom_game: vzd.DoomGame):
        self.events_this_step = self._get_events(doom_game)

        self.events_this_episode += self.events_this_step

        self.position_buffer.record_position((
            doom_game.get_game_variable(GameVariable.POSITION_X),
            doom_game.get_game_variable(GameVariable.POSITION_Y)))

        return

    def new_episode(self):
        self.events_this_episode = np.zeros(EVENTS_TYPES_NUMBER)
        self.extrinsic_reward = 0
        self.intrinsic_reward = 0

        if self.additional_reward_shaping_class is not None:
            self.additional_reward_shaping = self.additional_reward_shaping_class()

        return

    def episode_finished(self):
        self.event_buffer.record_events(self.events_this_episode)

        return

    def first_step(self, doom_game: vzd.DoomGame):
        self.last_position_x = doom_game.get_game_variable(GameVariable.POSITION_X)
        self.last_position_y = doom_game.get_game_variable(GameVariable.POSITION_Y)

        self.events_this_step = np.zeros(EVENTS_TYPES_NUMBER)
        self.current_vars = self._get_variables(doom_game)

        self._reset_previous_state()

        return

    def _reset_previous_state(self):
        self.last_vars = self.current_vars

    def _get_events(self, doom_game: vzd.DoomGame):
        self.current_vars = self._get_variables(doom_game)

        events = np.zeros(EVENTS_TYPES_NUMBER)

        # If died -> no event
        if self.current_vars[15] > self.last_vars[15]:
            return events

        # 0. Movement
        if self.current_vars[0] > self.last_vars[0]:
            events[0] = 1

        # 1. Health increase
        if self.current_vars[1] > self.last_vars[1]:
            events[1] = 1

        # 2. Armor increase
        if self.current_vars[2] > self.last_vars[2]:
            events[2] = 1

        # 3. Ammo decrease
        if self.current_vars[3] < self.last_vars[3]:
            events[3] = 1

        # 4. Ammo increase
        if self.current_vars[3] > self.last_vars[3]:
            events[4] = 1

        # 5-14. Weapon pickup 0-9
        for i in range(4, 14):
            if self.current_vars[i] > self.last_vars[i]:
                events[i + 1] = 1

        # 15-24 Kill increase - for each weapon
        if self.current_vars[14] > self.last_vars[14]:
            # events[15] = 1
            for i in range(0, 9):
                if self.current_vars[16] == i:  # If selected weapon
                    events[15 + i] = 1

        # 2
        if self.current_vars[17] > self.last_vars[17]:
            events[25] = 1

        self._reset_previous_state()

        return events

    def _get_variables(self, doom_game: vzd.DoomGame):
        variables = [
            self.get_distance_moved(doom_game),
            doom_game.get_game_variable(GameVariable.HEALTH),
            doom_game.get_game_variable(GameVariable.ARMOR),
            doom_game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),
            doom_game.get_game_variable(GameVariable.WEAPON0),
            doom_game.get_game_variable(GameVariable.WEAPON1),
            doom_game.get_game_variable(GameVariable.WEAPON2),
            doom_game.get_game_variable(GameVariable.WEAPON3),
            doom_game.get_game_variable(GameVariable.WEAPON4),
            doom_game.get_game_variable(GameVariable.WEAPON5),
            doom_game.get_game_variable(GameVariable.WEAPON6),
            doom_game.get_game_variable(GameVariable.WEAPON7),
            doom_game.get_game_variable(GameVariable.WEAPON8),
            doom_game.get_game_variable(GameVariable.WEAPON9),
            doom_game.get_game_variable(GameVariable.KILLCOUNT) + doom_game.get_game_variable(GameVariable.FRAGCOUNT),
            doom_game.get_game_variable(GameVariable.DEATHCOUNT),
            doom_game.get_game_variable(GameVariable.SELECTED_WEAPON),
            doom_game.get_game_variable(GameVariable.DAMAGECOUNT)
        ]

        return np.array(variables)

    def get_statistics(self):
        result = {
            "intrinsic_reward": self.intrinsic_reward,
            "extrinsic_reward": self.extrinsic_reward
        }

        events_to_record = [
            VizdoomEvent.MOVEMENT,
            VizdoomEvent.PICKUP_HEALTH,
            VizdoomEvent.PICKUP_ARMOUR,
            VizdoomEvent.PICKUP_AMMO,
            VizdoomEvent.DAMAGE_MONSTER,
            # VizdoomEvent.KILL_MONSTER, # Calculated later using weapon events
            VizdoomEvent.SHOOTING,
        ]

        for event_type in events_to_record:
            result[event_type.name] = self.events_this_episode[event_type.value]

        kill_count = 0

        for enum_id in range(VizdoomEvent.KILL_MONSTER_WEAPON_START.value,
                             VizdoomEvent.KILL_MONSTER_WEAPON_END.value + 1):
            kill_count += self.events_this_episode[enum_id]

        result["KILL_MONSTER"] = kill_count

        return result

    def get_distance_moved(self, doom_game: vzd.DoomGame):
        position_x = doom_game.get_game_variable(GameVariable.POSITION_X)
        position_y = doom_game.get_game_variable(GameVariable.POSITION_Y)

        delta_x = (position_x - self.last_position_x)
        delta_y = (position_y - self.last_position_y)

        distance_squared = delta_x ** 2 + delta_y ** 2

        self.distance_moved_squared += distance_squared

        return math.sqrt(self.distance_moved_squared)


class SimpleRewardShaping(ROERewardShaping):

    def get_reward(self, reward: float) -> float:
        self.intrinsic_reward = 0
        self.extrinsic_reward += reward
        return reward


class StaticBufferROERewardShaping(ROERewardShaping):
    static_buffer = None

    def __init__(
            self,
            event_buffer_class,
            event_buffer_kwargs,
            additional_reward_shaping_class=None
    ):
        ROERewardShaping.__init__(self, event_buffer_class, event_buffer_kwargs, additional_reward_shaping_class)
        if StaticBufferROERewardShaping.static_buffer is None:
            StaticBufferROERewardShaping.static_buffer = self.event_buffer
        else:
            self.event_buffer = StaticBufferROERewardShaping.static_buffer


class BotsAdditionalRewardShaping:
    def __init__(self):
        self.last_death_count = 0
        self.last_kill_count = 0
        pass

    def get_reward(self, game_vars):
        kill_count = game_vars[14]
        death_count = game_vars[15]

        additional_reward = 0.

        if kill_count > self.last_kill_count:
            additional_reward += 1.

        if death_count > self.last_death_count:
            additional_reward -= 1.

        self.last_kill_count = kill_count
        self.last_death_count = death_count

        return additional_reward
