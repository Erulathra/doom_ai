import numpy as np
import vizdoom as vzd

from enum import Enum

from vizdoom import GameVariable

from EventBuffer import EventBuffer
from PositionBuffer import PositionBuffer


class EventType(Enum):
    MOVEMENT = 1
    SHOOTING = 2
    PICKUP_AMMO = 3
    PICKUP_HEALTH = 4
    KILL_MONSTER = 5
    DAMAGE_MONSTER = 6
    PICKUP_ARMOUR = 7


class RewardShaping:
    def __init__(self, event_buffer: EventBuffer) -> None:
        self.event_buffer = event_buffer

        self.distance_moved_squared = 0
        self.intrinsic_reward = 0
        self.extrinsic_reward = 0

        self.position_buffer = PositionBuffer()

    def get_reward(self, reward: float) -> float:
        intrinsic_reward_this_step = self.event_buffer.intrinsic_reward(self.events_this_step)
        self.intrinsic_reward += intrinsic_reward_this_step
        self.extrinsic_reward += reward
        return reward + intrinsic_reward_this_step

    def step(self, doom_game: vzd.DoomGame):
        self.events_this_step = self._get_events(doom_game)

        self.events_this_episode += self.events_this_step

        self.position_buffer.record_position((
            doom_game.get_game_variable(GameVariable.POSITION_X),
            doom_game.get_game_variable(GameVariable.POSITION_Y)))

        return

    def new_episode(self):
        self.events_this_episode = np.zeros(len(EventType))
        self.extrinsic_reward = 0
        self.intrinsic_reward = 0

        return

    def episode_finished(self):
        self.event_buffer.record_events(self.events_this_episode)
        return

    def first_step(self, doom_game: vzd.DoomGame):
        self.events_this_step = np.zeros(len(EventType))
        self._reset_previous_state(doom_game)

        return

    def _reset_previous_state(self, doom_game):
        self.previous_health = doom_game.get_game_variable(GameVariable.HEALTH)
        self.previous_armour = doom_game.get_game_variable(GameVariable.ARMOR)
        self.selected_weapon = doom_game.get_game_variable(GameVariable.SELECTED_WEAPON)
        self.selected_weapon_ammo = doom_game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.previous_position_x = doom_game.get_game_variable(GameVariable.POSITION_X)
        self.previous_position_y = doom_game.get_game_variable(GameVariable.POSITION_Y)
        self.previous_kills = doom_game.get_game_variable(GameVariable.KILLCOUNT)
        self.previous_damage_count = doom_game.get_game_variable(GameVariable.DAMAGECOUNT)

    def _get_events(self, doom_game: vzd.DoomGame):
        events_methods = [
            self.get_moving_event,
            self.get_shooting_event,
            self.get_pickup_ammo_event,
            self.get_pickup_health_event,
            self.get_killing_event,
            self.get_damage_event,
            self.get_pickup_armour_event,
        ]

        events = np.zeros(len(EventType))

        for (i, method) in enumerate(events_methods):
            events[i] = int(method(doom_game) != 0)

        self._reset_previous_state(doom_game)

        return events

    def get_statistics(self):
        result = {
            "intrinsic_reward": self.intrinsic_reward,
            "extrinsic_reward": self.extrinsic_reward
        }

        events_types = range(1, len(EventType) + 1)

        for i, event_type in enumerate(events_types):
            event_type_enum = EventType(event_type)

            result[event_type_enum.name] = self.events_this_episode[i]

        return result

    # event logic
    def get_pickup_health_event(self, doom_game: vzd.DoomGame):
        health = doom_game.get_game_variable(GameVariable.HEALTH)
        if health > self.previous_health:
            return EventType.PICKUP_HEALTH.value

        return 0

    def get_pickup_ammo_event(self, doom_game: vzd.DoomGame):
        ammo = doom_game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        if ammo > self.selected_weapon_ammo:
            return EventType.PICKUP_AMMO.value

        return 0

    def get_pickup_armour_event(self, doom_game: vzd.DoomGame):
        armour = doom_game.get_game_variable(GameVariable.ARMOR)
        if armour > self.previous_armour:
            return EventType.PICKUP_ARMOUR.value

        return 0

    def get_shooting_event(self, doom_game: vzd.DoomGame):
        ammo = doom_game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        if ammo < self.selected_weapon_ammo:
            return EventType.SHOOTING.value

        return 0

    def get_killing_event(self, doom_game: vzd.DoomGame):
        kills = doom_game.get_game_variable(GameVariable.KILLCOUNT)
        if kills > self.previous_kills:
            return EventType.KILL_MONSTER.value

        return 0

    def get_damage_event(self, doom_game: vzd.DoomGame):
        damage_count = doom_game.get_game_variable(GameVariable.DAMAGECOUNT)
        if damage_count > self.previous_damage_count:
            return EventType.DAMAGE_MONSTER.value

        return 0

    def get_moving_event(self, doom_game: vzd.DoomGame):
        position_x = doom_game.get_game_variable(GameVariable.POSITION_X)
        position_y = doom_game.get_game_variable(GameVariable.POSITION_Y)

        delta_x = (position_x - self.previous_position_x)
        delta_y = (position_y - self.previous_position_y)

        distance_squared = delta_x ** 2 + delta_y ** 2

        self.distance_moved_squared += distance_squared

        if self.distance_moved_squared ** 0.5 > 1.0:
            self.distance_moved_squared = 0
            return EventType.MOVEMENT.value

        return 0


class SimpleRewardShaping(RewardShaping):

    def get_reward(self, reward: float) -> float:
        self.intrinsic_reward = 0
        self.extrinsic_reward += reward
        return reward
