import numpy as np
import vizdoom as vzd

from enum import Enum

from EventBuffer import EventBuffer
from PositionBuffer import PositionBuffer

# Game variables
HEALTH = 0
ARMOUR = 1
SELECTED_WEAPON = 2
AMMO = 3
DAMAGE_COUNT = 4
POSITION_X = 5
POSITION_Y = 6
KILLS = 7


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

    def get_reward(self, reward: float, doom_state: vzd.GameState) -> float:
        intrinsic_reward_this_step = self.event_buffer.intrinsic_reward(self.events_this_step)
        self.intrinsic_reward += intrinsic_reward_this_step
        self.extrinsic_reward += reward
        return reward + intrinsic_reward_this_step

    def step(self, doom_state: vzd.GameState):
        self.events_this_step = self._get_events(doom_state)

        self.events_this_episode += self.events_this_step

        game_variables = doom_state.game_variables
        self.position_buffer.record_position((game_variables[POSITION_X], game_variables[POSITION_Y]))

        return

    def new_episode(self, doom_state: vzd.GameState):
        self.events_this_episode = np.zeros(len(EventType))
        self.extrinsic_reward = 0
        self.intrinsic_reward = 0

        return

    def episode_finished(self):
        self.event_buffer.record_events(self.events_this_episode)

        return

    def first_step(self, doom_state: vzd.GameState):
        self.events_this_step = np.zeros(len(EventType))

        game_variables = doom_state.game_variables

        self.previous_health = game_variables[HEALTH]
        self.previous_armour = game_variables[ARMOUR]
        self.selected_weapon = game_variables[SELECTED_WEAPON]
        self.selected_weapon_ammo = game_variables[AMMO]
        self.previous_position_x = game_variables[POSITION_X]
        self.previous_position_y = game_variables[POSITION_Y]
        self.previous_kills = game_variables[KILLS]
        self.previous_damage_count = game_variables[DAMAGE_COUNT]

        return

    def _get_events(self, doom_state: vzd.GameState):
        game_variables = doom_state.game_variables

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
            events[i] = int(method(game_variables) != 0)

        self.previous_health = game_variables[HEALTH]
        self.previous_armour = game_variables[ARMOUR]
        self.selected_weapon = game_variables[SELECTED_WEAPON]
        self.selected_weapon_ammo = game_variables[AMMO]
        self.previous_position_x = game_variables[POSITION_X]
        self.previous_position_y = game_variables[POSITION_Y]
        self.previous_kills = game_variables[KILLS]
        self.previous_damage_count = game_variables[DAMAGE_COUNT]

        return events

    def get_statistics(self):
        result = {}
        result["intrinsic_reward"] = self.intrinsic_reward
        result["extrinsic_reward"] = self.extrinsic_reward

        events_types = range(1, len(EventType) + 1)
        event_mean = self.event_buffer.get_event_mean()
        for i, event_type in enumerate(events_types):
            event_type_enum = EventType(event_type)

            result[event_type_enum.name] = event_mean[i]

        return result

    # event logic
    def get_pickup_health_event(self, game_variables):
        if game_variables[HEALTH] > self.previous_health:
            return EventType.PICKUP_HEALTH.value

        return 0

    def get_pickup_ammo_event(self, game_variables):
        if game_variables[AMMO] > self.selected_weapon_ammo:
            return EventType.PICKUP_AMMO.value

        return 0

    def get_pickup_armour_event(self, game_variables):
        if game_variables[ARMOUR] > self.previous_armour:
            return EventType.PICKUP_ARMOUR.value

        return 0

    def get_shooting_event(self, game_variables):
        if game_variables[AMMO] < self.selected_weapon_ammo:
            return EventType.SHOOTING.value

        return 0

    def get_killing_event(self, game_variables):
        if game_variables[KILLS] > self.previous_kills:
            return EventType.KILL_MONSTER.value

        return 0

    def get_damage_event(self, game_variables):
        if game_variables[DAMAGE_COUNT] > self.previous_damage_count:
            return EventType.DAMAGE_MONSTER.value

        return 0

    def get_moving_event(self, game_variables):
        position_x = game_variables[POSITION_X]
        position_y = game_variables[POSITION_Y]

        delta_x = (position_x - self.previous_position_x)
        delta_y = (position_y - self.previous_position_y)

        distance_squared = delta_x ** 2 + delta_y ** 2

        self.distance_moved_squared += distance_squared

        if self.distance_moved_squared ** 0.5 > 1.0:
            self.distance_moved_squared = 0
            return EventType.MOVEMENT.value

        return 0


class SimpleRewardShaping(RewardShaping):

    def get_reward(self, reward: float, doom_state: vzd.GameState) -> float:
        self.intrinsic_reward = 0
        self.extrinsic_reward += reward
        return reward
