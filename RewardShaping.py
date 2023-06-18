import vizdoom as vzd

from enum import Enum


class EventType(Enum):
    MOVEMENT = 1
    SHOOTING = 2
    PICKUP_AMMO = 3
    PICKUP_HEALTH = 4
    KILL_MONSTER = 5
    DAMAGE_MONSTER = 6


class RewardShaping:
    def __init__(self, buffer_size=100) -> None:
        self._buffer_size = buffer_size
        self._events_buffer = []

        self._episode_events = []

        self._last_health = 100
        self._last_position = None
        self._whole_distance = 0
        self._last_distance = 0

        self._last_ammo = 6

        self._last_kill_count = 0

        self._last_damage_count = 0
        pass

    def get_reward(self, reward: float, doom_state: vzd.GameState) -> float:
        return reward + self._compute_roe_reward(doom_state)

    def _compute_roe_reward(self, doom_state: vzd.GameState) -> float:
        events_this_time_step = self._get_events_this_time_step(doom_state)
        reward = 0

        for event_type, event_value in events_this_time_step.items():
            if event_value == 0:
                continue

            episodic_mean_occurrence = self._compute_episodic_mean_occurrence(
                event_type
            )
            episodic_mean_occurrence = max(episodic_mean_occurrence, 0.01)

            reward += 1 / episodic_mean_occurrence
            self._episode_events.append(event_type)

        return reward

    # TODO:
    def _compute_episodic_mean_occurrence(self, event_type: EventType) -> float:
        if len(self._events_buffer) == 0:
            return 0

        result = 0
        for episode in self._events_buffer:
            result += episode[event_type]

        return result / len(self._events_buffer)

    def _get_events_this_time_step(self, doom_state: vzd.GameState):
        game_variables = doom_state.game_variables

        health = game_variables[0]
        armour = game_variables[1]
        selected_weapon = game_variables[2]
        selected_ammo = game_variables[3]
        damage_count = game_variables[4]
        position = [game_variables[5], game_variables[6]]
        kills = game_variables[7]

        events = {
            EventType.MOVEMENT: self._compute_distance_event(position),
            EventType.PICKUP_HEALTH: self._compute_health_event(health),
            EventType.PICKUP_AMMO: self._compute_pickup_ammo_event(selected_ammo),
            EventType.SHOOTING: self._compute_shoot_event(selected_ammo),
            EventType.KILL_MONSTER: self._compute_kill_event(kills),
            EventType.DAMAGE_MONSTER: self._compute_damage_event(damage_count),
        }

        self._last_health = health
        self._last_ammo = selected_ammo
        self._last_position = position
        self._last_kill_count = kills
        self._last_damage_count = damage_count

        return events

    def _compute_distance_event(self, position):
        if self._last_position is None:
            self._last_position = position
            return 0

        distance_vector = [
            position[0] - self._last_position[0],
            position[1] - self._last_position[1],
        ]

        distance = distance_vector[0] ** 2 + distance_vector[1] ** 2
        self._whole_distance += distance

        if (self._whole_distance - self._last_distance) > 1:
            self._last_distance = self._whole_distance
            return 1

        return 0

    def _compute_damage_event(self, damage_count):
        delta_damage = damage_count - self._last_damage_count

        if delta_damage > 0:
            return 1
        
        return 0

    def _compute_health_event(self, health):
        health = max(health, 0)
        delta_health = health - self._last_health

        if delta_health <= 0:
            return 0

        return 1

    def _compute_pickup_ammo_event(self, ammo) -> float:
        delta_ammo = ammo - self._last_ammo

        if delta_ammo > 0:
            return 1
        return 0

    def _compute_shoot_event(self, ammo) -> float:
        delta_ammo = ammo - self._last_ammo

        if delta_ammo < 0:
            return 1

        return 0

    def _compute_kill_event(self, kills):
        delta_kills = kills - self._last_kill_count

        if delta_kills > 0:
            return 1

        return 0

    def new_episode(self):
        episode_sumary = {}

        for event_type in EventType:
            episode_sumary[event_type] = self._episode_events.count(event_type)

        self._episode_events = []
        self._add_event(episode_sumary)

        # reset variables
        self._last_health = 100
        self._last_position = None
        self._whole_distance = 0
        self._last_distance = 0

        self._last_ammo = 6

        self._last_kill_count = 0

        self._last_damage_count = 0


    def _add_event(self, event):
        self._events_buffer.append(event)
        if len(self._events_buffer) > self._buffer_size:
            self._events_buffer.pop(0)