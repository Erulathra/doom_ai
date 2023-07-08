import vizdoom as vzd

from enum import Enum

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
    def __init__(self, buffer_size=100) -> None:
        self.buffer_size = buffer_size
        self.event_buffer = []

        self.distance_moved_squared = 0

        pass

    def get_reward(self, reward: float, doom_state: vzd.GameState) -> float:
        return reward 

    def new_episode(self, doom_state: vzd.GameState):
        return

    def first_step(self, doom_state: vzd.GameState):
        game_variables = doom_state.game_variables

        self.previous_health = game_variables[HEALTH]
        self.previous_armour = game_variables[ARMOUR]
        self.selected_weapon = game_variables[SELECTED_WEAPON]
        self.selected_weapon_ammo = game_variables[AMMO]
        self.previous_position_x = game_variables[POSITION_X]
        self.previous_position_y = game_variables[POSITION_Y]
        self.previous_kills = game_variables[KILLS]

        return

    def _get_events(self, doom_state: vzd.GameState):
        game_variables = doom_state.game_variables

        events = []

        events_methods = [
            self.get_pickup_health_event,
            self.get_pickup_ammo_event,
            self.get_pickup_armour_event,
            self.get_shooting_event,
            self.get_killing_event,
            self.get_moving_event,
        ]

        for method in events_methods:
            event = method(game_variables)
            if event != 0:
                events.append(event)

        self.previous_health = game_variables[HEALTH]
        self.previous_armour = game_variables[ARMOUR]
        self.selected_weapon = game_variables[SELECTED_WEAPON]
        self.selected_weapon_ammo = game_variables[AMMO]
        self.previous_position_x = game_variables[POSITION_X]
        self.previous_position_y = game_variables[POSITION_Y]
        self.previous_kills = game_variables[KILLS]

        return events


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

    def get_moving_event(self, game_variables):
        position_x = game_variables[POSITION_X]
        position_y = game_variables[POSITION_Y]

        delta_x = position_x - self.previous_position_x
        delta_y = position_y - self.previous_position_y

        distance_squared = delta_x ** 2 + delta_y ** 2

        self.distance_moved_squared += distance_squared

        if (self.distance_moved_squared > (1. ** 2)):
            self.distance_moved_squared = 0
            return EventType.MOVEMENT.value
        
        return 0