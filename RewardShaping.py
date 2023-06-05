import vizdoom as vzd


class RewardShaping:
    def __init__(self) -> None:
        self._last_health = None
        self._last_position = None
        self._last_damage_count = None
        pass

    def get_reward(self, reward: float, doom_state: vzd.GameState) -> float:
        game_variables = doom_state.game_variables

        health = game_variables[0]
        armor = game_variables[1]
        selected_weapon = game_variables[2]
        selected_ammo = game_variables[3]
        damage_count = game_variables[4]
        position = [game_variables[5], game_variables[6]]

        return (
            reward
            + self._compute_distance_reward(position)
            + self._compute_health_reward(health)
            + self._compute_damage_reward(damage_count)
        )

    def _compute_distance_reward(self, position):
        if self._last_position is None:
            self._last_position = position
            return 0

        distance_vector = [
            position[0] - self._last_position[0],
            position[1] - self._last_position[1]
        ]

        self._last_position = position

        distance = distance_vector[0] ** 2 + distance_vector[1] ** 2

        if distance < 7.0:
            return -0.0025
        
        return 0.0005

    def _compute_health_reward(self, health):
        if self._last_health is None:
            self._last_health = health
            return 0

        health = max(health, 0)
        delta_health = health - self._last_health

        self._last_health = health

        health_reward = 0.02 * max(delta_health, 0)
        health_penalty = 0.01 * min(delta_health, 0)

        return health_reward + health_penalty

    def _compute_damage_reward(self, damage_count):
        if self._last_damage_count is None:
            self._last_damage_count = damage_count
            return 0

        last_damage = self._last_damage_count
        self._last_damage_count = damage_count

        return 0.1 * (damage_count - last_damage)
