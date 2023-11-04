from typing import Any

from gymnasium.core import ActType

from VizDoomEnv import VizDoomEnv


class VizDoomBotsEnv(VizDoomEnv):
    def __init__(
            self,
            scenario: str,
            frame_skip=10,
            resolution=(160, 120),
            is_window_visible=False,
            doom_skill=-1,
            reward_shaping=None,
            memory_size=1,
            n_bots=3):

        game_args = '-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' + \
                         '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1'
        self.n_bots = n_bots

        super().__init__(
            scenario,
            frame_skip,
            resolution,
            is_window_visible,
            doom_skill,
            reward_shaping,
            memory_size,
            game_args=game_args)

    def _setup_game(self):
        super()._setup_game()

    def step(self, action: ActType):
        self._respawn_if_dead()

        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._reset_bots()
        return super().reset(seed=seed, options=options)

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()

    def _reset_bots(self):
        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')
