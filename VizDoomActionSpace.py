import itertools
import typing as t

import numpy as np
from vizdoom import Button

# Buttons that cannot be used together
MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
]

# Buttons that can only be used alone.
EXCLUSIVE_BUTTONS = [
    Button.ATTACK,

    Button.SELECT_PREV_WEAPON, Button.SELECT_NEXT_WEAPON,

    Button.SELECT_WEAPON0, Button.SELECT_WEAPON1, Button.SELECT_WEAPON2,
    Button.SELECT_WEAPON3, Button.SELECT_WEAPON4, Button.SELECT_WEAPON5,
    Button.SELECT_WEAPON6, Button.SELECT_WEAPON7, Button.SELECT_WEAPON8,
    Button.SELECT_WEAPON9
]


def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)

    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)


def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    mutual_exclusion_mask = np.array([np.isin(buttons, excluded_group)
                                      for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

    return np.any(np.sum(
        (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
        axis=-1) > 1, axis=-1)


def get_available_actions(buttons: np.array) -> t.List[t.List[float]]:
    action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

    illegal_mask = (has_excluded_pair(action_combinations, buttons)
                    | has_exclusive_button(action_combinations, buttons))

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]

    # print('Built action space of size {} from buttons {}'.format(len(possible_actions), buttons))
    return possible_actions.tolist()
