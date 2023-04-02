import random
import time

from rich.progress import track

import os.path

import vizdoom
import vizdoom as vzd

import torch
import gymnasium as gym

import numpy as np

import json

with open("settings.json") as file:
    settings = json.load(file)


def load_scenario(game):
    scenario_name = settings["scenario"]
    if settings["isVizdoomScenario"]:
        scenario_path = os.path.join(vzd.scenarios_path, scenario_name)
    else:
        scenario_path = scenario_name
    game.load_config(scenario_path)


def main():
    game = vizdoom.DoomGame()
    game.set_window_visible(False)

    load_scenario(game)
    game.init()

    available_actions = np.identity(3, dtype=np.uint8)

    epochs: int = settings["epochs"]
    steps_per_epoch: int = settings["stepsPerEpoch"]

    for epoch in track(range(epochs)):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            info = state.game_variables
            reward = game.make_action(random.choice(available_actions), 4)

    result = game.get_total_reward()
    print(f"Total reward: {result}")


if __name__ == "__main__":
    main()
