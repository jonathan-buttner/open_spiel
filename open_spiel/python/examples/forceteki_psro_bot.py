# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select an action from a saved Forceteki PSRO bot policy."""

import json
import os

from absl import app
from absl import flags
import numpy as np

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.examples import forceteki_psro_artifacts
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.games import forceteki


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", "",
                    "Forceteki PSRO artifact directory to load.")
flags.DEFINE_integer("player_id", 0, "OpenSpiel player id to act for.")
flags.DEFINE_integer("policy_index", -1,
                     "Policy index to use. Negative uses bot_policy.json.")
flags.DEFINE_bool("sample", False,
                  "Sample from the policy. False chooses max probability.")
flags.DEFINE_integer("seed", 1, "Sampling seed.")
flags.DEFINE_string("simulation_checkpoint", "",
                    "Optional Forceteki simulation checkpoint JSON file.")
flags.DEFINE_string("decks_path", "",
                    "Optional JSON file containing two Forceteki decklists.")
flags.DEFINE_string("player0_deck_path", "",
                    "Optional player 0 decklist JSON file.")
flags.DEFINE_string("player1_deck_path", "",
                    "Optional player 1 decklist JSON file.")
flags.DEFINE_integer("max_episode_steps", 1000,
                     "OpenSpiel-side cap for Forceteki rollout length.")
flags.DEFINE_string("ppo_device", "cpu", "Torch device used for policy load.")


def _game_params():
  params = {
      "players": 2,
      "max_game_length": FLAGS.max_episode_steps,
      "seed": str(FLAGS.seed),
  }
  if FLAGS.decks_path:
    params["decks_path"] = FLAGS.decks_path
  if FLAGS.player0_deck_path:
    params["player0_deck_path"] = FLAGS.player0_deck_path
  if FLAGS.player1_deck_path:
    params["player1_deck_path"] = FLAGS.player1_deck_path
  return params


def _policy_index_from_bot(run_dir, player_id):
  bot_path = os.path.join(run_dir, forceteki_psro_artifacts.BOT_FILENAME)
  with open(bot_path, encoding="utf-8") as bot_file:
    bot_policy = json.load(bot_file)
  for player_entry in bot_policy["players"]:
    if int(player_entry["player_id"]) == player_id:
      return int(player_entry["policy_index"])
  raise ValueError(f"No bot policy entry for player {player_id}")


def _load_state(game, params):
  if not FLAGS.simulation_checkpoint:
    return game.new_initial_state()
  with open(FLAGS.simulation_checkpoint, encoding="utf-8") as checkpoint_file:
    checkpoint = json.load(checkpoint_file)
  return forceteki.ForcetekiState(game, params, checkpoint=checkpoint)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.checkpoint_dir:
    raise app.UsageError("--checkpoint_dir is required")
  if FLAGS.player_id not in (0, 1):
    raise app.UsageError("--player_id must be 0 or 1")
  if bool(FLAGS.player0_deck_path) != bool(FLAGS.player1_deck_path):
    raise app.UsageError(
        "--player0_deck_path and --player1_deck_path must be provided together")

  params = _game_params()
  game = pyspiel.load_game("python_forceteki_swu", params)
  env = rl_environment.Environment(game)
  population = forceteki_psro_artifacts.load_policy_population(
      FLAGS.checkpoint_dir, env, device=FLAGS.ppo_device, load_optimizer=False)
  _close_state(getattr(env, "_state", None))

  policy_index = FLAGS.policy_index
  if policy_index < 0:
    policy_index = _policy_index_from_bot(FLAGS.checkpoint_dir, FLAGS.player_id)
  policy_obj = population[FLAGS.player_id][policy_index]

  state = _load_state(game, params)
  try:
    current_player = state.current_player()
    if current_player != FLAGS.player_id:
      raise app.UsageError(
          f"State current_player={current_player}; requested "
          f"--player_id={FLAGS.player_id}")
    rng = np.random.default_rng(FLAGS.seed) if FLAGS.sample else None
    action, probabilities = forceteki_psro_artifacts.policy_action(
        state, policy_obj, rng=rng)
    legal_action = (
        state.forceteki_legal_action(action) if action is not None else None)
    print(json.dumps({
        "player_id": FLAGS.player_id,
        "policy_index": policy_index,
        "action": action,
        "action_probabilities": probabilities,
        "forceteki_action": legal_action,
    }, indent=2, sort_keys=True))
  finally:
    _close_state(state)
    forceteki.close_all_workers()


if __name__ == "__main__":
  app.run(main)
