# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Evaluate saved Forceteki PSRO policies on deck matchups."""

import csv
import json
import os

from absl import app
from absl import flags
import numpy as np

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.examples import forceteki_psro_artifacts
from open_spiel.python.examples.forceteki_psro_utils import _close_state

# Registers python_forceteki_swu.
from open_spiel.python.games import forceteki  # pylint: disable=unused-import


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", "",
                    "Forceteki PSRO artifact directory to evaluate.")
flags.DEFINE_string("deck_a", "", "JSON decklist file for deck A.")
flags.DEFINE_string("deck_b", "", "JSON decklist file for deck B.")
flags.DEFINE_integer("games", 100, "Games per seat configuration.")
flags.DEFINE_bool("swap_seats", True,
                  "Also evaluate deck B as player 0 and deck A as player 1.")
flags.DEFINE_integer("seed", 1, "Evaluation RNG seed.")
flags.DEFINE_integer("max_episode_steps", 1000,
                     "OpenSpiel-side cap for Forceteki rollout length.")
flags.DEFINE_integer("worker_pool_size", 0,
                     "Max reusable Forceteki Node workers. Zero disables "
                     "worker pooling.")
flags.DEFINE_string("ppo_device", "cpu", "Torch device used for policy load.")
flags.DEFINE_string("output_json", "",
                    "Optional JSON file to write matchup results.")
flags.DEFINE_string("output_csv", "",
                    "Optional CSV file to write per-game results.")


def _game_params(deck0, deck1, seed):
  return {
      "players": 2,
      "max_game_length": FLAGS.max_episode_steps,
      "worker_pool_size": FLAGS.worker_pool_size,
      "player0_deck_path": deck0,
      "player1_deck_path": deck1,
      "seed": str(seed),
  }


def _sample_policy_index(probabilities, rng):
  probabilities = np.asarray(probabilities, dtype=float)
  if probabilities.size == 0:
    return 0
  probabilities = probabilities / probabilities.sum()
  return int(rng.choice(np.arange(probabilities.size), p=probabilities))


def _rollout(game, policies, rng):
  state = game.new_initial_state()
  try:
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = int(rng.choice(outcomes, p=np.asarray(probs) / np.sum(probs)))
      else:
        player = state.current_player()
        policy_obj = policies[player]
        action_probs = policy_obj.action_probabilities(state)
        actions, probs = zip(*sorted(action_probs.items()))
        probs = np.asarray(probs, dtype=float)
        action = int(rng.choice(actions, p=probs / probs.sum()))
      state.apply_action(action)

    returns = [float(value) for value in state.returns()]
    terminal_reason = getattr(
        state, "forceteki_terminal_reason",
        lambda: "unknown_terminal")()
    action_count = int(getattr(
        state, "forceteki_move_number", lambda: state.move_number())())
    round_count = forceteki_psro_artifacts.game_round_number(state)
    return returns, terminal_reason, action_count, round_count
  finally:
    _close_state(state)


def _evaluate_seats(population, meta_strategies, deck0, deck1, label0, label1,
                    game_offset, rng):
  rows = []
  for game_index in range(FLAGS.games):
    game_seed = f"{FLAGS.seed}-{game_offset + game_index}"
    game = pyspiel.load_game_as_turn_based(
        "python_forceteki_swu", _game_params(deck0, deck1, game_seed))
    env = rl_environment.Environment(game)
    _close_state(getattr(env, "_state", None))
    del env

    policy_indices = [
        _sample_policy_index(meta_strategies[player], rng)
        for player in range(2)
    ]
    policies = [
        population[player][policy_indices[player]]
        for player in range(2)
    ]
    returns, terminal_reason, action_count, round_count = _rollout(
        game, policies, rng)

    deck_a_player = 0 if label0 == "A" else 1
    deck_a_return = returns[deck_a_player]
    rows.append({
        "game_index": game_offset + game_index,
        "seed": game_seed,
        "player0_deck": label0,
        "player1_deck": label1,
        "player0_policy_index": policy_indices[0],
        "player1_policy_index": policy_indices[1],
        "player0_return": returns[0],
        "player1_return": returns[1],
        "deck_a_return": deck_a_return,
        "deck_a_win": deck_a_return > 0,
        "terminal_reason": terminal_reason,
        "action_count": action_count,
        "round_count": round_count,
      })
  return rows


def _summary(rows):
  deck_a_returns = np.asarray([row["deck_a_return"] for row in rows], dtype=float)
  action_counts = np.asarray([row["action_count"] for row in rows], dtype=float)
  round_counts = np.asarray(
      [row["round_count"] for row in rows if row["round_count"] is not None],
      dtype=float)
  terminal_counts = {}
  for row in rows:
    terminal_counts[row["terminal_reason"]] = (
        terminal_counts.get(row["terminal_reason"], 0) + 1)
  return {
      "games": len(rows),
      "deck_a_win_rate": float(np.mean(deck_a_returns > 0))
                         if len(rows) else 0.0,
      "deck_a_mean_return": float(np.mean(deck_a_returns))
                            if len(rows) else 0.0,
      "terminal_reasons": terminal_counts,
      "average_total_actions": float(np.mean(action_counts))
                               if len(rows) else 0.0,
      "average_rounds": float(np.mean(round_counts))
                        if round_counts.size else None,
  }


def _write_csv(path, rows):
  if not path:
    return
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _write_json(path, payload):
  if not path:
    return
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8") as json_file:
    json.dump(payload, json_file, indent=2, sort_keys=True)
    json_file.write("\n")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.checkpoint_dir:
    raise app.UsageError("--checkpoint_dir is required")
  if not FLAGS.deck_a or not FLAGS.deck_b:
    raise app.UsageError("--deck_a and --deck_b are required")
  if FLAGS.games < 1:
    raise app.UsageError("--games must be at least 1")

  base_game = pyspiel.load_game_as_turn_based(
      "python_forceteki_swu",
      _game_params(FLAGS.deck_a, FLAGS.deck_b, f"{FLAGS.seed}-loader"))
  env = rl_environment.Environment(base_game)
  population = forceteki_psro_artifacts.load_policy_population(
      FLAGS.checkpoint_dir,
      env,
      device=FLAGS.ppo_device,
      load_optimizer=False)
  _close_state(getattr(env, "_state", None))

  solver_state = forceteki_psro_artifacts.load_solver_state(FLAGS.checkpoint_dir)
  meta_strategies = solver_state["meta_strategies"]
  rng = np.random.default_rng(FLAGS.seed)

  rows = _evaluate_seats(
      population, meta_strategies, FLAGS.deck_a, FLAGS.deck_b, "A", "B", 0, rng)
  if FLAGS.swap_seats:
    rows.extend(_evaluate_seats(
        population,
        meta_strategies,
        FLAGS.deck_b,
        FLAGS.deck_a,
        "B",
        "A",
        FLAGS.games,
        rng))

  payload = {
      "checkpoint_dir": os.path.abspath(FLAGS.checkpoint_dir),
      "deck_a": os.path.abspath(FLAGS.deck_a),
      "deck_b": os.path.abspath(FLAGS.deck_b),
      "summary": _summary(rows),
      "games": rows,
  }
  print(json.dumps(payload["summary"], indent=2, sort_keys=True))
  _write_json(FLAGS.output_json, payload)
  if FLAGS.output_csv:
    _write_csv(FLAGS.output_csv, rows)
  forceteki.close_all_workers()


if __name__ == "__main__":
  app.run(main)
