# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Persistence helpers for Forceteki PSRO runs."""

import json
import os
from typing import Any

import numpy as np
import torch

from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy


MANIFEST_FILENAME = "manifest.json"
SOLVER_STATE_FILENAME = "solver_state.json"
RNG_STATE_FILENAME = "rng_state.pt"
POLICY_DIRNAME = "policies"
BOT_FILENAME = "bot_policy.json"
CONFIG_FLAG_NAMES = (
    "game_name",
    "n_players",
    "forceteki_seed",
    "max_episode_steps",
    "forceteki_worker_pool_size",
    "parallel_eval_workers",
    "meta_strategy_method",
    "gpsro_iterations",
    "sims_per_entry",
    "rollout_diagnostics",
    "number_policies_selected",
    "symmetric_game",
    "training_strategy_selector",
    "rectifier",
    "oracle_type",
    "number_training_episodes",
    "self_play_proportion",
    "hidden_layer_size",
    "n_hidden_layers",
    "batch_size",
    "sigma",
    "optimizer_str",
    "loss_str",
    "num_q_before_pi",
    "entropy_cost",
    "critic_learning_rate",
    "pi_learning_rate",
    "dqn_learning_rate",
    "update_target_network_every",
    "learn_every",
    "ppo_steps_per_batch",
    "ppo_num_minibatches",
    "ppo_update_epochs",
    "ppo_learning_rate",
    "ppo_gamma",
    "ppo_gae_lambda",
    "ppo_clip_coef",
    "ppo_entropy_coef",
    "ppo_value_coef",
    "ppo_max_grad_norm",
    "ppo_target_kl",
    "ppo_device",
    "ppo_intent_vocab_size",
    "ppo_kind_vocab_size",
    "ppo_control_vocab_size",
    "ppo_card_vocab_size",
    "seed",
    "verbose",
    "progress",
    "progress_interval_seconds",
    "log_to_file",
    "forceteki_log_dir",
    "log_file",
    "debug",
    "debug_dir",
    "output_dir",
    "resume_from",
    "init_policy_from",
    "decks_path",
    "player0_deck_path",
    "player1_deck_path",
)


def _json_default(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as json_file:
    json.dump(payload, json_file, indent=2, sort_keys=True, default=_json_default)
    json_file.write("\n")


def _read_json(path):
  with open(path, encoding="utf-8") as json_file:
    return json.load(json_file)


def _torch_load(path, map_location="cpu"):
  try:
    return torch.load(path, map_location=map_location, weights_only=False)
  except TypeError:
    return torch.load(path, map_location=map_location)


def flags_to_dict(flags_obj):
  """Returns a JSON-friendly snapshot of absl flag values."""
  return {
      name: getattr(flags_obj, name)
      for name in CONFIG_FLAG_NAMES
      if hasattr(flags_obj, name)
  }


def solver_state_dict(solver, completed_iterations):
  """Serializes the PSRO solver state needed to resume later."""
  return {
      "format": "forceteki_psro_solver_state_v1",
      "completed_iterations": int(completed_iterations),
      "meta_games": [game.tolist() for game in solver.get_meta_game()],
      "meta_strategies": [
          np.asarray(strategy).tolist()
          for strategy in solver.get_meta_strategies()
      ],
      "non_marginalized_meta_strategies": np.asarray(
          getattr(solver, "_non_marginalized_probabilities", [])
      ).tolist(),
      "policy_counts": [
          len(player_policies) for player_policies in solver.get_policies()
      ],
      "symmetric_game": bool(solver.symmetric_game),
      "iterations": int(getattr(solver, "_iterations", completed_iterations)),
  }


def _policy_path(output_dir, player_id, policy_index):
  return os.path.join(
      output_dir,
      POLICY_DIRNAME,
      f"player_{player_id}",
      f"policy_{policy_index}.pt")


def save_policy_population(output_dir, policies):
  """Saves a policy population and returns manifest entries for it."""
  entries = []
  for player_id, player_policies in enumerate(policies):
    player_entries = []
    for policy_index, policy_obj in enumerate(player_policies):
      if not isinstance(policy_obj, ForcetekiPPOPolicy):
        raise TypeError(
            "Forceteki PSRO artifact saving currently supports only "
            f"ForcetekiPPOPolicy, got {type(policy_obj).__name__}")
      path = _policy_path(output_dir, player_id, policy_index)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      torch.save(policy_obj.checkpoint(player_id, policy_index), path)
      player_entries.append({
          "player_id": player_id,
          "policy_index": policy_index,
          "class": "ForcetekiPPOPolicy",
          "path": os.path.relpath(path, output_dir),
      })
    entries.append(player_entries)
  return entries


def save_rng_state(output_dir):
  path = os.path.join(output_dir, RNG_STATE_FILENAME)
  torch.save({
      "numpy_random_state": np.random.get_state(),
      "torch_random_state": torch.get_rng_state(),
      "torch_cuda_random_state_all": (
          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
  }, path)
  return os.path.relpath(path, output_dir)


def restore_rng_state(run_dir):
  path = os.path.join(run_dir, RNG_STATE_FILENAME)
  if not os.path.exists(path):
    return
  state = _torch_load(path, map_location="cpu")
  if "numpy_random_state" in state:
    np.random.set_state(state["numpy_random_state"])
  if "torch_random_state" in state:
    torch.set_rng_state(state["torch_random_state"])
  cuda_state = state.get("torch_cuda_random_state_all")
  if cuda_state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(cuda_state)


def bot_policy_dict(solver_state):
  """Builds a simple highest-weight population policy descriptor."""
  bot_entries = []
  for player_id, strategy in enumerate(solver_state["meta_strategies"]):
    probabilities = np.asarray(strategy, dtype=float)
    policy_index = int(np.argmax(probabilities)) if probabilities.size else 0
    bot_entries.append({
        "player_id": player_id,
        "selection": "argmax_meta_strategy",
        "policy_index": policy_index,
        "probability": float(probabilities[policy_index])
                       if probabilities.size else 0.0,
    })
  return {
      "format": "forceteki_bot_policy_v1",
      "population_source": MANIFEST_FILENAME,
      "players": bot_entries,
  }


def save_run_artifacts(output_dir, solver, flags_obj, game_params,
                       completed_iterations):
  """Saves all artifacts for the current PSRO run."""
  os.makedirs(output_dir, exist_ok=True)
  policies = solver.get_policies()
  policy_entries = save_policy_population(output_dir, policies)
  solver_state = solver_state_dict(solver, completed_iterations)
  solver_state_path = os.path.join(output_dir, SOLVER_STATE_FILENAME)
  _write_json(solver_state_path, solver_state)
  rng_state_relpath = save_rng_state(output_dir)

  bot_policy = bot_policy_dict(solver_state)
  bot_policy_path = os.path.join(output_dir, BOT_FILENAME)
  _write_json(bot_policy_path, bot_policy)

  manifest = {
      "format": "forceteki_psro_run_v1",
      "output_dir": os.path.abspath(output_dir),
      "completed_iterations": int(completed_iterations),
      "flags": flags_to_dict(flags_obj),
      "game_params": game_params,
      "policy_population": policy_entries,
      "solver_state": os.path.relpath(solver_state_path, output_dir),
      "rng_state": rng_state_relpath,
      "bot_policy": os.path.relpath(bot_policy_path, output_dir),
  }
  manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
  _write_json(manifest_path, manifest)
  return manifest


def load_manifest(run_dir):
  return _read_json(os.path.join(run_dir, MANIFEST_FILENAME))


def load_solver_state(run_dir):
  return _read_json(os.path.join(run_dir, SOLVER_STATE_FILENAME))


def load_policy_population(run_dir, env, device=None, load_optimizer=True):
  """Loads all policies referenced by a run manifest."""
  manifest = load_manifest(run_dir)
  population = []
  for player_entries in manifest["policy_population"]:
    player_policies = []
    for entry in player_entries:
      checkpoint = _torch_load(
          os.path.join(run_dir, entry["path"]),
          map_location=device or "cpu")
      player_policies.append(ForcetekiPPOPolicy.from_checkpoint(
          env,
          checkpoint,
          device=device,
          load_optimizer=load_optimizer))
    population.append(player_policies)
  return population


def load_single_policy(policy_path, env, device=None, player_id=None,
                       load_optimizer=True):
  checkpoint = _torch_load(policy_path, map_location=device or "cpu")
  return ForcetekiPPOPolicy.from_checkpoint(
      env,
      checkpoint,
      player_id=player_id,
      device=device,
      load_optimizer=load_optimizer)


def restore_solver_state(solver, policies, solver_state):
  """Overwrites a freshly constructed solver with saved PSRO state."""
  solver._policies = policies  # pylint: disable=protected-access
  solver._new_policies = [[] for _ in policies]  # pylint: disable=protected-access
  solver._meta_games = [  # pylint: disable=protected-access
      np.asarray(meta_game, dtype=float)
      for meta_game in solver_state["meta_games"]
  ]
  solver._meta_strategy_probabilities = [  # pylint: disable=protected-access
      np.asarray(strategy, dtype=float)
      for strategy in solver_state["meta_strategies"]
  ]
  solver._non_marginalized_probabilities = np.asarray(  # pylint: disable=protected-access
      solver_state.get("non_marginalized_meta_strategies", []),
      dtype=float)
  solver._iterations = int(solver_state.get(  # pylint: disable=protected-access
      "iterations", solver_state.get("completed_iterations", 0)))


def flatten_run_path(run_dir, relative_path):
  return os.path.join(run_dir, relative_path)


def policy_action(state, policy_obj, rng=None):
  """Samples an action and returns both the action id and probability table."""
  probabilities = policy_obj.action_probabilities(state)
  if not probabilities:
    return None, probabilities
  actions, probs = zip(*sorted(probabilities.items()))
  probs = np.asarray(probs, dtype=float)
  probs /= probs.sum()
  if rng is None:
    action = int(actions[int(np.argmax(probs))])
  else:
    action = int(rng.choice(actions, p=probs))
  return action, {int(action): float(prob)
                  for action, prob in zip(actions, probs)}


def game_round_number(state):
  raw_state = getattr(state, "_state", {}).get("state", {})
  round_number = raw_state.get("roundNumber")
  return int(round_number) if round_number is not None else None


def json_load_file(path) -> Any:
  return _read_json(path)
