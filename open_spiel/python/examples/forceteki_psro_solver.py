# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Diagnostic PSRO solver for the Forceteki PSRO example."""

import concurrent.futures
import contextlib
import hashlib
import itertools
import random
import sys
import threading

import numpy as np

from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.games import forceteki


def _random_choice(outcomes, probabilities, rng):
  cumsum = np.cumsum(probabilities)
  return outcomes[np.searchsorted(cumsum / cumsum[-1], rng.random())]


def _rollout_seed(base_seed, profile_index, simulation_index):
  encoded = repr((int(base_seed), tuple(profile_index), int(simulation_index)))
  digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
  return int(digest[:16], 16)


def _sample_episode_with_diagnostics(state, policies, rng=None,
                                     policy_locks=None):
  """Samples one episode and returns final returns plus rollout diagnostics."""
  rng = rng or random
  policy_locks = policy_locks or {}
  while not state.is_terminal():
    if state.is_simultaneous_node():
      actions = [None] * state.num_players()
      for player in range(state.num_players()):
        policy = policies[player]
        with policy_locks.get(id(policy), contextlib.nullcontext()):
          state_policy = policy(state, player)
        outcomes, probs = zip(*state_policy.items())
        actions[player] = _random_choice(outcomes, probs, rng)
      state.apply_actions(actions)
      continue

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
    else:
      player = state.current_player()
      policy = policies[player]
      with policy_locks.get(id(policy), contextlib.nullcontext()):
        state_policy = policy(state)
      outcomes, probs = zip(*state_policy.items())

    state.apply_action(_random_choice(outcomes, probs, rng))

  returns = np.array(state.returns(), dtype=np.float32)
  reason = getattr(
      state, "forceteki_terminal_reason",
      lambda: "unknown_terminal")()
  move_number = getattr(
      state, "forceteki_move_number",
      lambda: state.move_number())()
  return returns, reason, int(move_number)


class DiagnosticPSROSolver(psro_v2.PSROSolver):
  """PSRO solver that can print ForceTeki rollout diagnostics per meta entry."""

  def __init__(self, *args, rollout_diagnostics=False, parallel_eval_workers=1,
               seed=1, progress_reporter=None, output=None, **kwargs):
    self._rollout_diagnostics = rollout_diagnostics
    self._parallel_eval_workers = max(1, int(parallel_eval_workers))
    self._seed = int(seed)
    self._progress_reporter = progress_reporter
    self._output = output or sys.stdout
    self._progress_iteration = None
    self._progress_total_iterations = None
    self._evaluation_rollouts_done = 0
    self._evaluation_rollouts_total = 0
    super().__init__(*args, **kwargs)

  def set_progress_context(self, iteration, total_iterations):
    self._progress_iteration = iteration
    self._progress_total_iterations = total_iterations

  def update_empirical_gamestate(self, seed=None):
    if not self._rollout_diagnostics and not self._progress_enabled():
      return super().update_empirical_gamestate(seed=seed)

    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    if self.symmetric_game:
      self._policies = self._game_num_players * self._policies
      self._new_policies = self._game_num_players * self._new_policies
      self._num_players = self._game_num_players

    updated_policies = [
        self._policies[k] + self._new_policies[k]
        for k in range(self._num_players)
    ]
    total_number_policies = [
        len(updated_policies[k]) for k in range(self._num_players)
    ]
    number_older_policies = [
        len(self._policies[k]) for k in range(self._num_players)
    ]
    number_new_policies = [
        len(self._new_policies[k]) for k in range(self._num_players)
    ]

    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(self._num_players)
    ]

    older_policies_slice = tuple(
        [slice(len(self._policies[k])) for k in range(self._num_players)])
    for k in range(self._num_players):
      meta_games[k][older_policies_slice] = self._meta_games[k]

    self._evaluation_rollouts_done = 0
    self._evaluation_rollouts_total = (
        self._count_missing_profiles(meta_games, total_number_policies,
                                     number_older_policies,
                                     number_new_policies) *
        self._sims_per_entry)
    if self._progress_enabled():
      self._progress_reporter.start(
          "evaluation",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}",
          rollouts=f"0/{self._evaluation_rollouts_total}")

    for current_player in range(self._num_players):
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, self._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        used_tuple = tuple(used_index)
        if not np.isnan(meta_games[current_player][used_tuple]):
          continue

        estimated_policies = [
            updated_policies[k][current_index[k]]
            for k in range(current_player)
        ] + [
            self._new_policies[current_player][current_index[current_player]]
        ] + [
            updated_policies[k][current_index[k]]
            for k in range(current_player + 1, self._num_players)
        ]

        utility_estimates = self._sample_episodes_with_diagnostics(
            estimated_policies, self._sims_per_entry, used_tuple)

        if self.symmetric_game:
          player_permutations = list(itertools.permutations(
              list(range(self._num_players))))
          for permutation in player_permutations:
            permuted_tuple = tuple([used_index[i] for i in permutation])
            for player in range(self._num_players):
              if np.isnan(meta_games[player][permuted_tuple]):
                meta_games[player][permuted_tuple] = 0.0
              meta_games[player][permuted_tuple] += (
                  utility_estimates[permutation[player]] /
                  len(player_permutations))
        else:
          for k in range(self._num_players):
            meta_games[k][used_tuple] = utility_estimates[k]

    if self._progress_enabled():
      self._progress_reporter.done(
          "evaluation",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}",
          rollouts=f"{self._evaluation_rollouts_done}/"
          f"{self._evaluation_rollouts_total}")

    if self.symmetric_game:
      self._policies = [self._policies[0]]
      self._new_policies = [self._new_policies[0]]
      updated_policies = [updated_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    return meta_games

  def _sample_episodes_with_diagnostics(self, policies, num_episodes,
                                        profile_index):
    totals = np.zeros(self._num_players)
    reason_counts = {
        "forceteki_terminal": 0,
        "open_spiel_cap": 0,
        "non_terminal": 0,
        "unknown_terminal": 0,
    }
    move_numbers = []
    nonzero_returns = 0
    policy_locks = {id(policy): threading.Lock() for policy in policies}

    if self._parallel_eval_workers > 1 and num_episodes > 1:
      max_workers = min(self._parallel_eval_workers, num_episodes)
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max_workers) as executor:
        results = executor.map(
            lambda sim_index: self._sample_one_episode_with_diagnostics(
                policies, profile_index, sim_index, policy_locks),
            range(num_episodes))
        for returns, reason, move_number in results:
          totals += returns.reshape(-1)
          reason_counts[reason] = reason_counts.get(reason, 0) + 1
          move_numbers.append(move_number)
          if np.any(returns != 0):
            nonzero_returns += 1
          self._evaluation_progress(profile_index)
    else:
      for sim_index in range(num_episodes):
        returns, reason, move_number = self._sample_one_episode_with_diagnostics(
            policies, profile_index, sim_index, policy_locks)
        totals += returns.reshape(-1)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        move_numbers.append(move_number)
        if np.any(returns != 0):
          nonzero_returns += 1
        self._evaluation_progress(profile_index)

    averages = totals / num_episodes
    if self._rollout_diagnostics:
      self._print_rollout_diagnostics(
          profile_index, num_episodes, averages, reason_counts, nonzero_returns,
          move_numbers)
    return averages

  def _count_missing_profiles(self, meta_games, total_number_policies,
                              number_older_policies, number_new_policies):
    count = 0
    for current_player in range(self._num_players):
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, self._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        if np.isnan(meta_games[current_player][tuple(used_index)]):
          count += 1
    return count

  def _evaluation_progress(self, profile_index):
    if not self._progress_enabled():
      return
    self._evaluation_rollouts_done += 1
    self._progress_reporter.update(
        "evaluation",
        "rollouts",
        self._evaluation_rollouts_done,
        self._evaluation_rollouts_total,
        force=self._evaluation_rollouts_done >= self._evaluation_rollouts_total,
        iteration=f"{self._progress_iteration}/"
        f"{self._progress_total_iterations}",
        profile=profile_index)

  def _progress_enabled(self):
    return (self._progress_reporter is not None and
            self._progress_reporter.enabled)

  def _sample_one_episode_with_diagnostics(self, policies, profile_index,
                                           sim_index, policy_locks):
    rng = random.Random(_rollout_seed(self._seed, profile_index, sim_index))
    with forceteki.forceteki_trace_context(
        rolloutType="evaluation",
        seed=self._seed,
        profileIndex=list(profile_index),
        simulationIndex=sim_index):
      state = self._game.new_initial_state()
      try:
        return _sample_episode_with_diagnostics(
            state, policies, rng=rng, policy_locks=policy_locks)
      finally:
        _close_state(state)

  def _print_rollout_diagnostics(self, profile_index, num_episodes,
                                 averages, reason_counts, nonzero_returns,
                                 move_numbers):
    if move_numbers:
      avg_steps = float(np.mean(move_numbers))
      min_steps = min(move_numbers)
      max_steps = max(move_numbers)
      step_summary = f"{avg_steps:.1f}/{min_steps}/{max_steps}"
    else:
      step_summary = "nan/nan/nan"

    print(
        "Rollout diagnostics "
        f"profile={profile_index} sims={num_episodes} "
        f"avg_returns={averages.tolist()} "
        f"forceteki_terminal={reason_counts.get('forceteki_terminal', 0)} "
        f"open_spiel_cap={reason_counts.get('open_spiel_cap', 0)} "
        f"non_terminal={reason_counts.get('non_terminal', 0)} "
        f"unknown_terminal={reason_counts.get('unknown_terminal', 0)} "
        f"nonzero_returns={nonzero_returns} "
        f"steps(avg/min/max)={step_summary}",
        file=self._output,
        flush=True)
