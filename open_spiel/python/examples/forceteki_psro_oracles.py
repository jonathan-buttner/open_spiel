# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""RL oracles for the Forceteki PSRO example."""

import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from datetime import timezone

import numpy as np

from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.games import forceteki


class ForcetekiTraceRLOracle(rl_oracle.RLOracle):
  """RL oracle that marks base PG/DQN training rollouts in Forceteki traces."""

  def __init__(self, *args, seed=1, crash_retry_limit=10, **kwargs):
    super().__init__(*args, **kwargs)
    self._seed = int(seed)
    self._forceteki_crash_retry_limit = int(crash_retry_limit)
    self._forceteki_crash_streak = 0
    self._forceteki_crash_debug_dir = ""
    self._forceteki_crash_output = None
    self._forceteki_last_failed_trace_path = ""
    self._forceteki_trace_training_rollout = 0
    self._progress_reporter = None
    self._progress_iteration = None
    self._progress_total_iterations = None

  def set_crash_recovery_context(self, debug_dir="", output=None):
    self._forceteki_crash_debug_dir = str(debug_dir or "")
    self._forceteki_crash_output = output

  def set_progress_reporter(self, progress_reporter):
    self._progress_reporter = progress_reporter

  def set_progress_context(self, iteration, total_iterations):
    self._progress_iteration = iteration
    self._progress_total_iterations = total_iterations

  def __call__(self, *args, **kwargs):
    if self._progress_enabled():
      self._progress_reporter.start(
          "training",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}")
    new_policies = super().__call__(*args, **kwargs)
    self._after_training(new_policies)
    if self._progress_enabled():
      self._progress_reporter.done(
          "training",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}")
    return new_policies

  def _after_training(self, new_policies):
    del new_policies

  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    self._forceteki_trace_training_rollout += 1
    rollout = self._forceteki_trace_training_rollout
    attempt = 0
    while True:
      attempt += 1
      self._forceteki_last_failed_trace_path = ""
      try:
        with forceteki.forceteki_trace_context(
            rolloutType="training",
            seed=self._seed,
            trainingRollout=rollout,
            rolloutAttempt=attempt):
          result = self._run_training_rollout_once(
              game, agents, **oracle_specific_execution_kwargs)
        self._forceteki_crash_streak = 0
        return result
      except Exception as exc:
        if self._forceteki_crash_retry_limit <= 0:
          raise
        self._forceteki_crash_streak += 1
        artifact_dir = self._save_rollout_crash_artifact(
            exc, rollout=rollout, attempt=attempt)
        if self._forceteki_crash_streak >= self._forceteki_crash_retry_limit:
          self._print_crash_recovery_line(
              "aborting",
              rollout,
              attempt,
              artifact_dir,
              exc)
          raise
        self._print_crash_recovery_line(
            "retrying",
            rollout,
            attempt,
            artifact_dir,
            exc)

  def _run_training_rollout_once(
      self, game, agents, **oracle_specific_execution_kwargs):
    _close_state(getattr(self._env, "_state", None))
    self._env._state = None  # pylint: disable=protected-access
    try:
      return super()._rollout(
          game, agents, **oracle_specific_execution_kwargs)
    except Exception:
      self._remember_failed_trace_path(getattr(self._env, "_state", None))
      raise
    finally:
      _close_state(getattr(self._env, "_state", None))
      self._env._state = None  # pylint: disable=protected-access

  def _remember_failed_trace_path(self, state):
    self._forceteki_last_failed_trace_path = str(
        getattr(state, "_trace_path", "") or "")

  def _save_rollout_crash_artifact(self, exc, rollout, attempt):
    trace_path = self._forceteki_last_failed_trace_path
    debug_dir = self._forceteki_crash_debug_dir
    if not debug_dir and trace_path:
      debug_dir = os.path.dirname(trace_path)
    if not debug_dir:
      return ""

    error_dir = os.path.join(
        debug_dir,
        "errors",
        f"{_timestamp_slug()}_training-rollout-{rollout}_attempt-{attempt}")
    os.makedirs(error_dir, exist_ok=True)
    stack_trace_path = os.path.join(error_dir, "stack_trace.txt")
    trace_copy_path = os.path.join(error_dir, "trace.ndjson")

    with open(stack_trace_path, "w", encoding="utf-8") as stack_trace_file:
      stack_trace_file.write("".join(
          traceback.format_exception(type(exc), exc, exc.__traceback__)))
    if trace_path and os.path.exists(trace_path):
      shutil.copyfile(trace_path, trace_copy_path)
    else:
      with open(trace_copy_path, "w", encoding="utf-8") as trace_file:
        json.dump(
            {"traceUnavailable": True, "sourceTracePath": trace_path},
            trace_file,
            sort_keys=True)
        trace_file.write("\n")
    return error_dir

  def _print_crash_recovery_line(self, status, rollout, attempt, artifact_dir,
                                 exc):
    output = self._forceteki_crash_output or sys.stderr
    artifact = artifact_dir or "unavailable"
    print(
        "Forceteki training rollout crash "
        f"{status}: rollout={rollout} attempt={attempt} "
        f"consecutive_crashes={self._forceteki_crash_streak}/"
        f"{self._forceteki_crash_retry_limit} artifact_dir={artifact} "
        f"error={exc}",
        file=output,
        flush=True)

  def _training_progress(self, episodes_per_oracle):
    if not self._progress_enabled():
      return
    target_per_policy = int(self._number_training_episodes) + 1
    total = episodes_per_oracle.size * target_per_policy
    capped_episodes = np.minimum(episodes_per_oracle, target_per_policy)
    current = int(capped_episodes.sum())
    completed_policies = int(np.sum(episodes_per_oracle >= target_per_policy))
    self._progress_reporter.update(
        "training",
        "episodes",
        current,
        total,
        force=completed_policies == episodes_per_oracle.size,
        iteration=f"{self._progress_iteration}/"
        f"{self._progress_total_iterations}",
        completed_policies=f"{completed_policies}/{episodes_per_oracle.size}")

  def _progress_enabled(self):
    return (self._progress_reporter is not None and
            self._progress_reporter.enabled)


class ForcetekiPPOOracle(ForcetekiTraceRLOracle):
  """PSRO oracle that trains factored Forceteki PPO responders."""

  def _after_training(self, new_policies):
    for player_policies in new_policies:
      for pol in player_policies:
        if isinstance(pol, ForcetekiPPOPolicy):
          pol.finish_training()

  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    del oracle_specific_execution_kwargs
    return super()._rollout(game, agents)

  def _run_training_rollout_once(
      self, game, agents, **oracle_specific_execution_kwargs):
    del oracle_specific_execution_kwargs
    state = None
    live_agents = [
        agent for agent in agents
        if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen()
    ]
    snapshots = self._snapshot_ppo_agents(live_agents)

    try:
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes, probs = zip(*state.chance_outcomes())
          state.apply_action(utils.random_choice(outcomes, probs))
          continue

        player = state.current_player()
        agent = agents[player]
        if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen():
          action = agent.training_action(state)
        else:
          action_probs = agent(state, player)
          outcomes, probs = zip(*action_probs.items())
          action = utils.random_choice(outcomes, probs)

        state.apply_action(action)
        rewards = state.returns() if state.is_terminal() else state.rewards()
        if not rewards:
          rewards = [0.0] * state.num_players()
        for live_agent in live_agents:
          live_agent.add_pending_reward(rewards[live_agent.player_id])

      for live_agent in live_agents:
        live_agent.finish_episode()
    except Exception:
      self._restore_ppo_agent_snapshots(snapshots)
      self._remember_failed_trace_path(state)
      raise
    finally:
      _close_state(state)

  def _snapshot_ppo_agents(self, agents):
    return {
        agent: (
            getattr(agent, "_pending", None),
            getattr(agent, "_pending_reward", 0.0),
            list(getattr(agent, "_buffer", [])),
        )
        for agent in agents
    }

  def _restore_ppo_agent_snapshots(self, snapshots):
    for agent, (pending, pending_reward, buffer) in snapshots.items():
      agent._pending = pending  # pylint: disable=protected-access
      agent._pending_reward = pending_reward  # pylint: disable=protected-access
      agent._buffer = list(buffer)  # pylint: disable=protected-access


def _timestamp_slug():
  timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
  return timestamp.replace("+00:00", "Z").replace(":", "-")
