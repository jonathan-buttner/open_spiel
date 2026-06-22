# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""RL oracles for the Forceteki PSRO example."""

from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.games import forceteki


class ForcetekiTraceRLOracle(rl_oracle.RLOracle):
  """RL oracle that marks base PG/DQN training rollouts in Forceteki traces."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._forceteki_trace_training_rollout = 0
    self._progress_reporter = None
    self._progress_iteration = None
    self._progress_total_iterations = None

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
    _close_state(getattr(self._env, "_state", None))
    self._env._state = None  # pylint: disable=protected-access
    try:
      with forceteki.forceteki_trace_context(
          rolloutType="training",
          trainingRollout=self._forceteki_trace_training_rollout):
        return super()._rollout(
            game, agents, **oracle_specific_execution_kwargs)
    finally:
      _close_state(getattr(self._env, "_state", None))
      self._env._state = None  # pylint: disable=protected-access

  def _training_progress(self, episodes_per_oracle):
    if not self._progress_enabled():
      return
    total = episodes_per_oracle.size * (int(self._number_training_episodes) + 1)
    current = int(episodes_per_oracle.sum())
    display_current = min(current, total) if total > 0 else current
    self._progress_reporter.update(
        "training",
        "episodes",
        display_current,
        total,
        force=current >= total,
        iteration=f"{self._progress_iteration}/"
        f"{self._progress_total_iterations}")

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
    self._forceteki_trace_training_rollout += 1
    state = game.new_initial_state()
    live_agents = [
        agent for agent in agents
        if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen()
    ]

    try:
      with forceteki.forceteki_trace_context(
          rolloutType="training",
          trainingRollout=self._forceteki_trace_training_rollout):
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
    finally:
      _close_state(state)
