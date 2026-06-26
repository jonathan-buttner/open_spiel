# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""RL oracles for the Forceteki PSRO example."""

import concurrent.futures
from dataclasses import dataclass

import numpy as np

from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.games import forceteki


@dataclass(frozen=True)
class _ParallelRolloutResult:
  indexes: tuple[tuple[int, int], ...]
  transitions_by_index: dict[tuple[int, int], list[dict]]


class ForcetekiTraceRLOracle(rl_oracle.RLOracle):
  """RL oracle that marks base PG/DQN training rollouts in Forceteki traces."""

  def __init__(self, *args, seed=1, **kwargs):
    super().__init__(*args, **kwargs)
    self._seed = int(seed)
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
          seed=self._seed,
          trainingRollout=self._forceteki_trace_training_rollout):
        return super()._rollout(
            game, agents, **oracle_specific_execution_kwargs)
    finally:
      _close_state(getattr(self._env, "_state", None))
      self._env._state = None  # pylint: disable=protected-access

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

  def __init__(self, *args, parallel_training_workers=1, **kwargs):
    super().__init__(*args, **kwargs)
    self._parallel_training_workers = max(1, int(parallel_training_workers))

  def __call__(self,
               game,
               training_parameters,
               strategy_sampler=utils.sample_strategy,
               **oracle_specific_execution_kwargs):
    if self._parallel_training_workers <= 1:
      return super().__call__(
          game,
          training_parameters,
          strategy_sampler=strategy_sampler,
          **oracle_specific_execution_kwargs)

    if self._progress_enabled():
      self._progress_reporter.start(
          "training",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}")
    new_policies = self._parallel_call(
        game,
        training_parameters,
        strategy_sampler=strategy_sampler,
        **oracle_specific_execution_kwargs)
    self._after_training(new_policies)
    if self._progress_enabled():
      self._progress_reporter.done(
          "training",
          iteration=f"{self._progress_iteration}/"
          f"{self._progress_total_iterations}")
    return new_policies

  def _after_training(self, new_policies):
    for player_policies in new_policies:
      for pol in player_policies:
        if isinstance(pol, ForcetekiPPOPolicy):
          pol.finish_training()

  def _parallel_call(self,
                     game,
                     training_parameters,
                     strategy_sampler=utils.sample_strategy,
                     **oracle_specific_execution_kwargs):
    del oracle_specific_execution_kwargs
    episodes_per_oracle = [[0
                            for _ in range(len(player_params))]
                           for player_params in training_parameters]
    episodes_per_oracle = np.array(episodes_per_oracle)
    scheduled_episodes = np.array(episodes_per_oracle, copy=True)
    new_policies = self.generate_new_policies(training_parameters)
    for player, player_policies in enumerate(new_policies):
      for policy_index, policy in enumerate(player_policies):
        setattr(policy, "_oracle_policy_index", policy_index)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._parallel_training_workers) as executor:
      futures = {}
      while futures or not self._has_terminated(episodes_per_oracle):
        while (len(futures) < self._parallel_training_workers and
               not self._has_terminated(scheduled_episodes)):
          agents, indexes = self._sample_unfinished_policies_for_episode(
              new_policies, training_parameters, scheduled_episodes,
              strategy_sampler)
          rollout_id = self._next_training_rollout_id()
          futures[executor.submit(
              self._collect_parallel_rollout,
              game,
              agents,
              tuple(tuple(index) for index in indexes),
              rollout_id)] = tuple(tuple(index) for index in indexes)
          scheduled_episodes = rl_oracle.update_episodes_per_oracles(
              scheduled_episodes, indexes)

        if not futures:
          break
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
          futures.pop(future)
          result = future.result()
          self._merge_parallel_rollout(new_policies, result)
          episodes_per_oracle = rl_oracle.update_episodes_per_oracles(
              episodes_per_oracle, result.indexes)
          self._training_progress(episodes_per_oracle)

    rl_oracle.freeze_all(new_policies)
    return new_policies

  def _sample_unfinished_policies_for_episode(self, new_policies,
                                              training_parameters,
                                              scheduled_episodes,
                                              strategy_sampler):
    for _ in range(100):
      agents, indexes = self.sample_policies_for_episode(
          new_policies, training_parameters, scheduled_episodes,
          strategy_sampler)
      if not self._indexes_scheduled_enough(scheduled_episodes, indexes):
        return agents, indexes

    unfinished = np.argwhere(
        scheduled_episodes <= self._number_training_episodes)
    if unfinished.size == 0:
      raise RuntimeError("No unfinished Forceteki PPO policies to schedule")
    player, policy_index = [int(value) for value in unfinished[0]]
    agent_params = training_parameters[player][policy_index]
    agents = strategy_sampler(
        agent_params["total_policies"],
        agent_params["probabilities_of_playing_policies"])
    agents[player] = new_policies[player][policy_index]
    return agents, [(player, policy_index)]

  def _indexes_scheduled_enough(self, scheduled_episodes, indexes):
    return any(
        scheduled_episodes[player][policy_index] >
        self._number_training_episodes
        for player, policy_index in indexes)

  def _next_training_rollout_id(self):
    self._forceteki_trace_training_rollout += 1
    return self._forceteki_trace_training_rollout

  def _collect_parallel_rollout(self, game, agents, indexes, rollout_id):
    agents = list(agents)
    live_players = {}
    transitions_by_index = {}
    for player, agent in enumerate(agents):
      if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen():
        live_players[player] = agent
        agents[player] = agent.rollout_snapshot()

    with forceteki.forceteki_trace_context(
        rolloutType="training",
        seed=self._seed,
        trainingRollout=rollout_id):
      state = game.new_initial_state()
      pending = {player: None for player in live_players}
      pending_rewards = {player: 0.0 for player in live_players}

      try:
        while not state.is_terminal():
          if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(utils.random_choice(outcomes, probs))
            continue

          player = state.current_player()
          agent = agents[player]
          if player in live_players:
            if pending[player] is not None:
              self._append_parallel_transition(
                  transitions_by_index, live_players[player],
                  pending[player], pending_rewards[player], done=False)
            action, pending[player] = agent.collect_training_action(state)
            pending_rewards[player] = 0.0
          else:
            action_probs = agent(state, player)
            outcomes, probs = zip(*action_probs.items())
            action = utils.random_choice(outcomes, probs)

          state.apply_action(action)
          rewards = state.returns() if state.is_terminal() else state.rewards()
          if not rewards:
            rewards = [0.0] * state.num_players()
          for live_player in live_players:
            if pending[live_player] is not None:
              pending_rewards[live_player] += rewards[live_player]

        for live_player, live_agent in live_players.items():
          if pending[live_player] is not None:
            self._append_parallel_transition(
                transitions_by_index, live_agent, pending[live_player],
                pending_rewards[live_player], done=True)
      finally:
        _close_state(state)

    return _ParallelRolloutResult(
        indexes=indexes,
        transitions_by_index=transitions_by_index)

  def _append_parallel_transition(self, transitions_by_index, live_agent,
                                  transition, reward, done):
    transition = dict(transition)
    transition["reward"] = float(reward)
    transition["done"] = bool(done)
    transitions_by_index.setdefault(
        self._live_policy_index(live_agent), []).append(transition)

  def _live_policy_index(self, live_agent):
    return (live_agent.player_id, getattr(live_agent, "_oracle_policy_index"))

  def _merge_parallel_rollout(self, new_policies, result):
    for player, policy_index in result.indexes:
      policy = new_policies[player][policy_index]
      transitions = result.transitions_by_index.get((player, policy_index), [])
      if transitions:
        policy.merge_training_episode(transitions)

  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    del oracle_specific_execution_kwargs
    self._forceteki_trace_training_rollout += 1
    with forceteki.forceteki_trace_context(
        rolloutType="training",
        seed=self._seed,
        trainingRollout=self._forceteki_trace_training_rollout):
      state = game.new_initial_state()
      live_agents = [
          agent for agent in agents
          if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen()
      ]

      try:
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
