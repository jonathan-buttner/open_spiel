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

  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    self._forceteki_trace_training_rollout += 1
    with forceteki.forceteki_trace_context(
        rolloutType="training",
        trainingRollout=self._forceteki_trace_training_rollout):
      return super()._rollout(
          game, agents, **oracle_specific_execution_kwargs)


class ForcetekiPPOOracle(ForcetekiTraceRLOracle):
  """PSRO oracle that trains factored Forceteki PPO responders."""

  def __call__(self, *args, **kwargs):
    new_policies = super().__call__(*args, **kwargs)
    for player_policies in new_policies:
      for pol in player_policies:
        if isinstance(pol, ForcetekiPPOPolicy):
          pol.finish_training()
    return new_policies

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
