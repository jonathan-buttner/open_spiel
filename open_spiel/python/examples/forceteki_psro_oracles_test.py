# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import threading
import time

from absl.testing import absltest

from open_spiel.python.examples import forceteki_psro_oracles


class FakePolicy:

  def __init__(self, player_id, frozen=True):
    self.player_id = player_id
    self._frozen = frozen
    self.merged_episodes = []

  def freeze(self):
    self._frozen = True

  def unfreeze(self):
    self._frozen = False

  def is_frozen(self):
    return self._frozen

  def merge_training_episode(self, transitions):
    self.merged_episodes.append(list(transitions))


class RecordingPPOOracle(forceteki_psro_oracles.ForcetekiPPOOracle):

  def __init__(self):
    super().__init__(
        env=object(),
        best_response_class=FakePolicy,
        best_response_kwargs={},
        number_training_episodes=1,
        parallel_training_workers=2,
        seed=1)
    self._lock = threading.Lock()
    self._active_rollouts = 0
    self.max_active_rollouts = 0
    self.rollout_ids = []

  def generate_new_policies(self, training_parameters):
    del training_parameters
    return [[FakePolicy(0, frozen=False)], [FakePolicy(1, frozen=False)]]

  def _collect_parallel_rollout(self, game, agents, indexes, rollout_id):
    del game, agents
    with self._lock:
      self._active_rollouts += 1
      self.max_active_rollouts = max(
          self.max_active_rollouts, self._active_rollouts)
      self.rollout_ids.append(rollout_id)
    try:
      time.sleep(0.05)
      return forceteki_psro_oracles._ParallelRolloutResult(
          indexes=indexes,
          transitions_by_index={
              index: [{"rollout_id": rollout_id}]
              for index in indexes
          })
    finally:
      with self._lock:
        self._active_rollouts -= 1


def _training_parameters():
  total_policies = [
      [FakePolicy(0, frozen=True)],
      [FakePolicy(1, frozen=True)],
  ]
  probabilities = [[1.0], [1.0]]
  return [
      [{
          "policy": total_policies[0][0],
          "total_policies": total_policies,
          "current_player": 0,
          "probabilities_of_playing_policies": probabilities,
      }],
      [{
          "policy": total_policies[1][0],
          "total_policies": total_policies,
          "current_player": 1,
          "probabilities_of_playing_policies": probabilities,
      }],
  ]


class ForcetekiPsroOraclesTest(absltest.TestCase):

  def test_parallel_ppo_oracle_submits_and_merges_rollouts(self):
    oracle = RecordingPPOOracle()

    new_policies = oracle(
        game=object(),
        training_parameters=_training_parameters())

    merged_counts = [
        len(new_policies[0][0].merged_episodes),
        len(new_policies[1][0].merged_episodes),
    ]
    self.assertEqual(merged_counts, [2, 2])
    self.assertGreaterEqual(oracle.max_active_rollouts, 2)
    self.assertLen(oracle.rollout_ids, 4)
    self.assertEqual(sorted(oracle.rollout_ids), [1, 2, 3, 4])
    self.assertTrue(new_policies[0][0].is_frozen())
    self.assertTrue(new_policies[1][0].is_frozen())


if __name__ == "__main__":
  absltest.main()
