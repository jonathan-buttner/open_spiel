# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from absl.testing import absltest

from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy


class FakeEnv:
  game = object()


class FakeState:

  _state = {"legalActions": [0]}

  def information_state_tensor(self, player_id):
    del player_id
    return [0.0, 1.0, 0.0]

  def legal_actions(self):
    return [0]


def _policy():
  return ForcetekiPPOPolicy(
      FakeEnv(),
      0,
      info_state_size=3,
      num_actions=2,
      hidden_layers_sizes=[4],
      steps_per_batch=100,
      num_minibatches=1,
      update_epochs=1,
      learning_rate=1e-3,
      gamma=0.99,
      gae_lambda=0.95,
      clip_coef=0.2,
      entropy_coef=0.01,
      value_coef=0.5,
      max_grad_norm=0.5,
      target_kl=None,
      device="cpu",
      intent_vocab_size=4,
      kind_vocab_size=4,
      control_vocab_size=4,
      card_vocab_size=4)


class ForcetekiPsroPpoTest(absltest.TestCase):

  def test_collect_training_action_does_not_mutate_learner_buffers(self):
    policy = _policy()

    action, transition = policy.collect_training_action(FakeState())

    self.assertEqual(action, 0)
    self.assertIsNone(policy._pending)  # pylint: disable=protected-access
    self.assertEqual(
        policy._pending_reward, 0.0)  # pylint: disable=protected-access
    self.assertEmpty(policy._buffer)  # pylint: disable=protected-access
    self.assertEqual(transition["action"], 0)
    self.assertIn("obs", transition)
    self.assertIn("logprob", transition)
    self.assertIn("value", transition)

  def test_rollout_snapshot_is_frozen_and_independent(self):
    policy = _policy()

    snapshot = policy.rollout_snapshot()
    snapshot.merge_training_episode([{"done": True}])

    self.assertTrue(snapshot.is_frozen())
    self.assertFalse(policy.is_frozen())
    self.assertEmpty(policy._buffer)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
