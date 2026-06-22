# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile

from absl.testing import absltest
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.examples import forceteki_psro_artifacts
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy


def _ppo_kwargs(env):
  return {
      "info_state_size": env.observation_spec()["info_state"][0],
      "num_actions": env.action_spec()["num_actions"],
      "hidden_layers_sizes": [8],
      "steps_per_batch": 4,
      "num_minibatches": 1,
      "update_epochs": 1,
      "learning_rate": 1e-3,
      "gamma": 0.99,
      "gae_lambda": 0.95,
      "clip_coef": 0.2,
      "entropy_coef": 0.01,
      "value_coef": 0.5,
      "max_grad_norm": 0.5,
      "target_kl": None,
      "device": "cpu",
      "intent_vocab_size": 8,
      "kind_vocab_size": 8,
      "control_vocab_size": 8,
      "card_vocab_size": 8,
  }


class ForcetekiPsroArtifactsTest(absltest.TestCase):

  def test_policy_checkpoint_round_trips_weights_and_optimizer(self):
    env = rl_environment.Environment("kuhn_poker")
    policy = ForcetekiPPOPolicy(env, 0, **_ppo_kwargs(env))
    policy.freeze()

    checkpoint = policy.checkpoint(player_id=0, policy_index=0)
    restored = ForcetekiPPOPolicy.from_checkpoint(env, checkpoint)

    self.assertTrue(restored.is_frozen())
    for key, value in policy.get_weights().items():
      np.testing.assert_array_equal(
          value.numpy(), restored.get_weights()[key].numpy())

  def test_policy_population_save_and_load(self):
    env = rl_environment.Environment("kuhn_poker")
    temp_dir = tempfile.mkdtemp()
    policies = [
        [ForcetekiPPOPolicy(env, 0, **_ppo_kwargs(env))],
        [ForcetekiPPOPolicy(env, 1, **_ppo_kwargs(env))],
    ]

    policy_entries = forceteki_psro_artifacts.save_policy_population(
        temp_dir, policies)
    forceteki_psro_artifacts._write_json(  # pylint: disable=protected-access
        os.path.join(temp_dir, forceteki_psro_artifacts.MANIFEST_FILENAME),
        {
            "format": "forceteki_psro_run_v1",
            "policy_population": policy_entries,
        })

    restored = forceteki_psro_artifacts.load_policy_population(temp_dir, env)

    self.assertLen(restored, 2)
    self.assertLen(restored[0], 1)
    self.assertIsInstance(restored[0][0], ForcetekiPPOPolicy)


if __name__ == "__main__":
  absltest.main()
