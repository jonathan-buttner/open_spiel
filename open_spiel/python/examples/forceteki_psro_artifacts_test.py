# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile

from absl.testing import absltest
import numpy as np
import torch

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

  def test_flags_to_dict_omits_removed_forceteki_seed(self):
    flags_obj = type("Flags", (), {
        "game_name": "python_forceteki_swu",
        "n_players": 2,
        "seed": 7,
        "forceteki_seed": "legacy",
    })()

    flags_dict = forceteki_psro_artifacts.flags_to_dict(flags_obj)

    self.assertEqual(flags_dict["seed"], 7)
    self.assertNotIn("forceteki_seed", flags_dict)

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

  def test_policy_checkpoint_widens_old_action_head(self):
    env = rl_environment.Environment("kuhn_poker")
    old_kwargs = _ppo_kwargs(env)
    old_kwargs["num_actions"] = 512
    checkpoint_policy = ForcetekiPPOPolicy(env, 0, **old_kwargs)
    checkpoint = checkpoint_policy.checkpoint(player_id=0, policy_index=0)
    checkpoint["kwargs"]["num_actions"] = 512
    env.action_spec = lambda: {"num_actions": 4096}

    restored = ForcetekiPPOPolicy.from_checkpoint(env, checkpoint)

    self.assertEqual(restored._num_actions, env.action_spec()["num_actions"])
    old_weights = checkpoint["network_state_dict"]["action_head.weight"]
    old_bias = checkpoint["network_state_dict"]["action_head.bias"]
    restored_weights = restored.get_weights()["action_head.weight"]
    restored_bias = restored.get_weights()["action_head.bias"]
    torch.testing.assert_close(
        restored_weights[:old_weights.shape[0]], old_weights)
    torch.testing.assert_close(restored_bias[:old_bias.shape[0]], old_bias)

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
