# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import contextlib
import io
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl import app
from absl import flags
from absl.testing import absltest

from open_spiel.python.examples import forceteki_psro


def _flags(**overrides):
  values = {
      "deck_pool_path": "",
      "decks_path": "",
      "debug": "off",
      "debug_dir": "forceteki_psro_debug",
      "forceteki_worker_pool_size": 0,
      "max_episode_steps": 1000,
      "n_players": 2,
      "oracle_type": "PPO",
      "output_dir": "",
      "parallel_eval_workers": 1,
      "parallel_training_workers": 1,
      "player0_deck_path": "",
      "player1_deck_path": "",
      "ppo_device": "cpu",
      "resume_from": "",
      "seed": 17,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class ForcetekiPsroTest(absltest.TestCase):

  def test_forceteki_seed_flag_is_not_registered(self):
    self.assertNotIn("forceteki_seed", flags.FLAGS)

  def test_game_params_use_seed_flag_for_forceteki_seed(self):
    params = forceteki_psro._game_params_from_flags(_flags(seed=23))

    self.assertEqual(params["seed"], "23")
    self.assertEqual(params["trace_mode"], "off")
    self.assertEqual(params["worker_pool_size"], 1)
    self.assertNotIn("trace_dir", params)

  def test_parallel_training_workers_must_be_positive(self):
    with self.assertRaisesRegex(app.UsageError,
                                "--parallel_training_workers"):
      forceteki_psro._validate_flags(_flags(parallel_training_workers=0))

  def test_effective_worker_pool_uses_training_workers(self):
    params = forceteki_psro._game_params_from_flags(
        _flags(parallel_training_workers=4))

    self.assertEqual(params["worker_pool_size"], 4)

  def test_effective_worker_pool_uses_eval_workers(self):
    params = forceteki_psro._game_params_from_flags(
        _flags(parallel_eval_workers=3))

    self.assertEqual(params["worker_pool_size"], 3)

  def test_effective_worker_pool_preserves_larger_explicit_pool(self):
    params = forceteki_psro._game_params_from_flags(
        _flags(
            forceteki_worker_pool_size=8,
            parallel_eval_workers=3,
            parallel_training_workers=4))

    self.assertEqual(params["worker_pool_size"], 8)

  def test_game_params_include_trace_dir_when_debug_enabled(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      params = forceteki_psro._game_params_from_flags(
          _flags(debug="FULL", debug_dir=temp_dir))

      self.assertEqual(params["trace_mode"], "full")
      self.assertIn("trace_dir", params)
      self.assertTrue(os.path.isdir(params["trace_dir"]))
      self.assertTrue(params["trace_dir"].startswith(temp_dir))

  def test_debug_rejects_boolean_values(self):
    for debug_value in ("True", "False", True, False):
      with self.subTest(debug_value=debug_value):
        with self.assertRaisesRegex(app.UsageError, "--debug must be one of"):
          forceteki_psro._game_params_from_flags(_flags(debug=debug_value))

  def test_game_params_include_deck_pool_from_environment(self):
    original_deck_pool_path = os.environ.get("FORCETEKI_DECK_POOL_PATH")
    os.environ["FORCETEKI_DECK_POOL_PATH"] = "/tmp/deck-pool"
    try:
      params = forceteki_psro._game_params_from_flags(_flags(seed=5))
    finally:
      if original_deck_pool_path is None:
        os.environ.pop("FORCETEKI_DECK_POOL_PATH", None)
      else:
        os.environ["FORCETEKI_DECK_POOL_PATH"] = original_deck_pool_path

    self.assertEqual(params["seed"], "5")
    self.assertEqual(params["deck_pool_path"], "/tmp/deck-pool")

  def test_missing_resume_dir_with_output_dir_starts_fresh(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      resume_from = os.path.join(temp_dir, "missing_run")
      output_dir = os.path.join(temp_dir, "new_run")
      output = io.StringIO()

      with contextlib.redirect_stdout(output):
        restored = forceteki_psro._restore_resume_state(
            _flags(resume_from=resume_from, output_dir=output_dir),
            env=object())

    self.assertEqual(restored, (None, None, None))
    self.assertIn("Resume directory not found", output.getvalue())
    self.assertIn(output_dir, output.getvalue())

  def test_missing_resume_dir_without_output_dir_errors(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      resume_from = os.path.join(temp_dir, "missing_run")

      with self.assertRaisesRegex(app.UsageError, "--output_dir is empty"):
        forceteki_psro._restore_resume_state(
            _flags(resume_from=resume_from),
            env=object())

  def test_existing_resume_path_that_is_not_directory_errors(self):
    with tempfile.NamedTemporaryFile() as temp_file:
      with self.assertRaisesRegex(app.UsageError, "not a directory"):
        forceteki_psro._restore_resume_state(
            _flags(resume_from=temp_file.name, output_dir="/tmp/new_run"),
            env=object())

  def test_existing_malformed_resume_dir_does_not_fall_back_to_fresh(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch.object(
          forceteki_psro.forceteki_psro_artifacts,
          "load_policy_population",
          side_effect=ValueError("bad artifacts")):
        with self.assertRaisesRegex(ValueError, "bad artifacts"):
          forceteki_psro._restore_resume_state(
              _flags(resume_from=temp_dir, output_dir="/tmp/new_run"),
              env=object())


if __name__ == "__main__":
  absltest.main()
