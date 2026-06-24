# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import types

from absl import flags
from absl.testing import absltest

from open_spiel.python.examples import forceteki_psro


def _flags(**overrides):
  values = {
      "n_players": 2,
      "max_episode_steps": 1000,
      "forceteki_worker_pool_size": 0,
      "seed": 17,
      "deck_pool_path": "",
      "decks_path": "",
      "player0_deck_path": "",
      "player1_deck_path": "",
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class ForcetekiPsroTest(absltest.TestCase):

  def test_forceteki_seed_flag_is_not_registered(self):
    self.assertNotIn("forceteki_seed", flags.FLAGS)

  def test_game_params_use_seed_flag_for_forceteki_seed(self):
    params = forceteki_psro._game_params_from_flags(_flags(seed=23))

    self.assertEqual(params["seed"], "23")

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


if __name__ == "__main__":
  absltest.main()
