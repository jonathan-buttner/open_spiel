# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import copy

from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.games import forceteki  # pylint: disable=unused-import
import pyspiel


def _without_prompt_uuids(value):
  if isinstance(value, dict):
    return {
        key: _without_prompt_uuids(nested)
        for key, nested in value.items()
        if key != "promptUuid"
    }
  if isinstance(value, list):
    return [_without_prompt_uuids(nested) for nested in value]
  return value


class ForcetekiTest(absltest.TestCase):

  def close_forceteki_states(self, *states):
    for state in states:
      state._worker.close()

  def test_load_game_and_reset_rl_environment(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    self.assertFalse(state.is_terminal())
    self.assertEqual(state.forceteki_terminal_reason(), "non_terminal")
    self.assertEqual(state.forceteki_move_number(), 0)
    self.assertNotEmpty(state.legal_actions())
    self.assertLen(state.observation_tensor(state.current_player()), 4096)

    env = rl_environment.Environment("python_forceteki_swu")
    timestep = env.reset()

    self.assertIn("info_state", timestep.observations)
    self.assertIn("legal_actions", timestep.observations)
    self.assertNotEmpty(timestep.observations["legal_actions"][0])
    self.close_forceteki_states(state, env._state)

  def test_terminal_reason_reports_open_spiel_cap(self):
    game = pyspiel.load_game("python_forceteki_swu", {"max_game_length": 0})
    state = game.new_initial_state()

    self.assertTrue(state.is_terminal())
    self.assertEqual(state.forceteki_terminal_reason(), "open_spiel_cap")
    self.assertEqual(state.forceteki_move_number(), 0)
    self.close_forceteki_states(state)

  def assert_forceteki_states_equal(self, left, right):
    self.assertEqual(str(left), str(right))
    self.assertEqual(left.current_player(), right.current_player())
    self.assertEqual(left.legal_actions(), right.legal_actions())
    self.assertEqual(left.returns(), right.returns())
    self.assertEqual(left._move_number, right._move_number)
    self.assertEqual(
        _without_prompt_uuids(left._state),
        _without_prompt_uuids(right._state))
    for player in range(2):
      self.assertSequenceAlmostEqual(
          left.observation_tensor(player),
          right.observation_tensor(player))

  def test_clone_after_reset(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    clone = state.clone()

    self.assert_forceteki_states_equal(state, clone)
    self.close_forceteki_states(state, clone)

  def test_clone_after_actions_and_continue(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    for _ in range(2):
      state.apply_action(state.legal_actions()[0])

    clone = state.clone()
    self.assert_forceteki_states_equal(state, clone)

    action = state.legal_actions()[0]
    state.apply_action(action)
    clone.apply_action(action)
    self.assert_forceteki_states_equal(state, clone)
    self.close_forceteki_states(state, clone)

  def test_clone_continues_independently(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    clone = state.clone()
    original_state = _without_prompt_uuids(state._state)

    clone.apply_action(clone.legal_actions()[0])

    self.assertEqual(_without_prompt_uuids(state._state), original_state)
    self.close_forceteki_states(state, clone)

  def test_deepcopy_uses_clone_path(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    clone = copy.deepcopy(state)

    self.assert_forceteki_states_equal(state, clone)
    self.close_forceteki_states(state, clone)


if __name__ == "__main__":
  absltest.main()
