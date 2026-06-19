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

  def tearDown(self):
    forceteki.close_all_workers()
    super().tearDown()

  def close_forceteki_states(self, *states):
    for state in states:
      state.close()

  def overwrite_state_for_legal_action_test(self, state, legal_decisions):
    state._state = {
        "currentPlayer": 0,
        "currentPlayerId": "player-0",
        "isTerminal": False,
        "legalActions": list(range(len(legal_decisions))),
        "legalDecisions": legal_decisions,
        "returns": [0, 0],
        "observationTensors": [[0.0] * 4096, [0.0] * 4096],
        "state": {
            "phase": "action",
            "roundNumber": 1,
            "actionNumber": 7,
            "players": {
                "player-0": {
                    "prompt": {
                        "menuTitle": "Choose an action",
                        "promptTitle": "Action Window",
                        "promptType": "actionWindow",
                    },
                },
            },
        },
    }
    state._recent_action_keys = []

  def test_load_game_and_reset_rl_environment(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    self.assertFalse(state.is_terminal())
    self.assertEqual(state.forceteki_terminal_reason(), "non_terminal")
    self.assertEqual(state.forceteki_move_number(), 0)
    self.assertNotEmpty(state.legal_actions())
    self.assertLen(state.observation_tensor(state.current_player()), 4096)
    structured_actions = state.forceteki_legal_actions()
    self.assertNotEmpty(structured_actions)
    self.assertTrue(all(isinstance(action, dict)
                        for action in structured_actions.values()))

    env = rl_environment.Environment("python_forceteki_swu")
    timestep = env.reset()

    self.assertIn("info_state", timestep.observations)
    self.assertIn("legal_actions", timestep.observations)
    self.assertNotEmpty(timestep.observations["legal_actions"][0])
    self.close_forceteki_states(state, env._state)

  def test_filters_selected_card_when_forward_action_exists(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    self.overwrite_state_for_legal_action_test(state, [
        {
            "actionId": 0,
            "id": "card:selected",
            "playerId": "player-0",
            "kind": "card-click",
            "label": "Click Selected",
            "card": {"uuid": "selected", "selected": True},
            "rawDecision": {
                "kind": "card-click",
                "playerId": "player-0",
                "cardUuid": "selected",
            },
        },
        {
            "actionId": 1,
            "id": "card:forward",
            "playerId": "player-0",
            "kind": "card-click",
            "label": "Click Forward",
            "card": {"uuid": "forward", "selected": False},
            "rawDecision": {
                "kind": "card-click",
                "playerId": "player-0",
                "cardUuid": "forward",
            },
        },
    ])

    self.assertEqual(state._legal_actions(None), [1])
    self.close_forceteki_states(state)

  def test_filters_cancel_when_forward_action_exists(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    self.overwrite_state_for_legal_action_test(state, [
        {
            "actionId": 0,
            "id": "button:cancel",
            "playerId": "player-0",
            "kind": "prompt-button",
            "label": "Cancel",
            "rawDecision": {
                "kind": "prompt-button",
                "playerId": "player-0",
                "buttonArg": "cancel",
                "buttonText": "Cancel",
            },
        },
        {
            "actionId": 1,
            "id": "card:forward",
            "playerId": "player-0",
            "kind": "card-click",
            "label": "Click Forward",
            "card": {"uuid": "forward", "selected": False},
            "rawDecision": {
                "kind": "card-click",
                "playerId": "player-0",
                "cardUuid": "forward",
            },
        },
    ])

    self.assertEqual(state._legal_actions(None), [1])
    self.close_forceteki_states(state)

  def test_keeps_backtracking_when_it_is_the_only_action(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    self.overwrite_state_for_legal_action_test(state, [
        {
            "actionId": 0,
            "id": "button:cancel",
            "playerId": "player-0",
            "kind": "prompt-button",
            "label": "Cancel",
            "rawDecision": {
                "kind": "prompt-button",
                "playerId": "player-0",
                "buttonArg": "cancel",
                "buttonText": "Cancel",
            },
        },
    ])

    self.assertEqual(state._legal_actions(None), [0])
    self.close_forceteki_states(state)

  def test_filters_recent_same_prompt_action_when_alternatives_exist(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    self.overwrite_state_for_legal_action_test(state, [
        {
            "actionId": 0,
            "id": "card:repeat",
            "playerId": "player-0",
            "kind": "card-click",
            "label": "Click Repeat",
            "card": {"uuid": "repeat", "selected": False},
            "rawDecision": {
                "kind": "card-click",
                "playerId": "player-0",
                "cardUuid": "repeat",
            },
        },
        {
            "actionId": 1,
            "id": "card:other",
            "playerId": "player-0",
            "kind": "card-click",
            "label": "Click Other",
            "card": {"uuid": "other", "selected": False},
            "rawDecision": {
                "kind": "card-click",
                "playerId": "player-0",
                "cardUuid": "other",
            },
        },
    ])
    state._recent_action_keys = [state._action_loop_key(0)]

    self.assertEqual(state._legal_actions(None), [1])
    self.close_forceteki_states(state)

  def test_clone_copies_recent_action_loop_guard(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    state._recent_action_keys = ["recent-key"]

    clone = state.clone()

    self.assertEqual(clone._recent_action_keys, ["recent-key"])
    self.close_forceteki_states(state, clone)

  def test_state_close_is_idempotent(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    worker = state._worker

    state.close()
    state.close()

    self.assertIsNone(state._worker)
    self.assertIsNone(worker._process)
    self.assertNotIn(worker, forceteki._LIVE_WORKERS)

  def test_close_all_workers_closes_live_workers(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    clone = state.clone()
    workers = [state._worker, clone._worker]

    forceteki.close_all_workers()

    self.assertTrue(all(worker._process is None for worker in workers))
    self.assertTrue(all(worker not in forceteki._LIVE_WORKERS
                        for worker in workers))
    self.close_forceteki_states(state, clone)

  def test_explicit_close_removes_original_and_clone_workers(self):
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()
    clone = state.clone()
    workers = [state._worker, clone._worker]

    self.close_forceteki_states(state, clone)

    self.assertTrue(all(worker._process is None for worker in workers))
    self.assertTrue(all(worker not in forceteki._LIVE_WORKERS
                        for worker in workers))

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
