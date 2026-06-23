# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import copy
import concurrent.futures
import json
import os
import queue
import threading
import tempfile

from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.games import forceteki  # pylint: disable=unused-import
import pyspiel


def _fake_worker_state(terminal=False):
  return {
      "currentPlayer": -4 if terminal else 0,
      "currentPlayerId": None if terminal else "player-0",
      "isTerminal": terminal,
      "legalActions": [] if terminal else [0, 1],
      "legalDecisions": [] if terminal else [
          {
              "actionId": 0,
              "id": "button:pass",
              "playerId": "player-0",
              "kind": "prompt-button",
              "label": "Pass",
              "rawDecision": {
                  "kind": "prompt-button",
                  "playerId": "player-0",
                  "buttonArg": "pass",
                  "buttonText": "Pass",
              },
          },
          {
              "actionId": 1,
              "id": "card:attack",
              "playerId": "player-0",
              "kind": "card-click",
              "label": "Click Unit",
              "card": {
                  "uuid": "Card_1",
                  "id": "unit-1",
                  "name": "Unit One",
                  "zone": "groundArena",
                  "controllerId": "player-0",
                  "type": "basicUnit",
                  "exhausted": False,
                  "selectable": True,
              },
              "rawDecision": {
                  "kind": "card-click",
                  "playerId": "player-0",
                  "cardUuid": "Card_1",
              },
          },
      ],
      "returns": [1, -1] if terminal else [0, 0],
      "observationTensors": [[0.0] * 32768, [0.0] * 32768],
      "state": {
          "gameId": "fake-game",
          "phase": "action",
          "roundNumber": 2,
          "actionNumber": 3 if terminal else 2,
          "activePlayerId": None if terminal else "player-0",
          "initiativePlayerId": "player-0",
          "isComplete": terminal,
          "winnerNames": ["player0"] if terminal else [],
          "players": {
              "player-0": {
                  "id": "player-0",
                  "name": "player0",
                  "hasInitiative": True,
                  "isActivePlayer": not terminal,
                  "availableResources": 1,
                  "resourcesTotal": 2,
                  "handCount": 3,
                  "deckCount": 45,
                  "discardCount": 0,
                  "base": {"remainingHp": 30},
                  "hand": [],
                  "discard": [],
                  "resources": [],
                  "groundArena": [{"uuid": "Card_1", "name": "Unit One"}],
                  "spaceArena": [],
                  "prompt": {
                      "menuTitle": "Choose an action",
                      "promptTitle": "Action Window",
                      "promptType": "actionWindow",
                  },
              },
              "player-1": {
                  "id": "player-1",
                  "name": "player1",
                  "hasInitiative": False,
                  "isActivePlayer": False,
                  "availableResources": 0,
                  "resourcesTotal": 2,
                  "handCount": 4,
                  "deckCount": 44,
                  "discardCount": 1,
                  "base": {"hp": 33, "damage": 5},
                  "hand": [],
                  "discard": [{"uuid": "Card_2", "name": "Discarded"}],
                  "resources": [],
                  "groundArena": [],
                  "spaceArena": [],
                  "prompt": {},
              },
          },
      },
  }


class FakeNodeWorker:
  instances = []
  fail_next_reset = False
  _lock = threading.Lock()

  def __init__(self, params):
    self.params = params
    self._process = object()
    self.reset_count = 0
    self.closed = False
    self.requests = []
    with FakeNodeWorker._lock:
      self.worker_id = len(FakeNodeWorker.instances)
      FakeNodeWorker.instances.append(self)

  def request(self, op, params=None):
    self.requests.append((op, copy.deepcopy(params)))
    if self.closed:
      raise RuntimeError("worker is closed")
    if op in ("reset", "restore_checkpoint"):
      if op == "reset" and FakeNodeWorker.fail_next_reset:
        FakeNodeWorker.fail_next_reset = False
        raise RuntimeError("reset failed")
      self.reset_count += 1
      return _fake_worker_state(terminal=False)
    if op == "step":
      return _fake_worker_state(terminal=True)
    if op == "export_checkpoint":
      return {"actionHistory": []}
    if op == "close":
      return {"closed": True}
    raise ValueError(op)

  def close(self):
    self.closed = True
    self._process = None


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


def _sample_deck(deck_id):
  return {
      "metadata": {"name": deck_id, "author": "test"},
      "deckID": deck_id,
      "leader": {"id": "SOR_010", "count": 1},
      "base": {"id": "SOR_027", "count": 1},
      "deck": [{"id": "SOR_044", "count": 50}],
      "sideboard": [],
  }


def _write_deck(deck_dir, file_name, deck_id):
  path = os.path.join(deck_dir, file_name)
  deck = _sample_deck(deck_id)
  with open(path, "w", encoding="utf-8") as deck_file:
    json.dump(deck, deck_file)
  return path, deck


class ForcetekiTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._original_trace_path = os.environ.pop("FORCETEKI_TRACE_PATH", None)
    self._original_pool_size = os.environ.pop(
        "FORCETEKI_WORKER_POOL_SIZE", None)
    self._original_deck_env = {
        name: os.environ.pop(name, None)
        for name in (
            "FORCETEKI_DECK_POOL_PATH",
            "FORCETEKI_DECKS_PATH",
            "FORCETEKI_PLAYER0_DECK_PATH",
            "FORCETEKI_PLAYER1_DECK_PATH",
        )
    }
    forceteki._TRACE_GLOBAL_ACTION_COUNT = 0

  def tearDown(self):
    forceteki.close_all_workers()
    if self._original_trace_path is not None:
      os.environ["FORCETEKI_TRACE_PATH"] = self._original_trace_path
    else:
      os.environ.pop("FORCETEKI_TRACE_PATH", None)
    if self._original_pool_size is not None:
      os.environ["FORCETEKI_WORKER_POOL_SIZE"] = self._original_pool_size
    else:
      os.environ.pop("FORCETEKI_WORKER_POOL_SIZE", None)
    for name, value in self._original_deck_env.items():
      if value is not None:
        os.environ[name] = value
      else:
        os.environ.pop(name, None)
    super().tearDown()

  def close_forceteki_states(self, *states):
    for state in states:
      state.close()

  def patch_node_worker(self):
    original_worker = forceteki._NodeWorker
    FakeNodeWorker.instances = []
    FakeNodeWorker.fail_next_reset = False
    forceteki._NodeWorker = FakeNodeWorker
    self.addCleanup(lambda: setattr(forceteki, "_NodeWorker", original_worker))

  def overwrite_state_for_legal_action_test(self, state, legal_decisions):
    state._state = {
        "currentPlayer": 0,
        "currentPlayerId": "player-0",
        "isTerminal": False,
        "legalActions": list(range(len(legal_decisions))),
        "legalDecisions": legal_decisions,
        "returns": [0, 0],
        "observationTensors": [[0.0] * 32768, [0.0] * 32768],
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
    self.assertLen(state.observation_tensor(state.current_player()), 32768)
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

  def test_reset_passes_decklists_to_worker(self):
    self.patch_node_worker()
    temp_dir = tempfile.mkdtemp()
    player0_path = os.path.join(temp_dir, "player0.json")
    player1_path = os.path.join(temp_dir, "player1.json")
    player0_deck = {
        "deckID": "deck-0",
        "leader": {"id": "SOR_010"},
        "base": {"id": "SOR_027"},
        "deck": [{"id": "SOR_044", "count": 3}],
        "sideboard": [],
    }
    player1_deck = {
        "deckID": "deck-1",
        "leader": {"id": "SOR_005"},
        "base": {"id": "SOR_029"},
        "deck": [{"id": "SOR_045", "count": 3}],
        "sideboard": [],
    }
    with open(player0_path, "w", encoding="utf-8") as player0_file:
      json.dump(player0_deck, player0_file)
    with open(player1_path, "w", encoding="utf-8") as player1_file:
      json.dump(player1_deck, player1_file)

    game = pyspiel.load_game("python_forceteki_swu", {
        "player0_deck_path": player0_path,
        "player1_deck_path": player1_path,
    })
    state = game.new_initial_state()

    worker = FakeNodeWorker.instances[0]
    self.assertEqual(worker.requests[0][0], "reset")
    self.assertEqual(
        worker.requests[0][1]["decks"], [player0_deck, player1_deck])
    self.close_forceteki_states(state)

  def test_reset_omits_decks_when_deck_paths_are_unset(self):
    self.patch_node_worker()
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    worker = FakeNodeWorker.instances[0]
    self.assertEqual(worker.requests[0][0], "reset")
    self.assertNotIn("decks", worker.requests[0][1])
    self.close_forceteki_states(state)

  def test_deck_pool_sampler_uses_all_decks_before_repeats(self):
    temp_dir = tempfile.mkdtemp()
    for index in range(5):
      _write_deck(temp_dir, f"deck-{index}.json", f"deck-{index}")
    with open(os.path.join(temp_dir, "ignored.txt"), "w",
              encoding="utf-8") as ignored_file:
      ignored_file.write("not a deck")
    sampler = forceteki._DeckPoolSampler(temp_dir, "seed")

    drawn = []
    while len(drawn) < 5:
      drawn.extend(deck["deckID"] for deck in sampler.sample_pair())

    self.assertLen(set(drawn[:5]), 5)

  def test_deck_pool_sampler_avoids_mirror_matches_when_possible(self):
    temp_dir = tempfile.mkdtemp()
    for index in range(3):
      _write_deck(temp_dir, f"deck-{index}.json", f"deck-{index}")
    sampler = forceteki._DeckPoolSampler(temp_dir, "seed")

    for _ in range(10):
      player0_deck, player1_deck = sampler.sample_pair()
      self.assertNotEqual(player0_deck["deckID"], player1_deck["deckID"])

  def test_deck_pool_sampler_is_reproducible_for_seed(self):
    temp_dir = tempfile.mkdtemp()
    for index in range(4):
      _write_deck(temp_dir, f"deck-{index}.json", f"deck-{index}")
    first_sampler = forceteki._DeckPoolSampler(temp_dir, "seed")
    second_sampler = forceteki._DeckPoolSampler(temp_dir, "seed")

    first_sequence = [
        [deck["deckID"] for deck in first_sampler.sample_pair()]
        for _ in range(5)
    ]
    second_sequence = [
        [deck["deckID"] for deck in second_sampler.sample_pair()]
        for _ in range(5)
    ]

    self.assertEqual(first_sequence, second_sequence)

  def test_deck_pool_path_passes_sampled_decks_to_worker(self):
    self.patch_node_worker()
    temp_dir = tempfile.mkdtemp()
    for index in range(3):
      _write_deck(temp_dir, f"deck-{index}.json", f"deck-{index}")
    game = pyspiel.load_game("python_forceteki_swu", {
        "deck_pool_path": temp_dir,
        "seed": "deck-seed",
    })

    states = [game.new_initial_state(), game.new_initial_state()]

    first_reset = FakeNodeWorker.instances[0].requests[0][1]
    second_reset = FakeNodeWorker.instances[1].requests[0][1]
    self.assertIn("decks", first_reset)
    self.assertIn("decks", second_reset)
    self.assertNotEqual(
        first_reset["decks"][0]["deckID"],
        first_reset["decks"][1]["deckID"])
    first_three_draws = [
        first_reset["decks"][0]["deckID"],
        first_reset["decks"][1]["deckID"],
        second_reset["decks"][0]["deckID"],
    ]
    self.assertLen(set(first_three_draws), 3)
    self.close_forceteki_states(*states)

  def test_deck_pool_path_can_come_from_environment(self):
    self.patch_node_worker()
    temp_dir = tempfile.mkdtemp()
    _write_deck(temp_dir, "deck-0.json", "deck-0")
    os.environ["FORCETEKI_DECK_POOL_PATH"] = temp_dir
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    worker = FakeNodeWorker.instances[0]
    self.assertEqual(worker.requests[0][1]["decks"][0]["deckID"], "deck-0")
    self.assertEqual(worker.requests[0][1]["decks"][1]["deckID"], "deck-0")
    self.close_forceteki_states(state)

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

  def test_trace_file_is_not_created_when_debug_disabled(self):
    self.patch_node_worker()
    trace_path = os.path.join(tempfile.mkdtemp(), "trace.ndjson")
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    state.apply_action(1)

    self.assertFalse(os.path.exists(trace_path))
    self.close_forceteki_states(state)

  def test_trace_file_records_decision_and_choice(self):
    self.patch_node_worker()
    trace_path = os.path.join(tempfile.mkdtemp(), "trace.ndjson")
    os.environ["FORCETEKI_TRACE_PATH"] = trace_path
    game = pyspiel.load_game("python_forceteki_swu")
    state = game.new_initial_state()

    with forceteki.forceteki_trace_context(
        rolloutType="evaluation", profileIndex=[1, 0], simulationIndex=2):
      state.apply_action(1)

    with open(trace_path, encoding="utf-8") as trace_file:
      entries = [json.loads(line) for line in trace_file]

    self.assertLen(entries, 1)
    entry = entries[0]
    self.assertEqual(entry["rolloutContext"]["rolloutType"], "evaluation")
    self.assertEqual(entry["gameId"], "fake-game")
    self.assertEqual(entry["decisionPlayerId"], "player-0")
    self.assertEqual(entry["phase"], "action")
    self.assertEqual(entry["turnNumber"], 2)
    self.assertEqual(entry["preDecisionState"]["gameId"], "fake-game")
    self.assertEqual(entry["postActionSnapshot"]["isComplete"], True)
    self.assertEqual(entry["stateView"]["myBaseHealth"], 30)
    self.assertEqual(entry["stateView"]["opponentBaseHealth"], 28)
    self.assertEqual(entry["chosenAction"]["actionId"], 1)
    self.assertEqual(entry["chosenAction"]["id"], "card:attack")
    self.assertTrue(any(
        action["actionId"] == entry["chosenAction"]["actionId"]
        for action in entry["legalActions"]))
    self.assertEqual(entry["rawLegalActions"], [0, 1])
    self.assertNotIn("observationTensor", entry)
    self.assertNotIn("observationTensors", entry)
    self.assertEqual(entry["postAction"]["terminalReason"],
                     "forceteki_terminal")
    self.close_forceteki_states(state)

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

  def test_worker_pool_size_zero_keeps_direct_worker_lifecycle(self):
    self.patch_node_worker()
    game = pyspiel.load_game("python_forceteki_swu", {"worker_pool_size": 0})

    first = game.new_initial_state()
    first_worker = first._worker
    first.close()
    second = game.new_initial_state()
    second_worker = second._worker
    second.close()

    self.assertLen(FakeNodeWorker.instances, 2)
    self.assertIsNot(first_worker, second_worker)
    self.assertIsNone(first_worker._process)
    self.assertIsNone(second_worker._process)

  def test_worker_pool_reuses_worker_after_state_close(self):
    self.patch_node_worker()
    game = pyspiel.load_game("python_forceteki_swu", {"worker_pool_size": 1})

    first = game.new_initial_state()
    worker = first._worker
    first.close()
    second = game.new_initial_state()
    second.close()

    self.assertLen(FakeNodeWorker.instances, 1)
    self.assertIsNone(second._worker)
    self.assertEqual(worker.reset_count, 2)
    self.assertIsNotNone(worker._process)

  def test_worker_pool_never_exceeds_size_under_concurrent_checkout(self):
    self.patch_node_worker()
    game = pyspiel.load_game("python_forceteki_swu", {"worker_pool_size": 2})
    started = queue.Queue()
    release = threading.Event()

    def hold_state():
      state = game.new_initial_state()
      try:
        started.put(state._worker.worker_id)
        release.wait(timeout=5)
      finally:
        state.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(hold_state) for _ in range(4)]
      first_batch = [started.get(timeout=2), started.get(timeout=2)]
      with self.assertRaises(queue.Empty):
        started.get(timeout=0.1)
      self.assertLessEqual(len(FakeNodeWorker.instances), 2)
      release.set()
      concurrent.futures.wait(futures, timeout=5)

    self.assertEqual(set(first_batch), {0, 1})
    self.assertLessEqual(len(FakeNodeWorker.instances), 2)
    self.assertTrue(all(future.done() for future in futures))
    for future in futures:
      future.result()

  def test_worker_pool_discards_worker_after_reset_failure(self):
    self.patch_node_worker()
    FakeNodeWorker.fail_next_reset = True
    game = pyspiel.load_game("python_forceteki_swu", {"worker_pool_size": 1})

    with self.assertRaisesRegex(RuntimeError, "reset failed"):
      game.new_initial_state()
    failed_worker = FakeNodeWorker.instances[0]
    state = game.new_initial_state()
    replacement_worker = state._worker
    state.close()

    self.assertLen(FakeNodeWorker.instances, 2)
    self.assertIsNone(failed_worker._process)
    self.assertIsNot(failed_worker, replacement_worker)
    self.assertIsNotNone(replacement_worker._process)

  def test_close_all_workers_closes_pooled_idle_workers(self):
    self.patch_node_worker()
    game = pyspiel.load_game("python_forceteki_swu", {"worker_pool_size": 1})
    state = game.new_initial_state()
    worker = state._worker
    state.close()

    self.assertIsNotNone(worker._process)
    forceteki.close_all_workers()

    self.assertIsNone(worker._process)

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
