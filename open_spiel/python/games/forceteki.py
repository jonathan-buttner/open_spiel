# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Forceteki Star Wars: Unlimited RL environment wrapper."""

import atexit
import contextlib
import contextvars
import json
import os
import subprocess
import threading
from typing import Any

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_NUM_DISTINCT_ACTIONS = 512
_OBSERVATION_TENSOR_SIZE = 4096
_DEFAULT_MAX_GAME_LENGTH = 1000
_RECENT_ACTION_KEY_LIMIT = 12
_LIVE_WORKERS = set()
_WORKER_POOLS = {}
_WORKER_REGISTRY_LOCK = threading.Lock()
_TRACE_PATH_ENV = "FORCETEKI_TRACE_PATH"
_TRACE_CONTEXT = contextvars.ContextVar("forceteki_trace_context", default={})
_TRACE_LOCK = threading.Lock()
_TRACE_GLOBAL_ACTION_COUNT = 0

_GAME_TYPE = pyspiel.GameType(
    short_name="python_forceteki_swu",
    long_name="Python Forceteki Star Wars: Unlimited",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "players": _NUM_PLAYERS,
        "max_game_length": _DEFAULT_MAX_GAME_LENGTH,
        "worker_pool_size": 0,
    })

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_DEFAULT_MAX_GAME_LENGTH)


@contextlib.contextmanager
def forceteki_trace_context(**metadata):
  """Adds metadata to Forceteki trace entries written in this context."""
  previous = _TRACE_CONTEXT.get()
  update = {key: value for key, value in metadata.items() if value is not None}
  token = _TRACE_CONTEXT.set({**previous, **update})
  try:
    yield
  finally:
    _TRACE_CONTEXT.reset(token)


class ForcetekiGame(pyspiel.Game):
  """OpenSpiel game that delegates state transitions to a Node worker."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    self._params = params or {}
    if int(self._params.get("players", _NUM_PLAYERS)) != _NUM_PLAYERS:
      raise ValueError("Forceteki SWU only supports 2 players")

  def new_initial_state(self):
    return ForcetekiState(self, self._params)

  def make_py_observer(self, iig_obs_type=None, params=None):
    del iig_obs_type
    return ForcetekiObserver(params)


class ForcetekiState(pyspiel.State):
  """Mutable OpenSpiel state backed by a long-lived Forceteki worker."""

  def __init__(self, game, params, checkpoint=None, move_number=0,
               recent_action_keys=None):
    super().__init__(game)
    self._params = dict(params)
    self._worker_pool = _worker_pool(params)
    self._worker_failed = False
    self._worker = (
        self._worker_pool.acquire() if self._worker_pool is not None
        else _NodeWorker(params))
    try:
      if checkpoint is None:
        self._state = self._request_worker("reset", _reset_params(params))
      else:
        self._state = self._request_worker("restore_checkpoint",
                                           {"checkpoint": checkpoint})
    except Exception:
      self._worker_failed = True
      self.close()
      raise
    self._move_number = move_number
    self._max_game_length = int(params.get("max_game_length",
                                           _DEFAULT_MAX_GAME_LENGTH))
    self._recent_action_keys = list(recent_action_keys or [])

  def current_player(self):
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return int(self._state["currentPlayer"])

  def _legal_actions(self, player):
    del player
    return self._loop_safe_legal_actions()

  def _all_legal_actions(self):
    legal_actions = self._state.get("legalActions", [])
    if legal_actions and isinstance(legal_actions[0], dict):
      return list(range(len(legal_actions)))
    return list(legal_actions)

  def _apply_action(self, action):
    action = int(action)
    action_key = self._action_loop_key(action)
    trace_path = _trace_path(self._params)
    pre_state = self._state
    legal_action_map = self.forceteki_legal_actions() if trace_path else {}
    forceteki_action = self._forceteki_action_for_open_spiel_action(action)
    self._apply_forceteki_action(forceteki_action)
    self._remember_action_key(action_key)
    if trace_path:
      self._append_trace_entry(
          trace_path, action, legal_action_map, pre_state)

  def _apply_forceteki_action(self, action):
    self._state = self._request_worker("step", {"action": action})
    self._move_number += 1

  def _action_to_string(self, player, action):
    del player
    action = int(action)
    action_strings = self._state.get("actionStrings", {})
    if str(action) in action_strings:
      return action_strings[str(action)]
    legal_action = self.forceteki_legal_action(action)
    if isinstance(legal_action, dict):
      return str(legal_action.get("label") or legal_action.get("id") or action)
    return str(legal_action)

  def is_terminal(self):
    return bool(self._state["isTerminal"]) or (
        self._move_number >= self._max_game_length)

  def returns(self):
    return [float(value) for value in self._state["returns"]]

  def forceteki_terminal_reason(self):
    if bool(self._state["isTerminal"]):
      return "forceteki_terminal"
    if self._move_number >= self._max_game_length:
      return "open_spiel_cap"
    return "non_terminal"

  def forceteki_move_number(self):
    return self._move_number

  def forceteki_legal_action(self, action):
    """Returns the structured Forceteki action behind an OpenSpiel action id."""
    action = int(action)
    legal_decision = self._legal_decision_for_action(action)
    if legal_decision is not None:
      return legal_decision
    return self._forceteki_action_for_open_spiel_action(action)

  def forceteki_legal_actions(self):
    """Returns structured legal actions keyed by OpenSpiel action id."""
    return {
        int(action): self.forceteki_legal_action(action)
        for action in self.legal_actions()
    }

  def _forceteki_action_for_open_spiel_action(self, action):
    legal_actions = self._state.get("legalActions", [])
    if legal_actions and isinstance(legal_actions[0], dict):
      if action < 0 or action >= len(legal_actions):
        raise ValueError(
            f"Illegal Forceteki action slot {action}. "
            f"Legal slots are: {list(range(len(legal_actions)))}")
      return legal_actions[action]
    return int(action)

  def _legal_decision_for_action(self, action):
    legal_decisions = self._state.get("legalDecisions", [])
    for decision in legal_decisions:
      if isinstance(decision, dict) and int(decision.get("actionId", -1)) == action:
        return decision
    if 0 <= action < len(legal_decisions):
      decision = legal_decisions[action]
      if isinstance(decision, dict):
        return decision
    return None

  def _loop_safe_legal_actions(self):
    legal_actions = self._all_legal_actions()
    if not legal_actions:
      return []

    forward_actions = [
        action for action in legal_actions
        if not self._is_backtracking_action(action)
    ]
    if forward_actions:
      legal_actions = forward_actions

    fresh_actions = [
        action for action in legal_actions
        if self._action_loop_key(action) not in self._recent_action_keys
    ]
    if fresh_actions:
      return fresh_actions
    return legal_actions

  def _is_backtracking_action(self, action):
    decision = self._legal_decision_for_action(action)
    if not isinstance(decision, dict):
      return False

    raw = self._raw_decision(decision)
    kind = decision.get("kind") or raw.get("kind")
    card = decision.get("card") or {}
    if kind in ("card-click", "display-card") and bool(card.get("selected")):
      return True

    button_arg = str(raw.get("buttonArg", "")).lower()
    button_text = str(raw.get("buttonText", decision.get("label", ""))).lower()
    return kind == "prompt-button" and (
        button_arg == "cancel" or button_text == "cancel")

  def _action_loop_key(self, action):
    decision = self._legal_decision_for_action(action)
    state = self._state.get("state", {})
    player_id = self._state.get("currentPlayerId")
    if player_id is None and isinstance(decision, dict):
      player_id = decision.get("playerId")
    prompt = {}
    if player_id:
      prompt = state.get("players", {}).get(player_id, {}).get("prompt", {})
    decision_id = (
        decision.get("id") if isinstance(decision, dict) else str(action))
    return json.dumps({
        "phase": state.get("phase"),
        "roundNumber": state.get("roundNumber"),
        "actionNumber": state.get("actionNumber"),
        "playerId": player_id,
        "prompt": {
            "menuTitle": prompt.get("menuTitle"),
            "promptTitle": prompt.get("promptTitle"),
            "promptType": prompt.get("promptType"),
        },
        "decisionId": decision_id,
    }, sort_keys=True)

  def _remember_action_key(self, action_key):
    if not action_key:
      return
    self._recent_action_keys.append(action_key)
    del self._recent_action_keys[:-_RECENT_ACTION_KEY_LIMIT]

  def _raw_decision(self, legal_action):
    return legal_action.get("rawAction") or legal_action.get("rawDecision") or {}

  def _append_trace_entry(self, trace_path, action, legal_action_map,
                          pre_worker_state):
    global _TRACE_GLOBAL_ACTION_COUNT
    with _TRACE_LOCK:
      _TRACE_GLOBAL_ACTION_COUNT += 1
      global_action_count = _TRACE_GLOBAL_ACTION_COUNT
      entry = self._trace_entry(
          global_action_count, action, legal_action_map, pre_worker_state)
      _write_trace_entry(trace_path, entry)

  def _trace_entry(self, global_action_count, action, legal_action_map,
                   pre_worker_state):
    pre_state = pre_worker_state.get("state", {})
    post_state = self._state.get("state", {})
    player_id = (
        pre_worker_state.get("currentPlayerId") or
        _action_player_id(legal_action_map.get(action)) or
        _state_active_player_id(pre_state))
    return {
        "globalActionCount": global_action_count,
        "actionCount": self._move_number,
        "moveNumber": self._move_number,
        "rolloutContext": dict(_TRACE_CONTEXT.get()),
        "decisionPlayerId": player_id,
        "openSpielCurrentPlayer": pre_worker_state.get("currentPlayer"),
        "turnNumber": pre_state.get("roundNumber"),
        "phase": pre_state.get("phase"),
        "preDecisionState": pre_state,
        "stateView": _build_state_view(pre_state, player_id),
        "legalActions": _trace_legal_actions(legal_action_map),
        "rawLegalActions": pre_worker_state.get("legalActions", []),
        "rawLegalDecisions": pre_worker_state.get("legalDecisions", []),
        "chosenAction": _trace_action(action, legal_action_map.get(action)),
        "postActionSnapshot": post_state,
        "postAction": {
            "currentPlayer": self._state.get("currentPlayer"),
            "currentPlayerId": self._state.get("currentPlayerId"),
            "isTerminal": bool(self._state.get("isTerminal")),
            "terminalReason": self.forceteki_terminal_reason(),
            "returns": self.returns(),
        },
    }

  def observation_tensor(self, player=None):
    if player is None:
      player = self.current_player()
    if player == pyspiel.PlayerId.TERMINAL:
      player = 0
    return np.asarray(self._state["observationTensors"][int(player)], dtype=np.float32)

  def information_state_tensor(self, player=None):
    # V1 exposes current legal visibility, not a perfect-recall information
    # state. This compatibility shim lets rollout-based OpenSpiel RL/PSRO
    # helpers use Forceteki policies while the full information-state encoder is
    # developed separately.
    return self.observation_tensor(player)

  def __str__(self):
    state = self._state.get("state", {})
    phase = state.get("phase", "?")
    current_player = self.current_player()
    legal_actions = self._state.get("legalActions", [])
    return f"forceteki phase={phase} player={current_player} legal={legal_actions}"

  def clone(self):
    clone_mode = str(self._params.get("clone_mode") or
                     os.environ.get("FORCETEKI_CLONE_MODE") or
                     "checkpoint")
    if clone_mode == "replay":
      checkpoint = self._request_worker("export_checkpoint")
      clone = ForcetekiState(self.get_game(), self._params)
      for action in checkpoint.get("actionHistory", []):
        if isinstance(action, dict):
          clone._apply_forceteki_action(action)
        else:
          clone.apply_action(int(action))
      clone._recent_action_keys = list(self._recent_action_keys)
      return clone

    checkpoint = self._request_worker("export_checkpoint")
    return ForcetekiState(
        self.get_game(),
        self._params,
        checkpoint=checkpoint,
        move_number=self._move_number,
        recent_action_keys=self._recent_action_keys)

  def __deepcopy__(self, memo):
    del memo
    return self.clone()

  def close(self):
    worker = getattr(self, "_worker", None)
    if worker is not None:
      worker_pool = getattr(self, "_worker_pool", None)
      if worker_pool is None:
        worker.close()
      else:
        worker_pool.release(
            worker, discard=getattr(self, "_worker_failed", False))
      self._worker = None

  def __del__(self):
    self.close()

  def _request_worker(self, op, params=None):
    try:
      return self._worker.request(op, params)
    except Exception:
      self._worker_failed = True
      raise


class ForcetekiObserver:
  """PyObserver exposing Forceteki's current-visibility tensor."""

  def __init__(self, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self.tensor = np.zeros(_OBSERVATION_TENSOR_SIZE, np.float32)
    self.dict = {"observation": self.tensor}

  def set_from(self, state, player):
    self.tensor[:] = state.observation_tensor(player)

  def string_from(self, state, player):
    del player
    return str(state)


def _trace_path(params):
  return str(params.get("trace_path") or os.environ.get(_TRACE_PATH_ENV) or "")


def _write_trace_entry(trace_path, entry):
  directory = os.path.dirname(trace_path)
  if directory:
    os.makedirs(directory, exist_ok=True)
  with open(trace_path, "a", encoding="utf-8") as trace_file:
    json.dump(entry, trace_file, separators=(",", ":"), sort_keys=True)
    trace_file.write("\n")


def _trace_legal_actions(legal_action_map):
  return [
      _trace_action(action, legal_action_map[action])
      for action in sorted(legal_action_map)
  ]


def _trace_action(action, legal_action):
  action_id = int(action)
  if isinstance(legal_action, dict):
    traced = dict(legal_action)
    traced["actionId"] = action_id
    return traced
  return {
      "actionId": action_id,
      "value": legal_action,
  }


def _action_player_id(legal_action):
  if not isinstance(legal_action, dict):
    return None
  raw = legal_action.get("rawAction") or legal_action.get("rawDecision") or {}
  return legal_action.get("playerId") or raw.get("playerId")


def _state_active_player_id(state):
  return state.get("activePlayerId")


def _build_state_view(state, player_id):
  if not isinstance(state, dict) or not player_id:
    return {}
  players = state.get("players", {})
  player = players.get(player_id)
  if not isinstance(player, dict):
    return {"playerId": player_id}
  opponent_id = next(
      (candidate for candidate in sorted(players) if candidate != player_id),
      None)
  opponent = players.get(opponent_id, {}) if opponent_id else {}
  prompt = player.get("prompt", {})
  return {
      "playerId": player_id,
      "opponentId": opponent_id,
      "turnNumber": state.get("roundNumber"),
      "phase": state.get("phase"),
      "initiativePlayerId": state.get("initiativePlayerId"),
      "myBaseHealth": _base_health(player.get("base")),
      "opponentBaseHealth": _base_health(opponent.get("base")),
      "myResourcesAvailable": player.get("availableResources"),
      "myResourcesTotal": player.get("resourcesTotal"),
      "opponentResourcesAvailable": opponent.get("availableResources"),
      "opponentResourcesTotal": opponent.get("resourcesTotal"),
      "myHandCount": player.get("handCount"),
      "opponentHandCount": opponent.get("handCount"),
      "myDeckCount": player.get("deckCount"),
      "opponentDeckCount": opponent.get("deckCount"),
      "myDiscardCount": player.get("discardCount"),
      "opponentDiscardCount": opponent.get("discardCount"),
      "myUnits": _zone_cards(player, ("groundArena", "spaceArena")),
      "opponentUnits": _zone_cards(opponent, ("groundArena", "spaceArena")),
      "myHand": _zone_cards(player, ("hand",)),
      "myDiscard": _zone_cards(player, ("discard",)),
      "opponentDiscard": _zone_cards(opponent, ("discard",)),
      "menuTitle": prompt.get("menuTitle"),
      "promptTitle": prompt.get("promptTitle"),
      "promptType": prompt.get("promptType"),
  }


def _zone_cards(player, zones):
  cards = []
  if not isinstance(player, dict):
    return cards
  for zone in zones:
    cards.extend(_card_view(card) for card in player.get(zone, []))
  return cards


def _card_view(card):
  if not isinstance(card, dict):
    return card
  keys = (
      "uuid", "id", "name", "zone", "damage", "hp", "remainingHp", "power",
      "cost", "controllerId", "ownerId", "type", "printedType", "arena",
      "aspects", "traits", "keywords", "exhausted", "selectable", "selected")
  view = {key: card.get(key) for key in keys if key in card}
  if "name" not in view and "internalName" in card:
    view["name"] = card.get("internalName")
  return view


def _base_health(base):
  if not isinstance(base, dict):
    return None
  if isinstance(base.get("remainingHp"), (int, float)):
    return base.get("remainingHp")
  if isinstance(base.get("hp"), (int, float)):
    return base.get("hp") - (base.get("damage") or 0)
  return None


class _NodeWorker:
  """Newline-delimited JSON client for Forceteki's simulation worker."""

  def __init__(self, params):
    self._seq = 0
    worker_path = _worker_path(params)
    forceteki_path = _forceteki_path(params)
    self._process = subprocess.Popen(
        ["node", worker_path],
        cwd=forceteki_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1)
    with _WORKER_REGISTRY_LOCK:
      _LIVE_WORKERS.add(self)

  def request(self, op: str, params: dict[str, Any] | None = None):
    self._seq += 1
    request = {"seq": self._seq, "op": op, "params": params or {}}
    assert self._process.stdin is not None
    assert self._process.stdout is not None
    self._process.stdin.write(json.dumps(request) + "\n")
    self._process.stdin.flush()
    line = self._process.stdout.readline()
    if not line:
      stderr = self._process.stderr.read() if self._process.stderr else ""
      raise RuntimeError(f"Forceteki worker exited before response: {stderr}")
    response = json.loads(line)
    if response.get("seq") != self._seq:
      raise RuntimeError(f"Forceteki worker sequence mismatch: {response}")
    if not response.get("ok"):
      raise RuntimeError(response.get("error", "Forceteki worker request failed"))
    return response["result"]

  def close(self):
    process = getattr(self, "_process", None)
    if process is None:
      with _WORKER_REGISTRY_LOCK:
        _LIVE_WORKERS.discard(self)
      return
    try:
      if process.poll() is None:
        try:
          self.request("close", {})
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
          pass
        process.terminate()
        try:
          process.wait(timeout=1)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait(timeout=1)
      for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
          try:
            stream.close()
          except OSError:
            pass
    finally:
      self._process = None
      with _WORKER_REGISTRY_LOCK:
        _LIVE_WORKERS.discard(self)


def close_all_workers():
  with _WORKER_REGISTRY_LOCK:
    pools = list(_WORKER_POOLS.values())
    workers = list(_LIVE_WORKERS)
    _WORKER_POOLS.clear()
  for pool in pools:
    pool.close()
  for worker in workers:
    worker.close()


class _NodeWorkerPool:
  """Bounded process-local pool for reusable Forceteki Node workers."""

  def __init__(self, params, size):
    self._params = dict(params)
    self._size = size
    self._idle = []
    self._live_count = 0
    self._closed = False
    self._condition = threading.Condition()

  def acquire(self):
    while True:
      with self._condition:
        if self._closed:
          raise RuntimeError("Forceteki worker pool is closed")
        if self._idle:
          return self._idle.pop()
        if self._live_count < self._size:
          self._live_count += 1
          break
        self._condition.wait()
    try:
      return _NodeWorker(self._params)
    except Exception:
      with self._condition:
        self._live_count -= 1
        self._condition.notify()
      raise

  def release(self, worker, discard=False):
    if worker is None:
      return
    should_close = discard
    with self._condition:
      if self._closed:
        should_close = True
      elif not should_close and getattr(worker, "_process", None) is not None:
        self._idle.append(worker)
        self._condition.notify()
        return
    worker.close()
    with self._condition:
      self._live_count = max(0, self._live_count - 1)
      self._condition.notify()

  def close(self):
    with self._condition:
      self._closed = True
      idle = list(self._idle)
      self._idle = []
      self._live_count = max(0, self._live_count - len(idle))
      self._condition.notify_all()
    for worker in idle:
      worker.close()


def _worker_pool(params):
  size = _worker_pool_size(params)
  if size < 0:
    raise ValueError("Forceteki worker_pool_size must be non-negative")
  if size == 0:
    return None
  key = (_NodeWorker, _worker_path(params), _forceteki_path(params))
  with _WORKER_REGISTRY_LOCK:
    pool = _WORKER_POOLS.get(key)
    if pool is None:
      pool = _NodeWorkerPool(params, size)
      _WORKER_POOLS[key] = pool
    elif pool._size != size:  # pylint: disable=protected-access
      existing_size = pool._size  # pylint: disable=protected-access
      raise ValueError(
          "Conflicting Forceteki worker_pool_size values for the same worker "
          f"configuration: existing={existing_size}, requested={size}")
    return pool


def _worker_pool_size(params):
  value = params.get("worker_pool_size")
  if value is None:
    value = os.environ.get("FORCETEKI_WORKER_POOL_SIZE", 0)
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"Invalid Forceteki worker_pool_size: {value}") from exc


def _forceteki_path(params):
  return str(params.get("forceteki_path") or
             os.environ.get("FORCETEKI_PATH") or
             "/Users/jbuttner/proj/home/forceteki")


def _worker_path(params):
  return str(params.get("worker_path") or
             os.environ.get("FORCETEKI_WORKER_PATH") or
             os.path.join(_forceteki_path(params),
                          "build/server/game/simulation/SimulationWorker.js"))


def _reset_params(params):
  reset = {
      "seed": str(params.get("seed") or os.environ.get("FORCETEKI_SEED") or
                  "open-spiel"),
  }
  card_data_path = params.get("card_data_path") or os.environ.get(
      "FORCETEKI_CARD_DATA_PATH")
  if card_data_path:
    reset["cardDataPath"] = str(card_data_path)
  first_player_id = params.get("preselected_first_player_id") or os.environ.get(
      "FORCETEKI_FIRST_PLAYER_ID")
  if first_player_id:
    reset["preselectedFirstPlayerId"] = str(first_player_id)
  return reset


atexit.register(close_all_workers)
pyspiel.register_game(_GAME_TYPE, ForcetekiGame)
