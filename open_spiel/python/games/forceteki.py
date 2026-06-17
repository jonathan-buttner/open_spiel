# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Forceteki Star Wars: Unlimited RL environment wrapper."""

import json
import os
import subprocess
from typing import Any

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_NUM_DISTINCT_ACTIONS = 512
_OBSERVATION_TENSOR_SIZE = 4096
_DEFAULT_MAX_GAME_LENGTH = 1000

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
    })

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_DEFAULT_MAX_GAME_LENGTH)


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

  def __init__(self, game, params, checkpoint=None, move_number=0):
    super().__init__(game)
    self._params = dict(params)
    self._worker = _NodeWorker(params)
    if checkpoint is None:
      self._state = self._worker.request("reset", _reset_params(params))
    else:
      self._state = self._worker.request("restore_checkpoint",
                                         {"checkpoint": checkpoint})
    self._move_number = move_number
    self._max_game_length = int(params.get("max_game_length",
                                           _DEFAULT_MAX_GAME_LENGTH))

  def current_player(self):
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return int(self._state["currentPlayer"])

  def _legal_actions(self, player):
    del player
    return list(self._state["legalActions"])

  def _apply_action(self, action):
    self._state = self._worker.request("step", {"action": int(action)})
    self._move_number += 1

  def _action_to_string(self, player, action):
    del player
    return self._state.get("actionStrings", {}).get(str(int(action)), str(action))

  def is_terminal(self):
    return bool(self._state["isTerminal"]) or (
        self._move_number >= self._max_game_length)

  def returns(self):
    return [float(value) for value in self._state["returns"]]

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
      checkpoint = self._worker.request("export_checkpoint")
      clone = ForcetekiState(self.get_game(), self._params)
      for action in checkpoint.get("actionHistory", []):
        clone.apply_action(int(action))
      return clone

    checkpoint = self._worker.request("export_checkpoint")
    return ForcetekiState(
        self.get_game(),
        self._params,
        checkpoint=checkpoint,
        move_number=self._move_number)

  def __deepcopy__(self, memo):
    del memo
    return self.clone()

  def __del__(self):
    worker = getattr(self, "_worker", None)
    if worker is not None:
      worker.close()


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
      return
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
        stream.close()
    self._process = None


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


pyspiel.register_game(_GAME_TYPE, ForcetekiGame)
