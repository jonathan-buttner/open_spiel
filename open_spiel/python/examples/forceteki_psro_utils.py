# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Utilities for the Forceteki PSRO example."""

import hashlib
import os
import signal
import sys
from datetime import datetime
from datetime import timezone


_INVALID_LOGIT = -1e9
_NONE_TOKEN = 0


def _stable_bucket(value, vocab_size):
  """Maps a structured value to a stable nonzero token bucket."""
  if not value or vocab_size <= 1:
    return _NONE_TOKEN
  encoded = repr(value).encode("utf-8")
  digest = hashlib.sha256(encoded).hexdigest()
  return 1 + (int(digest, 16) % (vocab_size - 1))


def _state_payload(state):
  return getattr(state, "_state", {}).get("state", {})


def _prompt_payload(state, player_id):
  if not player_id:
    return {}
  return _state_payload(state).get("players", {}).get(player_id, {}).get(
      "prompt", {})


def _raw_action(legal_action):
  if not isinstance(legal_action, dict):
    return {}
  return legal_action.get("rawAction") or legal_action.get("rawDecision") or {}


def _legal_action_map(state):
  if hasattr(state, "forceteki_legal_actions"):
    return state.forceteki_legal_actions()
  raw_state = getattr(state, "_state", {})
  legal_actions = raw_state.get("legalActions", [])
  if legal_actions and isinstance(legal_actions[0], dict):
    return {slot: legal_action for slot, legal_action in enumerate(legal_actions)}
  return {int(action): int(action) for action in state.legal_actions()}


def _close_state(state):
  close = getattr(state, "close", None)
  if close is not None:
    close()


class _GracefulInterruptController:
  """Tracks cooperative shutdown requests from Ctrl+C."""

  def __init__(self, wait_for_storage=False, output=None):
    self._wait_for_storage = bool(wait_for_storage)
    self._output = output or sys.stdout
    self._sigint_count = 0
    self._shutdown_requested = False

  @property
  def shutdown_requested(self):
    return self._shutdown_requested

  def handle_signal(self, signum, _frame):
    if signum == signal.SIGINT:
      self._sigint_count += 1
      if self._sigint_count == 1 and self._wait_for_storage:
        self._shutdown_requested = True
        print(
            "Ctrl+C received; Forceteki PSRO will stop after the next "
            "artifact save. Press Ctrl+C again to stop immediately.",
            file=self._output,
            flush=True)
        return
      raise KeyboardInterrupt
    raise SystemExit(128 + signum)

  def install(self):
    signal.signal(signal.SIGINT, self.handle_signal)
    signal.signal(signal.SIGTERM, self.handle_signal)
    return self


def _install_cleanup_signal_handlers(wait_for_storage=False, output=None):
  return _GracefulInterruptController(
      wait_for_storage=wait_for_storage,
      output=output).install()


def _debug_trace_path(debug_dir):
  timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
  timestamp = timestamp.replace("+00:00", "Z").replace(":", "-")
  debug_run_dir = os.path.join(
      debug_dir, f"{timestamp}_{os.getpid()}")
  os.makedirs(debug_run_dir, exist_ok=True)
  return os.path.join(debug_run_dir, "trace.ndjson")
