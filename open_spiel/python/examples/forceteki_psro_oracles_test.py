# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import io
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.examples import forceteki_psro_oracles
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy


class FakeState:

  def __init__(self, trace_path):
    self._trace_path = trace_path
    self.closed = False

  def close(self):
    self.closed = True


def _oracle(debug_dir, retry_limit=10):
  oracle = forceteki_psro_oracles.ForcetekiTraceRLOracle.__new__(
      forceteki_psro_oracles.ForcetekiTraceRLOracle)
  oracle._env = SimpleNamespace(_state=None)
  oracle._seed = 7
  oracle._forceteki_crash_retry_limit = retry_limit
  oracle._forceteki_crash_streak = 0
  oracle._forceteki_crash_debug_dir = debug_dir
  oracle._forceteki_crash_output = io.StringIO()
  oracle._forceteki_last_failed_trace_path = ""
  oracle._forceteki_trace_training_rollout = 0
  oracle._progress_reporter = None
  oracle._progress_iteration = None
  oracle._progress_total_iterations = None
  return oracle


class ForcetekiPsroOraclesTest(absltest.TestCase):

  def test_training_rollout_crash_saves_artifacts_and_retries(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      trace_path = os.path.join(temp_dir, "worker-0000.ndjson")
      with open(trace_path, "w", encoding="utf-8") as trace_file:
        trace_file.write("trace-entry\n")
      oracle = _oracle(temp_dir, retry_limit=10)
      states = []

      def rollout_once(self, game, agents, **kwargs):
        del game, agents, kwargs
        state = FakeState(trace_path)
        self._env._state = state
        states.append(state)
        if len(states) == 1:
          raise RuntimeError("boom")

      with mock.patch.object(rl_oracle.RLOracle, "_rollout", rollout_once):
        oracle._rollout(object(), [])

      errors_dir = os.path.join(temp_dir, "errors")
      [artifact_name] = os.listdir(errors_dir)
      artifact_dir = os.path.join(errors_dir, artifact_name)
      with open(os.path.join(artifact_dir, "stack_trace.txt"),
                encoding="utf-8") as stack_trace_file:
        stack_trace = stack_trace_file.read()
      with open(os.path.join(artifact_dir, "trace.ndjson"),
                encoding="utf-8") as copied_trace_file:
        copied_trace = copied_trace_file.read()

      self.assertLen(states, 2)
      self.assertTrue(states[0].closed)
      self.assertTrue(states[1].closed)
      self.assertIsNone(oracle._env._state)
      self.assertEqual(oracle._forceteki_trace_training_rollout, 1)
      self.assertEqual(oracle._forceteki_crash_streak, 0)
      self.assertIn("RuntimeError: boom", stack_trace)
      self.assertEqual(copied_trace, "trace-entry\n")
      self.assertIn("retrying", oracle._forceteki_crash_output.getvalue())

  def test_training_rollout_crash_retry_cap_reraises(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      trace_path = os.path.join(temp_dir, "worker-0000.ndjson")
      with open(trace_path, "w", encoding="utf-8") as trace_file:
        trace_file.write("trace-entry\n")
      oracle = _oracle(temp_dir, retry_limit=2)
      calls = []

      def rollout_once(self, game, agents, **kwargs):
        del game, agents, kwargs
        self._env._state = FakeState(trace_path)
        calls.append(1)
        raise RuntimeError("still broken")

      with mock.patch.object(rl_oracle.RLOracle, "_rollout", rollout_once):
        with self.assertRaisesRegex(RuntimeError, "still broken"):
          oracle._rollout(object(), [])

      self.assertLen(calls, 2)
      self.assertLen(os.listdir(os.path.join(temp_dir, "errors")), 2)
      self.assertEqual(oracle._forceteki_crash_streak, 2)
      self.assertIn("aborting", oracle._forceteki_crash_output.getvalue())

  def test_ppo_restore_snapshots_discards_failed_attempt_transitions(self):
    oracle = forceteki_psro_oracles.ForcetekiPPOOracle.__new__(
        forceteki_psro_oracles.ForcetekiPPOOracle)
    agent = ForcetekiPPOPolicy.__new__(ForcetekiPPOPolicy)
    original_pending = {"action": 1}
    original_buffer = [{"action": 0}]
    agent._pending = original_pending
    agent._pending_reward = 2.5
    agent._buffer = list(original_buffer)

    snapshots = oracle._snapshot_ppo_agents([agent])
    agent._pending = {"action": 99}
    agent._pending_reward = 7.0
    agent._buffer.append({"action": 100})

    oracle._restore_ppo_agent_snapshots(snapshots)

    self.assertIs(agent._pending, original_pending)
    self.assertEqual(agent._pending_reward, 2.5)
    self.assertEqual(agent._buffer, original_buffer)


if __name__ == "__main__":
  absltest.main()
