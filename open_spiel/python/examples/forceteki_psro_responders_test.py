# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from open_spiel.python.examples import forceteki_psro_responders


def _flags(**overrides):
  values = {
      "debug": False,
      "gpsro_iterations": 3,
      "meta_strategy_method": "uniform",
      "number_policies_selected": 1,
      "output_dir": "/tmp/forceteki_psro_test",
      "parallel_eval_workers": 1,
      "progress": False,
      "progress_interval_seconds": 30.0,
      "rectifier": "",
      "rollout_diagnostics": False,
      "seed": 1,
      "sims_per_entry": 1,
      "symmetric_game": False,
      "training_strategy_selector": "probabilistic",
      "verbose": False,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class FakeEnv:
  game = object()


class FakeInterruptController:

  def __init__(self):
    self.shutdown_requested = False


class FakeSolver:
  instances = []
  interrupt_controller = None

  def __init__(self, *args, **kwargs):
    del args, kwargs
    self.iteration_calls = 0
    self.progress_contexts = []
    FakeSolver.instances.append(self)

  def get_meta_game(self):
    return []

  def get_meta_strategies(self):
    return []

  def get_policies(self):
    return [["p0"], ["p1"]]

  def set_progress_context(self, iteration, total_iterations):
    self.progress_contexts.append((iteration, total_iterations))

  def iteration(self):
    self.iteration_calls += 1
    FakeSolver.interrupt_controller.shutdown_requested = True


class ForcetekiPsroRespondersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    FakeSolver.instances = []
    FakeSolver.interrupt_controller = None

  def test_run_psro_stops_after_completed_iteration_artifact_save(self):
    interrupt_controller = FakeInterruptController()
    FakeSolver.interrupt_controller = interrupt_controller

    with mock.patch.object(
        forceteki_psro_responders,
        "DiagnosticPSROSolver",
        FakeSolver), mock.patch.object(
            forceteki_psro_responders.forceteki_psro_artifacts,
            "save_run_artifacts") as save_run_artifacts:
      solver = forceteki_psro_responders.run_psro(
          FakeEnv(),
          oracle=object(),
          agents=["a0", "a1"],
          flags_obj=_flags(),
          game_params={"players": 2},
          interrupt_controller=interrupt_controller)

    self.assertIs(solver, FakeSolver.instances[0])
    self.assertEqual(solver.iteration_calls, 1)
    self.assertEqual(
        [call.args[4] for call in save_run_artifacts.call_args_list],
        [0, 1])
    self.assertEqual(solver.progress_contexts, [(1, 3)])

  def test_run_psro_writes_log_file_when_enabled(self):
    temp_dir = tempfile.mkdtemp()
    log_path = os.path.join(temp_dir, "run.log")

    with mock.patch.object(
        forceteki_psro_responders,
        "DiagnosticPSROSolver",
        FakeSolver):
      solver = forceteki_psro_responders.run_psro(
          FakeEnv(),
          oracle=object(),
          agents=["a0", "a1"],
          flags_obj=_flags(
              gpsro_iterations=0,
              log_to_file=True,
              log_file=log_path,
              output_dir=""),
          game_params={"players": 2})

    self.assertIs(solver, FakeSolver.instances[0])
    with open(log_path, encoding="utf-8") as log_file:
      contents = log_file.read()
    self.assertIn(f"Forceteki PSRO log: {log_path}", contents)
    self.assertIn("Iteration: 0", contents)
    self.assertIn("Policies per player: [1, 1]", contents)


if __name__ == "__main__":
  absltest.main()
