# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import io

from absl.testing import absltest
import numpy as np

from open_spiel.python.examples import forceteki_psro_progress
from open_spiel.python.examples.forceteki_psro_oracles import ForcetekiTraceRLOracle
from open_spiel.python.examples.forceteki_psro_solver import DiagnosticPSROSolver


class FakeClock:

  def __init__(self):
    self.now = 0.0

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class FakeProgressReporter:

  def __init__(self):
    self.enabled = True
    self.updates = []

  def update(self, phase, unit, current, total, force=False, **fields):
    self.updates.append({
        "phase": phase,
        "unit": unit,
        "current": current,
        "total": total,
        "force": force,
        "fields": fields,
    })


class ForcetekiPsroProgressTest(absltest.TestCase):

  def test_start_update_and_done_lines_include_elapsed_and_eta(self):
    clock = FakeClock()
    output = io.StringIO()
    reporter = forceteki_psro_progress.TextProgressReporter(
        output=output, time_fn=clock)

    reporter.start("psro", iteration="1/2")
    clock.advance(65)
    reporter.update("training", "episodes", 3, 10, iteration="1/2")
    reporter.done("psro", iteration="1/2")

    self.assertEqual(
        output.getvalue().splitlines(),
        [
            ("[progress] psro iteration=1/2 started elapsed=0d 00:00:00 "
             "eta=pending"),
            ("[progress] training iteration=1/2 episodes=3/10 30.0% "
             "elapsed=0d 00:01:05 eta=0d 00:02:31"),
            ("[progress] psro iteration=1/2 done elapsed=0d 00:01:05 "
             "eta=0d 00:00:00"),
        ])

  def test_duration_format_includes_days(self):
    self.assertEqual(
        forceteki_psro_progress.format_duration(65), "0d 00:01:05")
    self.assertEqual(
        forceteki_psro_progress.format_duration(90061), "1d 01:01:01")

  def test_update_throttles_and_force_bypasses_throttle(self):
    clock = FakeClock()
    output = io.StringIO()
    reporter = forceteki_psro_progress.TextProgressReporter(
        interval_seconds=30.0, output=output, time_fn=clock)

    self.assertTrue(reporter.update("training", "episodes", 1, 10))
    self.assertFalse(reporter.update("training", "episodes", 2, 10))
    self.assertTrue(
        reporter.update("training", "episodes", 3, 10, force=True))
    clock.advance(30)
    self.assertTrue(reporter.update("training", "episodes", 4, 10))

    self.assertLen(output.getvalue().splitlines(), 3)

  def test_disabled_reporter_and_zero_total(self):
    clock = FakeClock()
    disabled_output = io.StringIO()
    disabled = forceteki_psro_progress.TextProgressReporter(
        enabled=False, output=disabled_output, time_fn=clock)
    disabled.start("psro", iteration="1/1")
    self.assertFalse(disabled.update("evaluation", "rollouts", 1, 0))
    disabled.done("psro", iteration="1/1")
    self.assertEmpty(disabled_output.getvalue())

    output = io.StringIO()
    reporter = forceteki_psro_progress.TextProgressReporter(
        output=output, time_fn=clock)
    reporter.update("evaluation", "rollouts", 1, 0)
    self.assertIn("rollouts=1/0 n/a", output.getvalue())
    self.assertIn("eta=pending", output.getvalue())

  def test_done_line_can_include_run_eta(self):
    clock = FakeClock()
    output = io.StringIO()
    reporter = forceteki_psro_progress.TextProgressReporter(
        output=output, time_fn=clock)

    clock.advance(2 * 24 * 3600)
    reporter.done(
        "psro",
        iteration="2/10",
        run_eta=forceteki_psro_progress.format_duration(8 * 24 * 3600))

    self.assertEqual(
        output.getvalue().strip(),
        ("[progress] psro iteration=2/10 done elapsed=2d 00:00:00 "
         "eta=0d 00:00:00 run_eta=8d 00:00:00"))

  def test_training_progress_uses_episode_credits(self):
    reporter = FakeProgressReporter()
    oracle = ForcetekiTraceRLOracle.__new__(ForcetekiTraceRLOracle)
    oracle._progress_reporter = reporter
    oracle._progress_iteration = 1
    oracle._progress_total_iterations = 2
    oracle._number_training_episodes = 2

    oracle._training_progress(np.array([[1], [2]]))
    oracle._training_progress(np.array([[3], [3]]))

    self.assertEqual([update["current"] for update in reporter.updates],
                     [3, 6])
    self.assertEqual([update["total"] for update in reporter.updates], [6, 6])
    self.assertFalse(reporter.updates[0]["force"])
    self.assertTrue(reporter.updates[1]["force"])
    self.assertEqual(reporter.updates[0]["fields"]["iteration"], "1/2")

  def test_evaluation_progress_counts_missing_rollouts(self):
    reporter = FakeProgressReporter()
    solver = DiagnosticPSROSolver.__new__(DiagnosticPSROSolver)
    solver._num_players = 2
    solver._progress_reporter = reporter
    solver._progress_iteration = 1
    solver._progress_total_iterations = 2
    solver._evaluation_rollouts_done = 0
    solver._evaluation_rollouts_total = 8

    meta_games = [
        np.full((2, 2), np.nan),
        np.full((2, 2), np.nan),
    ]
    meta_games[0][0, 0] = 0.0
    meta_games[1][0, 0] = 0.0

    missing = solver._count_missing_profiles(
        meta_games,
        total_number_policies=[2, 2],
        number_older_policies=[1, 1],
        number_new_policies=[1, 1])
    self.assertEqual(missing, 4)

    solver._evaluation_progress((1, 0))
    solver._evaluation_rollouts_done = 7
    solver._evaluation_progress((1, 1))

    self.assertEqual(reporter.updates[0]["current"], 1)
    self.assertEqual(reporter.updates[0]["total"], 8)
    self.assertFalse(reporter.updates[0]["force"])
    self.assertTrue(reporter.updates[1]["force"])
    self.assertEqual(reporter.updates[1]["fields"]["profile"], (1, 1))


if __name__ == "__main__":
  absltest.main()
