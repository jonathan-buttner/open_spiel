# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import io
import signal

from absl.testing import absltest

from open_spiel.python.examples import forceteki_psro_utils


class ForcetekiPsroUtilsTest(absltest.TestCase):

  def test_first_sigint_with_storage_requests_graceful_shutdown(self):
    output = io.StringIO()
    controller = forceteki_psro_utils._GracefulInterruptController(
        wait_for_storage=True, output=output)

    controller.handle_signal(signal.SIGINT, None)

    self.assertTrue(controller.shutdown_requested)
    self.assertIn("will stop after the next artifact save", output.getvalue())
    self.assertIn("Press Ctrl+C again", output.getvalue())

  def test_second_sigint_with_storage_raises_keyboard_interrupt(self):
    controller = forceteki_psro_utils._GracefulInterruptController(
        wait_for_storage=True, output=io.StringIO())

    controller.handle_signal(signal.SIGINT, None)

    with self.assertRaises(KeyboardInterrupt):
      controller.handle_signal(signal.SIGINT, None)

  def test_first_sigint_without_storage_raises_keyboard_interrupt(self):
    controller = forceteki_psro_utils._GracefulInterruptController(
        wait_for_storage=False, output=io.StringIO())

    with self.assertRaises(KeyboardInterrupt):
      controller.handle_signal(signal.SIGINT, None)

    self.assertFalse(controller.shutdown_requested)

  def test_sigterm_exits_with_signal_status(self):
    controller = forceteki_psro_utils._GracefulInterruptController(
        wait_for_storage=True, output=io.StringIO())

    with self.assertRaises(SystemExit) as context:
      controller.handle_signal(signal.SIGTERM, None)

    self.assertEqual(context.exception.code, 128 + signal.SIGTERM)


if __name__ == "__main__":
  absltest.main()
