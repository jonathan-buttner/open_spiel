# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Line-based progress reporting for the Forceteki PSRO example."""

import sys
import time


def _format_elapsed(seconds):
  seconds = max(0, int(seconds))
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TextProgressReporter:
  """Prints durable, throttled progress lines to stdout."""

  def __init__(self, enabled=True, interval_seconds=30.0, output=None,
               time_fn=None):
    self._enabled = enabled
    self._interval_seconds = max(0.0, float(interval_seconds))
    self._output = output or sys.stdout
    self._time_fn = time_fn or time.time
    self._start_time = self._time_fn()
    self._last_update_time = {}

  @property
  def enabled(self):
    return self._enabled

  def start(self, phase, **fields):
    self._emit(phase, fields, "started")

  def done(self, phase, **fields):
    self._emit(phase, fields, "done")

  def update(self, phase, unit, current, total, force=False, **fields):
    if not self._enabled:
      return False

    now = self._time_fn()
    key = (phase, unit)
    if not force and key in self._last_update_time:
      if now - self._last_update_time[key] < self._interval_seconds:
        return False
    self._last_update_time[key] = now

    current = int(current)
    total = int(total)
    if total > 0:
      percent = f"{(100.0 * current / total):.1f}%"
    else:
      percent = "n/a"

    parts = [f"[progress] {phase}"]
    parts.extend(_format_fields(fields))
    parts.append(f"{unit}={current}/{total}")
    parts.append(percent)
    parts.append(f"elapsed={self._elapsed()}")
    print(" ".join(parts), file=self._output, flush=True)
    return True

  def _emit(self, phase, fields, status):
    if not self._enabled:
      return
    parts = [f"[progress] {phase}"]
    parts.extend(_format_fields(fields))
    parts.append(status)
    parts.append(f"elapsed={self._elapsed()}")
    print(" ".join(parts), file=self._output, flush=True)

  def _elapsed(self):
    return _format_elapsed(self._time_fn() - self._start_time)


def _format_fields(fields):
  return [f"{key}={value}" for key, value in fields.items()]
