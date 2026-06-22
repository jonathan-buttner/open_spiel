# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Line-based progress reporting for the Forceteki PSRO example."""

import os
import sys
import time


DEFAULT_LOG_DIR = "forceteki_psro_logs"
DEFAULT_LOG_FILENAME = "forceteki_psro.log"


class TeeOutput:
  """File-like writer that duplicates text to multiple output streams."""

  def __init__(self, *outputs):
    self._outputs = outputs

  def write(self, text):
    for output in self._outputs:
      output.write(text)
    return len(text)

  def flush(self):
    for output in self._outputs:
      output.flush()

  def close(self):
    for output in self._outputs:
      close = getattr(output, "close", None)
      if close is not None and output is not sys.stdout:
        close()


def resolve_log_path(flags_obj):
  """Returns the Forceteki PSRO log path implied by logging flags."""
  log_file = getattr(flags_obj, "log_file", "")
  if log_file:
    return log_file
  log_dir = getattr(flags_obj, "forceteki_log_dir", DEFAULT_LOG_DIR)
  return os.path.join(log_dir, DEFAULT_LOG_FILENAME)


def format_duration(seconds):
  seconds = max(0, int(seconds))
  days, remainder = divmod(seconds, 24 * 3600)
  hours, remainder = divmod(remainder, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"


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
    self._phase_start_times = {}

  @property
  def enabled(self):
    return self._enabled

  def start(self, phase, **fields):
    self._phase_start_times[phase] = self._time_fn()
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
    parts.append(f"eta={self._eta(phase, current, total, now)}")
    print(" ".join(parts), file=self._output, flush=True)
    return True

  def _emit(self, phase, fields, status):
    if not self._enabled:
      return
    fields = dict(fields)
    run_eta = fields.pop("run_eta", None)
    parts = [f"[progress] {phase}"]
    parts.extend(_format_fields(fields))
    parts.append(status)
    parts.append(f"elapsed={self._elapsed()}")
    if status == "done":
      parts.append(f"eta={format_duration(0)}")
    else:
      parts.append("eta=pending")
    if run_eta is not None:
      parts.append(f"run_eta={run_eta}")
    print(" ".join(parts), file=self._output, flush=True)

  def _elapsed(self):
    return format_duration(self._time_fn() - self._start_time)

  def _eta(self, phase, current, total, now):
    if total <= 0:
      return "pending"
    if current >= total:
      return format_duration(0)
    if current <= 0:
      return "pending"
    phase_start_time = self._phase_start_times.get(phase, self._start_time)
    elapsed = now - phase_start_time
    if elapsed <= 0:
      return "pending"
    remaining = max(0, total - current)
    return format_duration(elapsed * remaining / current)


def _format_fields(fields):
  return [f"{key}={value}" for key, value in fields.items()]
