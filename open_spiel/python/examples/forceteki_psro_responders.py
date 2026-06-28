# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Responder factories and runner helpers for the Forceteki PSRO example."""

import os
import sys
import time

from absl import app

from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.examples import forceteki_psro_artifacts
from open_spiel.python.examples import forceteki_psro_progress
from open_spiel.python.examples.forceteki_psro_oracles import ForcetekiPPOOracle
from open_spiel.python.examples.forceteki_psro_oracles import ForcetekiTraceRLOracle
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy
from open_spiel.python.examples.forceteki_psro_solver import DiagnosticPSROSolver


def _debug_enabled(flags_obj):
  debug_value = getattr(flags_obj, "debug", "off")
  return bool(debug_value) and str(debug_value).lower() != "off"


def init_pg_responder(env, flags_obj):
  """Initializes a policy-gradient RL oracle and frozen initial policies."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.PGPolicy
  agent_kwargs = {
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "loss_str": flags_obj.loss_str,
      "loss_class": False,
      "hidden_layers_sizes": (
          [flags_obj.hidden_layer_size] * flags_obj.n_hidden_layers),
      "entropy_cost": flags_obj.entropy_cost,
      "critic_learning_rate": flags_obj.critic_learning_rate,
      "pi_learning_rate": flags_obj.pi_learning_rate,
      "num_critic_before_pi": flags_obj.num_q_before_pi,
      "optimizer_str": flags_obj.optimizer_str,
  }
  oracle = ForcetekiTraceRLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=flags_obj.number_training_episodes,
      self_play_proportion=flags_obj.self_play_proportion,
      sigma=flags_obj.sigma,
      seed=flags_obj.seed,
      crash_retry_limit=flags_obj.forceteki_crash_retry_limit)
  agents = [agent_class(env, player_id, **agent_kwargs)
            for player_id in range(flags_obj.n_players)]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_dqn_responder(env, flags_obj):
  """Initializes a DQN RL oracle and frozen initial policies."""
  state_representation_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.DQNPolicy
  agent_kwargs = {
      "state_representation_size": state_representation_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": (
          [flags_obj.hidden_layer_size] * flags_obj.n_hidden_layers),
      "batch_size": flags_obj.batch_size,
      "learning_rate": flags_obj.dqn_learning_rate,
      "update_target_network_every": flags_obj.update_target_network_every,
      "learn_every": flags_obj.learn_every,
      "optimizer_str": flags_obj.optimizer_str,
  }
  oracle = ForcetekiTraceRLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=flags_obj.number_training_episodes,
      self_play_proportion=flags_obj.self_play_proportion,
      sigma=flags_obj.sigma,
      seed=flags_obj.seed,
      crash_retry_limit=flags_obj.forceteki_crash_retry_limit)
  agents = [agent_class(env, player_id, **agent_kwargs)
            for player_id in range(flags_obj.n_players)]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_ppo_responder(env, flags_obj):
  """Initializes a factored PPO oracle and frozen initial policies."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = ForcetekiPPOPolicy
  agent_kwargs = {
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": (
          [flags_obj.hidden_layer_size] * flags_obj.n_hidden_layers),
      "steps_per_batch": flags_obj.ppo_steps_per_batch,
      "num_minibatches": flags_obj.ppo_num_minibatches,
      "update_epochs": flags_obj.ppo_update_epochs,
      "learning_rate": flags_obj.ppo_learning_rate,
      "gamma": flags_obj.ppo_gamma,
      "gae_lambda": flags_obj.ppo_gae_lambda,
      "clip_coef": flags_obj.ppo_clip_coef,
      "entropy_coef": flags_obj.ppo_entropy_coef,
      "value_coef": flags_obj.ppo_value_coef,
      "max_grad_norm": flags_obj.ppo_max_grad_norm,
      "target_kl": flags_obj.ppo_target_kl,
      "device": flags_obj.ppo_device,
      "intent_vocab_size": flags_obj.ppo_intent_vocab_size,
      "kind_vocab_size": flags_obj.ppo_kind_vocab_size,
      "control_vocab_size": flags_obj.ppo_control_vocab_size,
      "card_vocab_size": flags_obj.ppo_card_vocab_size,
  }
  oracle = ForcetekiPPOOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=flags_obj.number_training_episodes,
      self_play_proportion=flags_obj.self_play_proportion,
      sigma=flags_obj.sigma,
      seed=flags_obj.seed,
      crash_retry_limit=flags_obj.forceteki_crash_retry_limit)
  agents = [agent_class(env, player_id, **agent_kwargs)
            for player_id in range(flags_obj.n_players)]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_oracle(env, flags_obj):
  oracle_type = flags_obj.oracle_type.upper()
  if oracle_type == "PG":
    return init_pg_responder(env, flags_obj)
  if oracle_type == "DQN":
    return init_dqn_responder(env, flags_obj)
  if oracle_type == "PPO":
    return init_ppo_responder(env, flags_obj)
  raise app.UsageError(f"Unsupported --oracle_type={flags_obj.oracle_type}")


def print_solver_summary(solver, iteration, elapsed_seconds, flags_obj,
                         output=None):
  output = output or sys.stdout
  meta_game = solver.get_meta_game()
  meta_probabilities = solver.get_meta_strategies()
  policies = solver.get_policies()
  policy_counts = [len(player_policies) for player_policies in policies]

  print(f"Iteration: {iteration}", file=output)
  print(f"Elapsed seconds: {elapsed_seconds:.2f}", file=output)
  print(f"Policies per player: {policy_counts}", file=output)
  print(f"Meta strategies: {meta_probabilities}", file=output)
  if flags_obj.verbose:
    print(f"Meta game: {meta_game}", file=output)
  print("-" * 80, file=output, flush=True)


def _shutdown_requested(interrupt_controller):
  return bool(getattr(interrupt_controller, "shutdown_requested", False))


def _open_output(flags_obj):
  if not getattr(flags_obj, "log_to_file", False):
    return sys.stdout, None

  log_path = forceteki_psro_progress.resolve_log_path(flags_obj)
  log_dir = os.path.dirname(log_path)
  if log_dir:
    os.makedirs(log_dir, exist_ok=True)
  log_file = open(log_path, "a", encoding="utf-8")
  log_output = forceteki_psro_progress.TimestampedLineOutput(log_file)
  output = forceteki_psro_progress.TeeOutput(sys.stdout, log_output)
  print(f"Forceteki PSRO log: {log_path}", file=output, flush=True)
  return output, log_output


def run_psro(env, oracle, agents, flags_obj, game_params,
             restored_policies=None, restored_solver_state=None,
             interrupt_controller=None):
  output, log_file = _open_output(flags_obj)
  try:
    return _run_psro(
        env,
        oracle,
        agents,
        flags_obj,
        game_params,
        output,
        restored_policies=restored_policies,
        restored_solver_state=restored_solver_state,
        interrupt_controller=interrupt_controller)
  finally:
    output.flush()
    if log_file is not None:
      log_file.close()


def _run_psro(env, oracle, agents, flags_obj, game_params, output,
              restored_policies=None, restored_solver_state=None,
              interrupt_controller=None):
  progress_reporter = forceteki_psro_progress.TextProgressReporter(
      enabled=getattr(flags_obj, "progress", True),
      interval_seconds=getattr(flags_obj, "progress_interval_seconds", 30.0),
      output=output)
  if hasattr(oracle, "set_crash_recovery_context"):
    oracle.set_crash_recovery_context(
        debug_dir=game_params.get("trace_dir", ""), output=output)
  if hasattr(oracle, "set_progress_reporter"):
    oracle.set_progress_reporter(progress_reporter)

  solver = DiagnosticPSROSolver(
      env.game,
      oracle,
      initial_policies=agents,
      training_strategy_selector=flags_obj.training_strategy_selector,
      rectifier=flags_obj.rectifier,
      sims_per_entry=flags_obj.sims_per_entry,
      number_policies_selected=flags_obj.number_policies_selected,
      meta_strategy_method=flags_obj.meta_strategy_method,
      prd_iterations=50000,
      prd_gamma=1e-10,
      sample_from_marginals=True,
      symmetric_game=flags_obj.symmetric_game,
      rollout_diagnostics=flags_obj.rollout_diagnostics or _debug_enabled(
          flags_obj),
      parallel_eval_workers=flags_obj.parallel_eval_workers,
      seed=flags_obj.seed,
      progress_reporter=progress_reporter,
      output=output,
      defer_initial_update=restored_solver_state is not None)

  start_time = time.time()
  start_iteration = 0
  if restored_solver_state is not None:
    forceteki_psro_artifacts.restore_solver_state(
        solver, restored_policies, restored_solver_state)
    start_iteration = int(restored_solver_state.get(
        "completed_iterations", restored_solver_state.get("iterations", 0)))

  print_solver_summary(
      solver, start_iteration, time.time() - start_time, flags_obj, output)
  if flags_obj.output_dir:
    forceteki_psro_artifacts.save_run_artifacts(
        flags_obj.output_dir, solver, flags_obj, game_params, start_iteration)
    if _shutdown_requested(interrupt_controller):
      return solver

  for iteration in range(start_iteration + 1, flags_obj.gpsro_iterations + 1):
    if hasattr(oracle, "set_progress_context"):
      oracle.set_progress_context(iteration, flags_obj.gpsro_iterations)
    solver.set_progress_context(iteration, flags_obj.gpsro_iterations)
    progress_reporter.start(
        "psro", iteration=f"{iteration}/{flags_obj.gpsro_iterations}")
    solver.iteration()
    iteration_end_time = time.time()
    completed_iterations = iteration - start_iteration
    remaining_iterations = flags_obj.gpsro_iterations - iteration
    done_fields = {"iteration": f"{iteration}/{flags_obj.gpsro_iterations}"}
    if completed_iterations > 0:
      average_iteration_seconds = (
          (iteration_end_time - start_time) / completed_iterations)
      done_fields["run_eta"] = forceteki_psro_progress.format_duration(
          average_iteration_seconds * remaining_iterations)
    progress_reporter.done(
        "psro", **done_fields)
    print_solver_summary(
        solver, iteration, iteration_end_time - start_time, flags_obj, output)
    if flags_obj.output_dir:
      forceteki_psro_artifacts.save_run_artifacts(
          flags_obj.output_dir, solver, flags_obj, game_params, iteration)
      if _shutdown_requested(interrupt_controller):
        return solver
  return solver
