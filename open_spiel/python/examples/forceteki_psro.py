# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Rollout-only PSRO runner for Forceteki SWU.

This is intentionally narrower than psro_v2_example.py. Forceteki v1 delegates
state transitions to a live Node worker, so clone/tree-based policy aggregation
and exact exploitability analysis are not available yet. This runner keeps the
sampled PSRO meta-game loop and RL oracle training, then reports rollout-based
meta-game estimates and meta-strategies.
"""

# pylint: disable=unused-import

import os

from absl import app
from absl import flags
import numpy as np

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.examples import forceteki_psro_artifacts
from open_spiel.python.examples.forceteki_psro_oracles import ForcetekiPPOOracle
from open_spiel.python.examples.forceteki_psro_oracles import ForcetekiTraceRLOracle
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiActionFactorizer
from open_spiel.python.examples.forceteki_psro_ppo import ForcetekiPPOPolicy
from open_spiel.python.examples.forceteki_psro_ppo import _FactoredActorCritic
from open_spiel.python.examples.forceteki_psro_responders import init_dqn_responder
from open_spiel.python.examples.forceteki_psro_responders import init_oracle
from open_spiel.python.examples.forceteki_psro_responders import init_pg_responder
from open_spiel.python.examples.forceteki_psro_responders import init_ppo_responder
from open_spiel.python.examples.forceteki_psro_responders import print_solver_summary
from open_spiel.python.examples.forceteki_psro_responders import run_psro
from open_spiel.python.examples.forceteki_psro_solver import DiagnosticPSROSolver
from open_spiel.python.examples.forceteki_psro_solver import _sample_episode_with_diagnostics
from open_spiel.python.examples.forceteki_psro_utils import _INVALID_LOGIT
from open_spiel.python.examples.forceteki_psro_utils import _NONE_TOKEN
from open_spiel.python.examples.forceteki_psro_utils import _close_state
from open_spiel.python.examples.forceteki_psro_utils import _debug_trace_dir
from open_spiel.python.examples.forceteki_psro_utils import _install_cleanup_signal_handlers
from open_spiel.python.examples.forceteki_psro_utils import _legal_action_map
from open_spiel.python.examples.forceteki_psro_utils import _prompt_payload
from open_spiel.python.examples.forceteki_psro_utils import _raw_action
from open_spiel.python.examples.forceteki_psro_utils import _stable_bucket
from open_spiel.python.examples.forceteki_psro_utils import _state_payload

# Registers python_forceteki_swu.
from open_spiel.python.games import forceteki  # pylint: disable=unused-import


FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_forceteki_swu", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_integer("max_episode_steps", 1000,
                     "OpenSpiel-side cap for Forceteki rollout length.")
flags.DEFINE_integer("forceteki_worker_pool_size", 0,
                     "Max reusable Forceteki Node workers. Zero disables "
                     "worker pooling.")
flags.DEFINE_integer("parallel_eval_workers", 1,
                     "Threads used for evaluation rollouts. Values greater "
                     "than one require --forceteki_worker_pool_size to be at "
                     "least this large.")

flags.DEFINE_string("meta_strategy_method", "uniform",
                    "Meta-strategy method: uniform, nash, alpharank, or prd.")
flags.DEFINE_integer("gpsro_iterations", 1, "Number of PSRO iterations.")
flags.DEFINE_integer("sims_per_entry", 1,
                     "Rollouts used to estimate each meta-game entry.")
flags.DEFINE_bool("rollout_diagnostics", True,
                  "Print per-entry terminal/cap diagnostics for evaluation "
                  "rollouts.")
flags.DEFINE_integer("number_policies_selected", 1,
                     "New strategies trained at each PSRO iteration.")
flags.DEFINE_bool("symmetric_game", False,
                  "Whether to treat the game as symmetric.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Strategy selector used for oracle training.")
flags.DEFINE_string("rectifier", "", "Rectifier: '' or 'rectified'.")

flags.DEFINE_string("oracle_type", "PG",
                    "RL oracle type. Supported: PG, DQN, PPO.")
flags.DEFINE_integer("number_training_episodes", 10,
                     "Training episodes per RL policy per PSRO iteration.")
flags.DEFINE_integer("forceteki_crash_retry_limit", 10,
                     "Consecutive training rollout crashes to capture and "
                     "retry before aborting. Zero disables crash recovery.")
flags.DEFINE_float("self_play_proportion", 0.0,
                   "Probability of replacing sampled opponents with self-play.")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size.")
flags.DEFINE_integer("n_hidden_layers", 2, "Number of hidden layers.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("sigma", 0.0, "Policy copy Gaussian noise.")
flags.DEFINE_string("optimizer_str", "adam", "Optimizer: adam or sgd.")

flags.DEFINE_string("loss_str", "qpg", "Policy-gradient loss.")
flags.DEFINE_integer("num_q_before_pi", 8, "Critic updates before policy update.")
flags.DEFINE_float("entropy_cost", 0.001, "Entropy regularization cost.")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate.")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")
flags.DEFINE_integer("update_target_network_every", 1000,
                     "DQN target network update period.")
flags.DEFINE_integer("learn_every", 10, "DQN learning period.")

flags.DEFINE_integer("ppo_steps_per_batch", 128,
                     "Approximate PPO decision steps per update.")
flags.DEFINE_integer("ppo_num_minibatches", 4, "PPO minibatches per update.")
flags.DEFINE_integer("ppo_update_epochs", 4, "PPO update epochs.")
flags.DEFINE_float("ppo_learning_rate", 2.5e-4, "PPO learning rate.")
flags.DEFINE_float("ppo_gamma", 0.99, "PPO discount factor.")
flags.DEFINE_float("ppo_gae_lambda", 0.95, "PPO GAE lambda.")
flags.DEFINE_float("ppo_clip_coef", 0.2, "PPO ratio clipping coefficient.")
flags.DEFINE_float("ppo_entropy_coef", 0.01, "PPO entropy coefficient.")
flags.DEFINE_float("ppo_value_coef", 0.5, "PPO value coefficient.")
flags.DEFINE_float("ppo_max_grad_norm", 0.5, "PPO gradient clipping norm.")
flags.DEFINE_float("ppo_target_kl", None, "Optional PPO target KL.")
flags.DEFINE_string("ppo_device", "cpu", "PPO torch device.")
flags.DEFINE_integer("ppo_intent_vocab_size", 128,
                     "Hash buckets for Forceteki action intents.")
flags.DEFINE_integer("ppo_kind_vocab_size", 32,
                     "Hash buckets for Forceteki action kinds.")
flags.DEFINE_integer("ppo_control_vocab_size", 512,
                     "Hash buckets for Forceteki prompt/control fields.")
flags.DEFINE_integer("ppo_card_vocab_size", 256,
                     "Card pointer buckets for Forceteki actions.")

flags.DEFINE_integer("seed", 1, "Numpy seed.")
flags.DEFINE_bool("verbose", True, "Print iteration details.")
flags.DEFINE_bool("progress", True,
                  "Print line-based progress updates during PSRO runs.")
flags.DEFINE_float("progress_interval_seconds", 30.0,
                   "Minimum seconds between non-final progress updates.")
flags.DEFINE_bool("log_to_file", True,
                  "Also write Forceteki PSRO run output to a log file.")
flags.DEFINE_string("forceteki_log_dir", "forceteki_psro_logs",
                    "Directory used for the default Forceteki PSRO log file.")
flags.DEFINE_string("log_file", "",
                    "Explicit Forceteki PSRO log file path. Overrides "
                    "--forceteki_log_dir when set.")
flags.DEFINE_string("debug", "off",
                    "Forceteki decision trace mode: full, minimal, or off.")
flags.DEFINE_string("debug_dir", "forceteki_psro_debug",
                    "Directory used for timestamped --debug trace runs.")
flags.DEFINE_string("output_dir", "",
                    "Directory where reloadable PSRO artifacts are written.")
flags.DEFINE_string("resume_from", "",
                    "Existing artifact directory to resume from.")
flags.DEFINE_string("init_policy_from", "",
                    "Policy checkpoint or artifact directory used to seed a "
                    "new run without restoring full PSRO state.")
flags.DEFINE_string("decks_path", "",
                    "JSON file containing two Forceteki decklists.")
flags.DEFINE_string("player0_deck_path", "",
                    "JSON file containing player 0's Forceteki decklist.")
flags.DEFINE_string("player1_deck_path", "",
                    "JSON file containing player 1's Forceteki decklist.")
flags.DEFINE_string("deck_pool_path", "",
                    "Directory of SWUDB-format Forceteki deck JSON files. "
                    "Overrides FORCETEKI_DECK_POOL_PATH when set.")


_DEBUG_MODES = frozenset(("full", "minimal", "off"))


def _debug_mode_from_flags(flags_obj):
  debug_value = str(getattr(flags_obj, "debug", "off")).lower()
  if debug_value not in _DEBUG_MODES:
    raise app.UsageError(
        "--debug must be one of: full, minimal, off")
  return debug_value


def _game_params_from_flags(flags_obj):
  debug_mode = _debug_mode_from_flags(flags_obj)
  crash_retry_limit = int(getattr(flags_obj, "forceteki_crash_retry_limit", 10))
  trace_mode = debug_mode
  if crash_retry_limit > 0 and debug_mode == "off":
    trace_mode = "minimal"
  params = {
      "players": flags_obj.n_players,
      "max_game_length": flags_obj.max_episode_steps,
      "worker_pool_size": flags_obj.forceteki_worker_pool_size,
      "seed": str(flags_obj.seed),
      "trace_mode": trace_mode,
      "trace_mode_explicit": True,
  }
  if trace_mode != "off":
    params["trace_dir"] = _debug_trace_dir(flags_obj.debug_dir)
  deck_pool_path = (
      flags_obj.deck_pool_path or os.environ.get("FORCETEKI_DECK_POOL_PATH"))
  if deck_pool_path:
    params["deck_pool_path"] = deck_pool_path
  if flags_obj.decks_path:
    params["decks_path"] = flags_obj.decks_path
  if flags_obj.player0_deck_path:
    params["player0_deck_path"] = flags_obj.player0_deck_path
  if flags_obj.player1_deck_path:
    params["player1_deck_path"] = flags_obj.player1_deck_path
  return params


def _seed_agents_from_policy(flags_obj, env, agents):
  if not flags_obj.init_policy_from:
    return agents
  init_path = flags_obj.init_policy_from
  if os.path.isdir(init_path):
    population = forceteki_psro_artifacts.load_policy_population(
        init_path, env, device=flags_obj.ppo_device)
    solver_state = forceteki_psro_artifacts.load_solver_state(init_path)
    bot_policy = forceteki_psro_artifacts.bot_policy_dict(solver_state)
    seeded_agents = list(agents)
    for player_entry in bot_policy["players"]:
      player_id = int(player_entry["player_id"])
      policy_index = int(player_entry["policy_index"])
      seeded_agents[player_id] = population[player_id][policy_index]
      seeded_agents[player_id].freeze()
    return seeded_agents

  policy_obj = forceteki_psro_artifacts.load_single_policy(
      init_path, env, device=flags_obj.ppo_device)
  seeded_agents = list(agents)
  seeded_agents[policy_obj.player_id] = policy_obj
  policy_obj.freeze()
  return seeded_agents


def _restore_resume_state(flags_obj, env):
  """Loads resume artifacts, or allows a missing directory for fresh output."""
  if not flags_obj.resume_from:
    return None, None, None

  if os.path.exists(flags_obj.resume_from):
    if not os.path.isdir(flags_obj.resume_from):
      raise app.UsageError(
          f"--resume_from exists but is not a directory: "
          f"{flags_obj.resume_from}")
    restored_policies = forceteki_psro_artifacts.load_policy_population(
        flags_obj.resume_from, env, device=flags_obj.ppo_device)
    restored_solver_state = forceteki_psro_artifacts.load_solver_state(
        flags_obj.resume_from)
    forceteki_psro_artifacts.restore_rng_state(flags_obj.resume_from)
    restored_agents = [
        player_policies[0] for player_policies in restored_policies
    ]
    return restored_policies, restored_solver_state, restored_agents

  if flags_obj.output_dir:
    print(
        f"Resume directory not found: {flags_obj.resume_from}. "
        f"Starting a fresh run and writing artifacts to {flags_obj.output_dir}.")
    return None, None, None

  raise app.UsageError(
      f"--resume_from does not exist and --output_dir is empty: "
      f"{flags_obj.resume_from}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if FLAGS.n_players != 2:
    raise app.UsageError("Forceteki SWU only supports --n_players=2")
  if FLAGS.parallel_eval_workers < 1:
    raise app.UsageError("--parallel_eval_workers must be at least 1")
  if FLAGS.forceteki_worker_pool_size < 0:
    raise app.UsageError("--forceteki_worker_pool_size must be non-negative")
  if FLAGS.forceteki_crash_retry_limit < 0:
    raise app.UsageError("--forceteki_crash_retry_limit must be non-negative")
  if (FLAGS.parallel_eval_workers > 1 and
      FLAGS.forceteki_worker_pool_size < FLAGS.parallel_eval_workers):
    raise app.UsageError(
        "--parallel_eval_workers > 1 requires "
        "--forceteki_worker_pool_size >= --parallel_eval_workers")
  if FLAGS.resume_from and FLAGS.init_policy_from:
    raise app.UsageError("--resume_from and --init_policy_from are exclusive")
  if ((FLAGS.output_dir or FLAGS.resume_from or FLAGS.init_policy_from) and
      FLAGS.oracle_type.upper() != "PPO"):
    raise app.UsageError(
        "Reloadable Forceteki artifacts currently support --oracle_type=PPO")
  if bool(FLAGS.player0_deck_path) != bool(FLAGS.player1_deck_path):
    raise app.UsageError(
        "--player0_deck_path and --player1_deck_path must be provided together")
  if FLAGS.decks_path and (FLAGS.player0_deck_path or FLAGS.player1_deck_path):
    raise app.UsageError(
        "Use either --decks_path or per-player deck paths, not both")
  resolved_deck_pool_path = (
      FLAGS.deck_pool_path or os.environ.get("FORCETEKI_DECK_POOL_PATH"))
  if resolved_deck_pool_path and (FLAGS.decks_path or FLAGS.player0_deck_path):
    raise app.UsageError(
        "Use either deck_pool_path/FORCETEKI_DECK_POOL_PATH or fixed deck "
        "paths, not both")

  np.random.seed(FLAGS.seed)
  os.environ["FORCETEKI_SEED"] = str(FLAGS.seed)
  env = None
  interrupt_controller = _install_cleanup_signal_handlers(
      wait_for_storage=bool(FLAGS.output_dir))
  try:
    game_params = _game_params_from_flags(FLAGS)
    trace_dir = game_params.get("trace_dir")
    if trace_dir:
      print(f"Forceteki debug traces: {trace_dir}")
    game = pyspiel.load_game_as_turn_based(
        FLAGS.game_name, game_params)
    env = rl_environment.Environment(game)
    oracle, agents = init_oracle(env, FLAGS)
    restored_policies, restored_solver_state, restored_agents = (
        _restore_resume_state(FLAGS, env))
    if restored_agents is not None:
      agents = restored_agents
    else:
      agents = _seed_agents_from_policy(FLAGS, env, agents)

    run_psro(
        env,
        oracle,
        agents,
        FLAGS,
        game_params,
        restored_policies=restored_policies,
        restored_solver_state=restored_solver_state,
        interrupt_controller=interrupt_controller)
  finally:
    if env is not None:
      _close_state(getattr(env, "_state", None))
    forceteki.close_all_workers()


if __name__ == "__main__":
  app.run(main)
