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
from open_spiel.python.examples.forceteki_psro_utils import _debug_trace_path
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
flags.DEFINE_string("forceteki_seed", "", "Optional Forceteki worker seed.")
flags.DEFINE_integer("max_episode_steps", 1000,
                     "OpenSpiel-side cap for Forceteki rollout length.")

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
flags.DEFINE_bool("debug", False,
                  "Write Forceteki decision trace entries to trace.ndjson.")
flags.DEFINE_string("debug_dir", "forceteki_psro_debug",
                    "Directory used for timestamped --debug trace runs.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if FLAGS.n_players != 2:
    raise app.UsageError("Forceteki SWU only supports --n_players=2")

  np.random.seed(FLAGS.seed)
  if FLAGS.forceteki_seed:
    os.environ["FORCETEKI_SEED"] = FLAGS.forceteki_seed
  if FLAGS.debug:
    trace_path = _debug_trace_path(FLAGS.debug_dir)
    os.environ["FORCETEKI_TRACE_PATH"] = trace_path
    print(f"Forceteki debug trace: {trace_path}")

  env = None
  _install_cleanup_signal_handlers()
  try:
    game = pyspiel.load_game_as_turn_based(
        FLAGS.game_name,
        {
            "players": FLAGS.n_players,
            "max_game_length": FLAGS.max_episode_steps,
        })
    env = rl_environment.Environment(game)
    oracle, agents = init_oracle(env, FLAGS)
    run_psro(env, oracle, agents, FLAGS)
  finally:
    if env is not None:
      _close_state(getattr(env, "_state", None))
    forceteki.close_all_workers()


if __name__ == "__main__":
  app.run(main)
