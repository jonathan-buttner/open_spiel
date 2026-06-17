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

import os
import time

from absl import app
from absl import flags
import numpy as np

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy

# Registers python_forceteki_swu.
from open_spiel.python.games import forceteki  # pylint: disable=unused-import


FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_forceteki_swu", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_string("forceteki_seed", "", "Optional Forceteki worker seed.")
flags.DEFINE_integer("max_episode_steps", 200,
                     "OpenSpiel-side cap for Forceteki rollout length.")

flags.DEFINE_string("meta_strategy_method", "uniform",
                    "Meta-strategy method: uniform, nash, alpharank, or prd.")
flags.DEFINE_integer("gpsro_iterations", 1, "Number of PSRO iterations.")
flags.DEFINE_integer("sims_per_entry", 1,
                     "Rollouts used to estimate each meta-game entry.")
flags.DEFINE_integer("number_policies_selected", 1,
                     "New strategies trained at each PSRO iteration.")
flags.DEFINE_bool("symmetric_game", False,
                  "Whether to treat the game as symmetric.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Strategy selector used for oracle training.")
flags.DEFINE_string("rectifier", "", "Rectifier: '' or 'rectified'.")

flags.DEFINE_string("oracle_type", "PG",
                    "RL oracle type. Supported: PG, DQN. PPO is not wired yet.")
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

flags.DEFINE_integer("seed", 1, "Numpy seed.")
flags.DEFINE_bool("verbose", True, "Print iteration details.")


def init_pg_responder(env):
  """Initializes a policy-gradient RL oracle and frozen initial policies."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.PGPolicy
  agent_kwargs = {
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "loss_str": FLAGS.loss_str,
      "loss_class": False,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "entropy_cost": FLAGS.entropy_cost,
      "critic_learning_rate": FLAGS.critic_learning_rate,
      "pi_learning_rate": FLAGS.pi_learning_rate,
      "num_critic_before_pi": FLAGS.num_q_before_pi,
      "optimizer_str": FLAGS.optimizer_str,
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)
  agents = [agent_class(env, player_id, **agent_kwargs)
            for player_id in range(FLAGS.n_players)]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_dqn_responder(env):
  """Initializes a DQN RL oracle and frozen initial policies."""
  state_representation_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.DQNPolicy
  agent_kwargs = {
      "state_representation_size": state_representation_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "learning_rate": FLAGS.dqn_learning_rate,
      "update_target_network_every": FLAGS.update_target_network_every,
      "learn_every": FLAGS.learn_every,
      "optimizer_str": FLAGS.optimizer_str,
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)
  agents = [agent_class(env, player_id, **agent_kwargs)
            for player_id in range(FLAGS.n_players)]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_oracle(env):
  oracle_type = FLAGS.oracle_type.upper()
  if oracle_type == "PG":
    return init_pg_responder(env)
  if oracle_type == "DQN":
    return init_dqn_responder(env)
  if oracle_type == "PPO":
    raise app.UsageError(
        "PPO is not wired as a PSRO oracle yet. Use PG/DQN for this runner, "
        "or add a ForcetekiPPOOracle that trains one responder against sampled "
        "population opponents.")
  raise app.UsageError(f"Unsupported --oracle_type={FLAGS.oracle_type}")


def print_solver_summary(solver, iteration, elapsed_seconds):
  meta_game = solver.get_meta_game()
  meta_probabilities = solver.get_meta_strategies()
  policies = solver.get_policies()
  policy_counts = [len(player_policies) for player_policies in policies]

  print(f"Iteration: {iteration}")
  print(f"Elapsed seconds: {elapsed_seconds:.2f}")
  print(f"Policies per player: {policy_counts}")
  print(f"Meta strategies: {meta_probabilities}")
  if FLAGS.verbose:
    print(f"Meta game: {meta_game}")
  print("-" * 80)


def run_psro(env, oracle, agents):
  solver = psro_v2.PSROSolver(
      env.game,
      oracle,
      initial_policies=agents,
      training_strategy_selector=FLAGS.training_strategy_selector,
      rectifier=FLAGS.rectifier,
      sims_per_entry=FLAGS.sims_per_entry,
      number_policies_selected=FLAGS.number_policies_selected,
      meta_strategy_method=FLAGS.meta_strategy_method,
      prd_iterations=50000,
      prd_gamma=1e-10,
      sample_from_marginals=True,
      symmetric_game=FLAGS.symmetric_game)

  start_time = time.time()
  print_solver_summary(solver, 0, time.time() - start_time)
  for iteration in range(1, FLAGS.gpsro_iterations + 1):
    solver.iteration()
    print_solver_summary(solver, iteration, time.time() - start_time)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if FLAGS.n_players != 2:
    raise app.UsageError("Forceteki SWU only supports --n_players=2")

  np.random.seed(FLAGS.seed)
  if FLAGS.forceteki_seed:
    os.environ["FORCETEKI_SEED"] = FLAGS.forceteki_seed

  game = pyspiel.load_game_as_turn_based(
      FLAGS.game_name,
      {
          "players": FLAGS.n_players,
          "max_game_length": FLAGS.max_episode_steps,
      })
  env = rl_environment.Environment(game)
  oracle, agents = init_oracle(env)
  run_psro(env, oracle, agents)


if __name__ == "__main__":
  app.run(main)
