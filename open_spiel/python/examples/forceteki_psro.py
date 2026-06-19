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

import hashlib
import itertools
import os
import signal
import time

from absl import app
from absl import flags
import numpy as np
import torch
from torch import nn
from torch import optim

import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import utils

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

_INVALID_LOGIT = -1e9
_NONE_TOKEN = 0


def _stable_bucket(value, vocab_size):
  """Maps a structured value to a stable nonzero token bucket."""
  if not value or vocab_size <= 1:
    return _NONE_TOKEN
  encoded = repr(value).encode("utf-8")
  digest = hashlib.sha256(encoded).hexdigest()
  return 1 + (int(digest, 16) % (vocab_size - 1))


def _state_payload(state):
  return getattr(state, "_state", {}).get("state", {})


def _prompt_payload(state, player_id):
  if not player_id:
    return {}
  return _state_payload(state).get("players", {}).get(player_id, {}).get(
      "prompt", {})


def _raw_action(legal_action):
  if not isinstance(legal_action, dict):
    return {}
  return legal_action.get("rawAction") or legal_action.get("rawDecision") or {}


def _legal_action_map(state):
  if hasattr(state, "forceteki_legal_actions"):
    return state.forceteki_legal_actions()
  raw_state = getattr(state, "_state", {})
  legal_actions = raw_state.get("legalActions", [])
  if legal_actions and isinstance(legal_actions[0], dict):
    return {slot: legal_action for slot, legal_action in enumerate(legal_actions)}
  return {int(action): int(action) for action in state.legal_actions()}


def _close_state(state):
  close = getattr(state, "close", None)
  if close is not None:
    close()


def _install_cleanup_signal_handlers():
  def _handle_signal(signum, _frame):
    if signum == signal.SIGINT:
      raise KeyboardInterrupt
    raise SystemExit(128 + signum)

  signal.signal(signal.SIGINT, _handle_signal)
  signal.signal(signal.SIGTERM, _handle_signal)


class ForcetekiActionFactorizer:
  """Extracts PPO action factors from structured Forceteki legal actions."""

  _CARD_FIELDS = ("base", "leader")
  _CARD_ZONES = ("hand", "discard", "resources", "groundArena", "spaceArena")

  def __init__(self, intent_vocab_size, kind_vocab_size, control_vocab_size,
               card_vocab_size):
    self.intent_vocab_size = intent_vocab_size
    self.kind_vocab_size = kind_vocab_size
    self.control_vocab_size = control_vocab_size
    self.card_vocab_size = card_vocab_size

  def factor(self, state, action_slot, legal_action):
    del action_slot
    if not isinstance(legal_action, dict):
      return {
          "intent": _NONE_TOKEN,
          "kind": _stable_bucket("legacy-slot", self.kind_vocab_size),
          "control": _NONE_TOKEN,
          "card": _NONE_TOKEN,
      }

    raw = _raw_action(legal_action)
    kind = legal_action.get("kind") or raw.get("kind")
    card = legal_action.get("card") or {}
    prompt = _prompt_payload(state, raw.get("playerId") or
                             legal_action.get("playerId"))
    intent = {
        "kind": kind,
        "buttonArg": raw.get("buttonArg"),
        "buttonText": raw.get("buttonText"),
        "cardSelected": card.get("selected"),
        "cardZone": card.get("zone"),
        "promptType": prompt.get("promptType"),
    }
    control = {
        "kind": raw.get("kind") or kind,
        "command": raw.get("command"),
        "method": raw.get("method"),
        "buttonArg": raw.get("buttonArg"),
        "buttonText": raw.get("buttonText"),
        "value": raw.get("value"),
        "statefulPromptType": raw.get("statefulPromptType"),
        "promptTitle": prompt.get("promptTitle"),
        "promptType": prompt.get("promptType"),
        "cardSelected": card.get("selected"),
    }
    card_uuid = (
        raw.get("cardUuid") or
        card.get("uuid") or
        legal_action.get("sourceCardUuid"))
    return {
        "intent": _stable_bucket(intent, self.intent_vocab_size),
        "kind": _stable_bucket(kind, self.kind_vocab_size),
        "control": _stable_bucket(control, self.control_vocab_size),
        "card": self._card_slot(state, card_uuid),
    }

  def pack(self, state):
    legal_actions = _legal_action_map(state)
    return {
        "factors": {
            int(slot): self.factor(state, int(slot), legal_action)
            for slot, legal_action in legal_actions.items()
        },
        "legal_actions": sorted(int(slot) for slot in legal_actions),
    }

  def _card_slot(self, state, card_uuid):
    if not card_uuid:
      return _NONE_TOKEN
    simulation_state = _state_payload(state)
    players = simulation_state.get("players", {})
    next_slot = 1
    for player_id in sorted(players):
      player = players[player_id]
      for field in self._CARD_FIELDS:
        card = player.get(field)
        if isinstance(card, dict) and card.get("uuid") == card_uuid:
          return min(next_slot, self.card_vocab_size - 1)
        next_slot += 1
      for zone in self._CARD_ZONES:
        for card in player.get(zone, []):
          if isinstance(card, dict) and card.get("uuid") == card_uuid:
            return min(next_slot, self.card_vocab_size - 1)
          next_slot += 1
    return _NONE_TOKEN


class _FactoredActorCritic(nn.Module):
  """Shared trunk with separate heads for factored Forceteki actions."""

  def __init__(self, obs_size, hidden_layers_sizes, num_actions,
               intent_vocab_size, kind_vocab_size, control_vocab_size,
               card_vocab_size):
    super().__init__()
    layers = []
    previous_size = obs_size
    for hidden_size in hidden_layers_sizes:
      layers.append(nn.Linear(previous_size, hidden_size))
      layers.append(nn.Tanh())
      previous_size = hidden_size
    self.trunk = nn.Sequential(*layers)
    self.intent_head = nn.Linear(previous_size, intent_vocab_size)
    self.kind_head = nn.Linear(previous_size, kind_vocab_size)
    self.control_head = nn.Linear(previous_size, control_vocab_size)
    self.card_head = nn.Linear(previous_size, card_vocab_size)
    self.action_head = nn.Linear(previous_size, num_actions)
    self.value_head = nn.Linear(previous_size, 1)

  def forward(self, obs):
    hidden = self.trunk(obs)
    return {
        "intent": self.intent_head(hidden),
        "kind": self.kind_head(hidden),
        "control": self.control_head(hidden),
        "card": self.card_head(hidden),
        "action": self.action_head(hidden),
        "value": self.value_head(hidden).squeeze(-1),
    }


class ForcetekiPPOPolicy(policy.Policy):
  """Factored PPO policy over structured Forceteki legal actions."""

  def __init__(self, env, player_id, **kwargs):
    super().__init__(env.game, player_id)
    self._env = env
    self._player_id = player_id
    self._kwargs = dict(kwargs)
    self._frozen = False
    self._device = torch.device(kwargs.get("device", "cpu"))
    self._num_actions = kwargs["num_actions"]
    self._obs_size = kwargs["info_state_size"]
    self._factorizer = ForcetekiActionFactorizer(
        kwargs["intent_vocab_size"],
        kwargs["kind_vocab_size"],
        kwargs["control_vocab_size"],
        kwargs["card_vocab_size"])
    self._network = _FactoredActorCritic(
        self._obs_size,
        kwargs["hidden_layers_sizes"],
        self._num_actions,
        kwargs["intent_vocab_size"],
        kwargs["kind_vocab_size"],
        kwargs["control_vocab_size"],
        kwargs["card_vocab_size"]).to(self._device)
    self._optimizer = optim.Adam(
        self._network.parameters(), lr=kwargs["learning_rate"], eps=1e-5)
    self._buffer = []
    self._pending = None
    self._pending_reward = 0.0

  @property
  def player_id(self):
    return self._player_id

  def freeze(self):
    self._frozen = True

  def unfreeze(self):
    self._frozen = False

  def is_frozen(self):
    return self._frozen

  def get_weights(self):
    return {
        key: value.detach().cpu().clone()
        for key, value in self._network.state_dict().items()
    }

  def copy_with_noise(self, sigma=0.0):
    copied = ForcetekiPPOPolicy(self._env, self._player_id, **self._kwargs)
    state_dict = self.get_weights()
    if sigma > 0.0:
      state_dict = {
          key: value + sigma * torch.randn_like(value)
          if value.is_floating_point() else value
          for key, value in state_dict.items()
      }
    copied._network.load_state_dict(state_dict)
    copied.unfreeze()
    return copied

  def action_probabilities(self, state, player_id=None):
    del player_id
    pack = self._factorizer.pack(state)
    if not pack["legal_actions"]:
      return {}
    obs = self._obs_tensor(state)
    with torch.no_grad():
      outputs = self._network(obs)
      probabilities = {
          action: float(self._action_probability(outputs, pack, action))
          for action in pack["legal_actions"]
      }
    total = sum(probabilities.values())
    if total <= 0.0:
      uniform = 1.0 / len(probabilities)
      return {action: uniform for action in probabilities}
    return {action: prob / total for action, prob in probabilities.items()}

  def training_action(self, state):
    if self._pending is not None:
      self._finish_pending(done=False)
    pack = self._factorizer.pack(state)
    obs = self._obs_tensor(state)
    with torch.no_grad():
      outputs = self._network(obs)
      action, logprob, value = self._sample_action(outputs, pack)
    self._pending = {
        "obs": obs.squeeze(0).detach().cpu(),
        "pack": pack,
        "action": int(action),
        "logprob": float(logprob.detach().cpu()),
        "value": float(value.detach().cpu()),
    }
    self._pending_reward = 0.0
    return int(action)

  def add_pending_reward(self, reward):
    if self._pending is not None:
      self._pending_reward += float(reward)

  def finish_episode(self):
    if self._pending is not None:
      self._finish_pending(done=True)
    if len(self._buffer) >= self._kwargs["steps_per_batch"]:
      self._learn()

  def finish_training(self):
    if self._pending is not None:
      self._finish_pending(done=True)
    if self._buffer:
      self._learn()

  def _finish_pending(self, done):
    transition = dict(self._pending)
    transition["reward"] = self._pending_reward
    transition["done"] = bool(done)
    self._buffer.append(transition)
    self._pending = None
    self._pending_reward = 0.0

  def _obs_tensor(self, state):
    obs = np.asarray(
        state.information_state_tensor(self._player_id), dtype=np.float32)
    return torch.as_tensor(obs, dtype=torch.float32,
                           device=self._device).view(1, -1)

  def _learn(self):
    batch = self._buffer
    self._buffer = []
    obs = torch.stack([sample["obs"] for sample in batch]).to(self._device)
    old_logprobs = torch.tensor(
        [sample["logprob"] for sample in batch],
        dtype=torch.float32,
        device=self._device)
    old_values = torch.tensor(
        [sample["value"] for sample in batch],
        dtype=torch.float32,
        device=self._device)
    rewards = torch.tensor(
        [sample["reward"] for sample in batch],
        dtype=torch.float32,
        device=self._device)
    dones = torch.tensor(
        [sample["done"] for sample in batch],
        dtype=torch.float32,
        device=self._device)

    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for index in reversed(range(len(batch))):
      if index == len(batch) - 1:
        next_value = torch.tensor(0.0, device=self._device)
        next_non_terminal = 1.0 - dones[index]
      else:
        next_value = old_values[index + 1]
        next_non_terminal = 1.0 - dones[index]
      delta = (
          rewards[index] +
          self._kwargs["gamma"] * next_value * next_non_terminal -
          old_values[index])
      last_gae = (
          delta +
          self._kwargs["gamma"] * self._kwargs["gae_lambda"] *
          next_non_terminal * last_gae)
      advantages[index] = last_gae
    returns = advantages + old_values

    batch_indices = np.arange(len(batch))
    minibatch_size = max(1, min(
        len(batch),
        len(batch) // max(1, self._kwargs["num_minibatches"])))
    for _ in range(self._kwargs["update_epochs"]):
      np.random.shuffle(batch_indices)
      for start in range(0, len(batch), minibatch_size):
        indices = batch_indices[start:start + minibatch_size]
        new_logprobs, entropy, new_values = self._evaluate_samples(
            obs, batch, indices)
        logratio = new_logprobs - old_logprobs[indices]
        ratio = logratio.exp()
        mb_advantages = advantages[indices]
        if len(mb_advantages) > 1:
          mb_advantages = ((mb_advantages - mb_advantages.mean()) /
                           (mb_advantages.std() + 1e-8))
        pg_loss_1 = -mb_advantages * ratio
        pg_loss_2 = -mb_advantages * torch.clamp(
            ratio, 1.0 - self._kwargs["clip_coef"],
            1.0 + self._kwargs["clip_coef"])
        pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
        value_loss = 0.5 * ((new_values - returns[indices]) ** 2).mean()
        loss = (
            pg_loss -
            self._kwargs["entropy_coef"] * entropy.mean() +
            self._kwargs["value_coef"] * value_loss)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._network.parameters(), self._kwargs["max_grad_norm"])
        self._optimizer.step()

        target_kl = self._kwargs.get("target_kl")
        if target_kl is not None:
          approx_kl = ((ratio - 1.0) - logratio).mean()
          if float(approx_kl.detach().cpu()) > target_kl:
            return

  def _evaluate_samples(self, obs, batch, indices):
    logprobs = []
    entropies = []
    values = []
    for index in indices:
      outputs = self._network(obs[index:index + 1])
      logprob, entropy, value = self._action_logprob_entropy_value(
          outputs, batch[index]["pack"], batch[index]["action"])
      logprobs.append(logprob)
      entropies.append(entropy)
      values.append(value)
    return torch.stack(logprobs), torch.stack(entropies), torch.stack(values)

  def _sample_action(self, outputs, pack):
    intent_dist = self._dist(outputs["intent"], self._intent_mask(pack))
    intent = int(intent_dist.sample().item())
    kind_dist = self._dist(outputs["kind"], self._kind_mask(pack, intent))
    kind = int(kind_dist.sample().item())
    control_dist = self._dist(
        outputs["control"], self._control_mask(pack, intent, kind))
    control = int(control_dist.sample().item())
    card_dist = self._dist(
        outputs["card"], self._card_mask(pack, intent, kind, control))
    card = int(card_dist.sample().item())
    action_dist = self._dist(
        outputs["action"],
        self._action_mask(pack, intent, kind, control, card))
    action = int(action_dist.sample().item())
    logprob, _, value = self._action_logprob_entropy_value(
        outputs, pack, action)
    return action, logprob, value

  def _action_probability(self, outputs, pack, action):
    factors = pack["factors"][action]
    intent_dist = self._dist(outputs["intent"], self._intent_mask(pack))
    kind_dist = self._dist(
        outputs["kind"], self._kind_mask(pack, factors["intent"]))
    control_dist = self._dist(
        outputs["control"],
        self._control_mask(pack, factors["intent"], factors["kind"]))
    card_dist = self._dist(
        outputs["card"],
        self._card_mask(
            pack, factors["intent"], factors["kind"], factors["control"]))
    action_dist = self._dist(
        outputs["action"],
        self._action_mask(
            pack, factors["intent"], factors["kind"], factors["control"],
            factors["card"]))
    return (
        intent_dist.probs[0, factors["intent"]] *
        kind_dist.probs[0, factors["kind"]] *
        control_dist.probs[0, factors["control"]] *
        card_dist.probs[0, factors["card"]] *
        action_dist.probs[0, action])

  def _action_logprob_entropy_value(self, outputs, pack, action):
    factors = pack["factors"][action]
    intent_dist = self._dist(outputs["intent"], self._intent_mask(pack))
    kind_dist = self._dist(
        outputs["kind"], self._kind_mask(pack, factors["intent"]))
    control_dist = self._dist(
        outputs["control"],
        self._control_mask(pack, factors["intent"], factors["kind"]))
    card_dist = self._dist(
        outputs["card"],
        self._card_mask(
            pack, factors["intent"], factors["kind"], factors["control"]))
    action_dist = self._dist(
        outputs["action"],
        self._action_mask(
            pack, factors["intent"], factors["kind"], factors["control"],
            factors["card"]))
    logprob = (
        intent_dist.log_prob(torch.tensor([factors["intent"]],
                                         device=self._device))[0] +
        kind_dist.log_prob(torch.tensor([factors["kind"]],
                                       device=self._device))[0] +
        control_dist.log_prob(torch.tensor([factors["control"]],
                                          device=self._device))[0] +
        card_dist.log_prob(torch.tensor([factors["card"]],
                                       device=self._device))[0] +
        action_dist.log_prob(torch.tensor([action],
                                         device=self._device))[0])
    entropy = (
        intent_dist.entropy()[0] + kind_dist.entropy()[0] +
        control_dist.entropy()[0] + card_dist.entropy()[0] +
        action_dist.entropy()[0])
    return logprob, entropy, outputs["value"][0]

  def _dist(self, logits, mask):
    masked_logits = torch.where(
        mask.view(1, -1), logits,
        torch.full_like(logits, _INVALID_LOGIT))
    return torch.distributions.Categorical(logits=masked_logits)

  def _mask(self, values, size):
    mask = torch.zeros(size, dtype=torch.bool, device=self._device)
    for value in values:
      if 0 <= int(value) < size:
        mask[int(value)] = True
    if not bool(mask.any()):
      mask[_NONE_TOKEN] = True
    return mask

  def _intent_mask(self, pack):
    return self._mask(
        [factor["intent"] for factor in pack["factors"].values()],
        self._kwargs["intent_vocab_size"])

  def _kind_mask(self, pack, intent):
    return self._mask(
        [
            factor["kind"] for factor in pack["factors"].values()
            if factor["intent"] == intent
        ],
        self._kwargs["kind_vocab_size"])

  def _control_mask(self, pack, intent, kind):
    return self._mask(
        [
            factor["control"] for factor in pack["factors"].values()
            if factor["intent"] == intent and factor["kind"] == kind
        ],
        self._kwargs["control_vocab_size"])

  def _card_mask(self, pack, intent, kind, control):
    return self._mask(
        [
            factor["card"] for factor in pack["factors"].values()
            if (factor["intent"] == intent and factor["kind"] == kind and
                factor["control"] == control)
        ],
        self._kwargs["card_vocab_size"])

  def _action_mask(self, pack, intent, kind, control, card):
    return self._mask(
        [
            action for action, factor in pack["factors"].items()
            if (factor["intent"] == intent and factor["kind"] == kind and
                factor["control"] == control and factor["card"] == card)
        ],
        self._num_actions)


class ForcetekiPPOOracle(rl_oracle.RLOracle):
  """PSRO oracle that trains factored Forceteki PPO responders."""

  def __call__(self, *args, **kwargs):
    new_policies = super().__call__(*args, **kwargs)
    for player_policies in new_policies:
      for pol in player_policies:
        if isinstance(pol, ForcetekiPPOPolicy):
          pol.finish_training()
    return new_policies

  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    del oracle_specific_execution_kwargs
    state = game.new_initial_state()
    live_agents = [
        agent for agent in agents
        if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen()
    ]

    try:
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes, probs = zip(*state.chance_outcomes())
          state.apply_action(utils.random_choice(outcomes, probs))
          continue

        player = state.current_player()
        agent = agents[player]
        if isinstance(agent, ForcetekiPPOPolicy) and not agent.is_frozen():
          action = agent.training_action(state)
        else:
          action_probs = agent(state, player)
          outcomes, probs = zip(*action_probs.items())
          action = utils.random_choice(outcomes, probs)

        state.apply_action(action)
        rewards = state.returns() if state.is_terminal() else state.rewards()
        if not rewards:
          rewards = [0.0] * state.num_players()
        for live_agent in live_agents:
          live_agent.add_pending_reward(rewards[live_agent.player_id])

      for live_agent in live_agents:
        live_agent.finish_episode()
    finally:
      _close_state(state)


def _sample_episode_with_diagnostics(state, policies):
  """Samples one episode and returns final returns plus rollout diagnostics."""
  while not state.is_terminal():
    if state.is_simultaneous_node():
      actions = [None] * state.num_players()
      for player in range(state.num_players()):
        state_policy = policies[player](state, player)
        outcomes, probs = zip(*state_policy.items())
        actions[player] = utils.random_choice(outcomes, probs)
      state.apply_actions(actions)
      continue

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
    else:
      player = state.current_player()
      state_policy = policies[player](state)
      outcomes, probs = zip(*state_policy.items())

    state.apply_action(utils.random_choice(outcomes, probs))

  returns = np.array(state.returns(), dtype=np.float32)
  reason = getattr(
      state, "forceteki_terminal_reason",
      lambda: "unknown_terminal")()
  move_number = getattr(
      state, "forceteki_move_number",
      lambda: state.move_number())()
  return returns, reason, int(move_number)


class DiagnosticPSROSolver(psro_v2.PSROSolver):
  """PSRO solver that can print ForceTeki rollout diagnostics per meta entry."""

  def __init__(self, *args, rollout_diagnostics=False, **kwargs):
    self._rollout_diagnostics = rollout_diagnostics
    super().__init__(*args, **kwargs)

  def update_empirical_gamestate(self, seed=None):
    if not self._rollout_diagnostics:
      return super().update_empirical_gamestate(seed=seed)

    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    if self.symmetric_game:
      self._policies = self._game_num_players * self._policies
      self._new_policies = self._game_num_players * self._new_policies
      self._num_players = self._game_num_players

    updated_policies = [
        self._policies[k] + self._new_policies[k]
        for k in range(self._num_players)
    ]
    total_number_policies = [
        len(updated_policies[k]) for k in range(self._num_players)
    ]
    number_older_policies = [
        len(self._policies[k]) for k in range(self._num_players)
    ]
    number_new_policies = [
        len(self._new_policies[k]) for k in range(self._num_players)
    ]

    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(self._num_players)
    ]

    older_policies_slice = tuple(
        [slice(len(self._policies[k])) for k in range(self._num_players)])
    for k in range(self._num_players):
      meta_games[k][older_policies_slice] = self._meta_games[k]

    for current_player in range(self._num_players):
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, self._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        used_tuple = tuple(used_index)
        if not np.isnan(meta_games[current_player][used_tuple]):
          continue

        estimated_policies = [
            updated_policies[k][current_index[k]]
            for k in range(current_player)
        ] + [
            self._new_policies[current_player][current_index[current_player]]
        ] + [
            updated_policies[k][current_index[k]]
            for k in range(current_player + 1, self._num_players)
        ]

        utility_estimates = self._sample_episodes_with_diagnostics(
            estimated_policies, self._sims_per_entry, used_tuple)

        if self.symmetric_game:
          player_permutations = list(itertools.permutations(
              list(range(self._num_players))))
          for permutation in player_permutations:
            permuted_tuple = tuple([used_index[i] for i in permutation])
            for player in range(self._num_players):
              if np.isnan(meta_games[player][permuted_tuple]):
                meta_games[player][permuted_tuple] = 0.0
              meta_games[player][permuted_tuple] += (
                  utility_estimates[permutation[player]] /
                  len(player_permutations))
        else:
          for k in range(self._num_players):
            meta_games[k][used_tuple] = utility_estimates[k]

    if self.symmetric_game:
      self._policies = [self._policies[0]]
      self._new_policies = [self._new_policies[0]]
      updated_policies = [updated_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    return meta_games

  def _sample_episodes_with_diagnostics(self, policies, num_episodes,
                                        profile_index):
    totals = np.zeros(self._num_players)
    reason_counts = {
        "forceteki_terminal": 0,
        "open_spiel_cap": 0,
        "non_terminal": 0,
        "unknown_terminal": 0,
    }
    move_numbers = []
    nonzero_returns = 0

    for _ in range(num_episodes):
      state = self._game.new_initial_state()
      try:
        returns, reason, move_number = _sample_episode_with_diagnostics(
            state, policies)
      finally:
        _close_state(state)
      totals += returns.reshape(-1)
      reason_counts[reason] = reason_counts.get(reason, 0) + 1
      move_numbers.append(move_number)
      if np.any(returns != 0):
        nonzero_returns += 1

    averages = totals / num_episodes
    self._print_rollout_diagnostics(
        profile_index, num_episodes, averages, reason_counts, nonzero_returns,
        move_numbers)
    return averages

  def _print_rollout_diagnostics(self, profile_index, num_episodes,
                                 averages, reason_counts, nonzero_returns,
                                 move_numbers):
    if move_numbers:
      avg_steps = float(np.mean(move_numbers))
      min_steps = min(move_numbers)
      max_steps = max(move_numbers)
      step_summary = f"{avg_steps:.1f}/{min_steps}/{max_steps}"
    else:
      step_summary = "nan/nan/nan"

    print(
        "Rollout diagnostics "
        f"profile={profile_index} sims={num_episodes} "
        f"avg_returns={averages.tolist()} "
        f"forceteki_terminal={reason_counts.get('forceteki_terminal', 0)} "
        f"open_spiel_cap={reason_counts.get('open_spiel_cap', 0)} "
        f"non_terminal={reason_counts.get('non_terminal', 0)} "
        f"unknown_terminal={reason_counts.get('unknown_terminal', 0)} "
        f"nonzero_returns={nonzero_returns} "
        f"steps(avg/min/max)={step_summary}")


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


def init_ppo_responder(env):
  """Initializes a factored PPO oracle and frozen initial policies."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = ForcetekiPPOPolicy
  agent_kwargs = {
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "steps_per_batch": FLAGS.ppo_steps_per_batch,
      "num_minibatches": FLAGS.ppo_num_minibatches,
      "update_epochs": FLAGS.ppo_update_epochs,
      "learning_rate": FLAGS.ppo_learning_rate,
      "gamma": FLAGS.ppo_gamma,
      "gae_lambda": FLAGS.ppo_gae_lambda,
      "clip_coef": FLAGS.ppo_clip_coef,
      "entropy_coef": FLAGS.ppo_entropy_coef,
      "value_coef": FLAGS.ppo_value_coef,
      "max_grad_norm": FLAGS.ppo_max_grad_norm,
      "target_kl": FLAGS.ppo_target_kl,
      "device": FLAGS.ppo_device,
      "intent_vocab_size": FLAGS.ppo_intent_vocab_size,
      "kind_vocab_size": FLAGS.ppo_kind_vocab_size,
      "control_vocab_size": FLAGS.ppo_control_vocab_size,
      "card_vocab_size": FLAGS.ppo_card_vocab_size,
  }
  oracle = ForcetekiPPOOracle(
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
    return init_ppo_responder(env)
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
  solver = DiagnosticPSROSolver(
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
      symmetric_game=FLAGS.symmetric_game,
      rollout_diagnostics=FLAGS.rollout_diagnostics)

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
    oracle, agents = init_oracle(env)
    run_psro(env, oracle, agents)
  finally:
    if env is not None:
      _close_state(getattr(env, "_state", None))
    forceteki.close_all_workers()


if __name__ == "__main__":
  app.run(main)
