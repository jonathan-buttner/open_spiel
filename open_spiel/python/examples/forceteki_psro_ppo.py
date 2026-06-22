# Copyright 2026 The OpenSpiel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Factored PPO policy for the Forceteki PSRO example."""

import numpy as np
import torch
from torch import nn
from torch import optim

from open_spiel.python import policy
from open_spiel.python.examples.forceteki_psro_utils import _INVALID_LOGIT
from open_spiel.python.examples.forceteki_psro_utils import _NONE_TOKEN
from open_spiel.python.examples.forceteki_psro_utils import _legal_action_map
from open_spiel.python.examples.forceteki_psro_utils import _prompt_payload
from open_spiel.python.examples.forceteki_psro_utils import _raw_action
from open_spiel.python.examples.forceteki_psro_utils import _stable_bucket
from open_spiel.python.examples.forceteki_psro_utils import _state_payload


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

  def checkpoint(self, player_id=None, policy_index=None):
    """Returns a reloadable Torch checkpoint payload for this policy."""
    return {
        "format": "forceteki_ppo_policy_v1",
        "player_id": self._player_id,
        "policy_player_id": self._player_id,
        "policy_index": policy_index,
        "population_player_id": player_id,
        "kwargs": dict(self._kwargs),
        "frozen": self._frozen,
        "network_state_dict": self.get_weights(),
        "optimizer_state_dict": self._optimizer.state_dict(),
    }

  def load_checkpoint(self, checkpoint, load_optimizer=True):
    """Loads network and optimizer state from a checkpoint payload."""
    self._network.load_state_dict(checkpoint["network_state_dict"])
    if load_optimizer and checkpoint.get("optimizer_state_dict") is not None:
      self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("frozen", True):
      self.freeze()
    else:
      self.unfreeze()

  @classmethod
  def from_checkpoint(cls, env, checkpoint, player_id=None, device=None,
                      load_optimizer=True):
    """Builds a policy from a checkpoint payload."""
    kwargs = dict(checkpoint["kwargs"])
    if device is not None:
      kwargs["device"] = device
    restored_player_id = (
        checkpoint.get("policy_player_id", checkpoint.get("player_id", 0))
        if player_id is None else player_id)
    policy_obj = cls(env, restored_player_id, **kwargs)
    policy_obj.load_checkpoint(checkpoint, load_optimizer=load_optimizer)
    return policy_obj

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
