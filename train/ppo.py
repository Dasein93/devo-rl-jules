import collections
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def set_seed(seed, torch_deterministic=True):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def flatten_obs(obs_dict):
    """
    Flattens the observation dictionary from PettingZoo into a stacked tensor.

    Args:
        obs_dict (dict): A dictionary where keys are agent IDs and values are their observations.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A stacked numpy array of observations, padded to the max length.
            - list: A sorted list of agent IDs corresponding to the observations.
    """
    agent_ids = sorted(obs_dict.keys())
    if not agent_ids:
        # Return empty values if no agents are active
        return np.array([]), []

    # Get all observations and find the maximum length
    obs_list = [np.asarray(obs_dict[agent_id]) for agent_id in agent_ids]
    max_len = max(obs.shape[0] for obs in obs_list)

    # Pad observations and stack them
    padded_obs = [np.pad(obs, (0, max_len - obs.shape[0]), "constant") for obs in obs_list]
    return np.vstack(padded_obs), agent_ids


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, num_actions), std=0.01),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize a layer with orthogonal weights and constant bias."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        """Get the value of an observation."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Get an action and the value of an observation.
        If an action is provided, it will be used instead of sampling.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value
