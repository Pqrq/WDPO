import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

import copy
import wandb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from agent.finetune.wpo_loss import WPOLoss
from env.bandit import ContinuousBanditEnv

# --- Neural Network Architectures ---

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        feat = self.net(x)
        mu = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        # Clamping log_std for stability is good practice
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        # Return Independent Normal to match WPOLoss expectations
        return Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        # Input: Concatenation of State and Action
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# --- WPO Agent ---

class WPOAgent(nn.Module):
    def __init__(
            self,
            env,
            hidden_dim=64,
            lr_policy=3e-4,
            lr_critic=1e-3,
            lr_dual=1e-2,  # Dual variables often need faster adjustment
            device="cpu"
    ):
        super().__init__()
        self.env = env
        self.device = device

        # Dimensions
        # Handle cases where observation is scalar or array
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # 1. Networks
        self.policy = PolicyNet(self.obs_dim, self.act_dim, hidden_dim).to(device)
        # Target policy for KL constraints (Old Policy)
        self.target_policy = copy.deepcopy(self.policy).to(device)
        self.target_policy.eval()  # Target is frozen during update

        # Critic (Q-Function) - Essential for WPO Gradient
        self.q_net = QNet(self.obs_dim, self.act_dim, hidden_dim).to(device)

        # 2. WPO Loss Module
        self.wpo_loss_module = WPOLoss(action_dim=self.act_dim).to(device)

        # 3. Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr_critic)

        # Optimizer for the dual variables (alphas) inside WPOLoss
        self.alpha_opt = optim.Adam(self.wpo_loss_module.parameters(), lr=lr_dual)

    def learn(self, total_timesteps):
        """
        Main training loop for Bandits/Simple Env
        """
        obs, _ = self.env.reset()

        for step in range(total_timesteps):
            # --- 1. Data Collection ---
            # Convert obs to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get Action distribution and sample
            with torch.no_grad():
                dist = self.policy(obs_tensor)
                action = dist.sample()  # [1, act_dim]
                action_np = action.cpu().numpy()[0]

            # Step Environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            # --- 2. Update Step ---
            stats = self._update(obs_tensor, action, reward)

            # --- 3. Logging ---
            log_dict = {
                "WPO/reward": reward,
                "WPO/step": step,
                **{f"WPO_debug/{k}": v for k, v in stats.items()}
            }
            wandb.log(log_dict)

            # Reset if done (though usually not needed for pure Bandits)
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

    def _update(self, obs, action, reward):
        """
        Performs Q-learning update and WPO Policy update.
        """
        # Ensure inputs are tensors on device
        reward = torch.FloatTensor([reward]).unsqueeze(1).to(self.device)  # [1, 1]

        # ==========================
        # A. Critic Update (MSE)
        # ==========================
        # For bandits: Target is just Reward (Gamma=0)
        # For sequential: Target = r + gamma * V(next_s)
        q_pred = self.q_net(obs, action)
        q_loss = F.mse_loss(q_pred, reward)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ==========================
        # B. WPO Policy Update
        # ==========================

        # 1. Update Target Policy (Old Policy)
        # In this simple loop, we set target = policy BEFORE the update
        # effectively constraining the update to stay close to where we just were.
        self.target_policy.load_state_dict(self.policy.state_dict())

        # 2. Re-evaluate distribution (to track gradients)
        policy_dist = self.policy(obs)
        target_policy_dist = self.target_policy(obs)

        # 3. Calculate Q-Gradient w.r.t Action (The "Drift" direction)
        # We need to sample actions from the CURRENT policy to differentiate through it?
        # WPO formulation usually takes samples and moves them.
        # Here we re-sample 'N' actions for stable gradient estimation.
        N_SAMPLES = 16
        # [B, N, D] -> [1, 16, D]
        actions_sampled = policy_dist.sample((N_SAMPLES,)).transpose(0, 1)

        # We need grad of Q(s, a) w.r.t a
        actions_sampled.requires_grad_(True)
        # Expand obs to match samples: [1, D] -> [1, 1, D] -> [1, N, D]
        obs_expanded = obs.unsqueeze(1).expand(-1, N_SAMPLES, -1)

        # Flatten for Q-net: [B*N, ...]
        q_vals = self.q_net(
            obs_expanded.reshape(-1, self.obs_dim),
            actions_sampled.reshape(-1, self.act_dim)
        )
        q_vals = q_vals.sum()  # Scalar for backward

        # Compute Gradients
        grads = torch.autograd.grad(q_vals, actions_sampled, create_graph=False)[0]
        # q_grad_wrt_actions: [B, N, D]

        # 4. Compute WPO Loss
        loss, wpo_stats = self.wpo_loss_module(
            policy_dist=policy_dist,
            target_policy_dist=target_policy_dist,
            actions_sampled=actions_sampled.detach(),  # Detach actions, we only need the grad direction
            q_grad_wrt_actions=grads.detach()  # Treat Q-grad as constant vector field
        )

        # 5. Optimize Policy and Dual Variables
        self.policy_opt.zero_grad()
        self.alpha_opt.zero_grad()

        loss.backward()

        self.policy_opt.step()
        self.alpha_opt.step()

        wpo_stats['q_loss'] = q_loss.item()
        return wpo_stats