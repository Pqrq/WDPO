import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
import wandb
import copy

# Import your bandit environment
from env.bandit import ContinuousBanditEnv


# =============================================================================
# 1. Lightweight Network Definitions (Mirrors your WPO_Gaussian)
# =============================================================================

class SimpleActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return Independent(Normal(mu, torch.exp(log_std)), 1)


class SimpleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        # Double Q architecture in one module for simplicity
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# =============================================================================
# 2. Toy WPO Agent (The Logic We Are Testing)
# =============================================================================

class ToyWPOAgent:
    def __init__(self, env, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Hyperparameters from WandB Config
        self.gamma = 0.0  # Bandit!
        self.tau = config.tau
        self.epsilon_mean = config.epsilon_mean
        self.epsilon_std = config.epsilon_mean * 0.1
        self.dual_lr = config.dual_lr

        # Networks
        self.actor = SimpleActor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = SimpleCritic(self.obs_dim, self.act_dim).to(self.device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Duals
        self.log_alpha_mean = nn.Parameter(torch.tensor([0.0]).to(self.device))
        self.log_alpha_std = nn.Parameter(torch.tensor([0.0]).to(self.device))

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.dual_opt = optim.Adam([self.log_alpha_mean, self.log_alpha_std], lr=self.dual_lr)

    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.actor(obs_t).sample().cpu().numpy()[0]

    def update(self, batch):
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        # Bandit: next_obs/done don't matter much with gamma=0, but we keep structure

        # --- 1. Critic Update (Double Q) ---
        with torch.no_grad():
            # Sample Target Actions
            N_SAMPLES = 8
            # Flatten batch for sampling
            obs_repeat = obs.repeat_interleave(N_SAMPLES, dim=0)

            target_dist = self.target_actor(obs_repeat)
            next_act = target_dist.sample()

            t_q1, t_q2 = self.target_critic(obs_repeat, next_act)
            t_q = torch.min(t_q1, t_q2)

            # Reshape back to [B, N] and average
            t_q = t_q.view(obs.shape[0], N_SAMPLES, 1).mean(dim=1)
            target = rew + self.gamma * t_q

        curr_q1, curr_q2 = self.critic(obs, act)
        q_loss = F.mse_loss(curr_q1, target) + F.mse_loss(curr_q2, target)

        self.critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # --- 2. Actor & Dual Update ---
        # Re-sample for actor update
        N_ACTOR = 8
        obs_repeat = obs.repeat_interleave(N_ACTOR, dim=0)

        policy_dist = self.actor(obs_repeat)
        target_dist = self.target_actor(obs_repeat)

        actions_sampled = policy_dist.sample()
        actions_sampled.requires_grad_(True)

        # Drift: grad Q
        q1_val, _ = self.critic(obs_repeat, actions_sampled)
        q_sum = q1_val.sum()
        grads = torch.autograd.grad(q_sum, actions_sampled)[0]

        # Unpack for WPO Loss
        mu = policy_dist.base_dist.loc
        sigma = policy_dist.base_dist.scale
        target_mu = target_dist.base_dist.loc
        target_sigma = target_dist.base_dist.scale

        # Drift Loss
        avg_grad = grads.view(obs.shape[0], N_ACTOR, -1).mean(dim=1)
        # Reshape mu to match [B, D] (it already is, but just in case)
        mu_flat = mu.view(obs.shape[0], N_ACTOR, -1)[:, 0, :]
        sigma_flat = sigma.view(obs.shape[0], N_ACTOR, -1)[:, 0, :]

        drift_term = (sigma_flat.pow(2) * avg_grad).detach()
        loss_drift = -torch.sum(mu_flat * drift_term, dim=-1).mean()

        # Constraint Loss (Simplified KL)
        # We just compute KL between the distributions directly
        kl = kl_divergence(target_dist, policy_dist).mean()

        # Duals
        alpha = F.softplus(self.log_alpha_mean) + 1e-8
        alpha = torch.clamp(alpha, max=100.0)

        loss_dual_update = alpha * (self.epsilon_mean - kl.detach())
        loss_constraint = alpha.detach() * kl

        total_loss = loss_drift + loss_constraint + loss_dual_update

        self.actor_opt.zero_grad()
        self.dual_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        self.dual_opt.step()

        # --- 3. Soft Update ---
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return q_loss.item(), loss_drift.item(), alpha.item()


# =============================================================================
# 3. The Sweep Loop
# =============================================================================

def train_sweep():
    run = wandb.init()
    config = run.config

    # 1. Setup
    env = ContinuousBanditEnv()
    agent = ToyWPOAgent(env, config)

    # Simple Buffer
    buffer_obs = []
    buffer_act = []
    buffer_rew = []

    rewards_history = []

    # 2. Loop
    # Fast bandit loop: 50 iterations, 50 steps each
    for itr in range(50):
        # Collect Data
        for _ in range(50):
            obs, _ = env.reset()
            act = agent.get_action(obs)
            _, rew, _, _, _ = env.step(act)

            # Simple Storage
            buffer_obs.append(obs)
            buffer_act.append(act)
            buffer_rew.append(rew)
            rewards_history.append(rew)

            # Keep buffer small (sliding window for bandit)
            if len(buffer_obs) > 1000:
                buffer_obs.pop(0)
                buffer_act.pop(0)
                buffer_rew.pop(0)

        # Update
        if not len(buffer_obs) >= 64:
            continue

        for _ in range(50):
            # Random Batch
            idx = np.random.randint(0, len(buffer_obs), 64)
            batch = {
                'obs': np.array(buffer_obs)[idx],
                'act': np.array(buffer_act)[idx],
                'rew': np.array(buffer_rew)[idx].reshape(-1, 1)  # No Scaling here, let sweep find it!
            }
            q_l, d_l, alph = agent.update(batch)

        # Log
        avg_r = np.mean(rewards_history[-50:])
        wandb.log({"reward": avg_r, "itr": itr, "alpha": alph})

    # 3. Calculate Stability
    final_rewards = rewards_history[-250:]  # Last 5 iterations
    mean_r = np.mean(final_rewards)
    std_r = np.std(final_rewards)

    # Goal: Maximize Mean, Minimize Std
    # If Std is high (crashing), score drops significantly
    stability_score = mean_r - (2.0 * std_r)

    wandb.log({
        "final_mean": mean_r,
        "final_std": std_r,
        "stability_score": stability_score
    })


# =============================================================================
# 4. Sweep Config & Launch
# =============================================================================

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'stability_score', 'goal': 'maximize'},
    'parameters': {
        'epsilon_mean': {'min': 0.001, 'max': 0.2},  # Relaxed constraint range
        'dual_lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
        'actor_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
        'tau': {'values': [0.001, 0.005, 0.01]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="wpo-bandit-stability")
    wandb.agent(sweep_id, train_sweep, count=50)