import os
import pickle
import numpy as np
import torch
import logging
import wandb
import copy
import time
from torch import optim
import torch.nn.functional as F

from torch.distributions import Normal, Independent, kl_divergence

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=100000, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        # Handle vector env inputs
        n_samples = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + n_samples) % self.max_size

        self.obs[indices] = obs
        self.next_obs[indices] = next_obs
        self.actions[indices] = action
        self.rewards[indices] = reward
        self.dones[indices] = done

        self.ptr = (self.ptr + n_samples) % self.max_size
        self.size = min(self.size + n_samples, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )


class TrainWPOAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # WPO Hyperparams from Config
        self.gamma = cfg.train.gamma
        self.target_update_period = cfg.train.target_update_period
        self.warmup_steps = cfg.train.warmup_steps
        self.tau = cfg.train.tau

        self.epsilon_mean = cfg.train.epsilon_mean
        self.epsilon_std = cfg.train.epsilon_std
        self.policy_loss_scale = cfg.train.policy_loss_scale
        self.kl_loss_scale = cfg.train.kl_loss_scale
        self.dual_loss_scale = cfg.train.dual_loss_scale
        self.per_dim_constraining = cfg.train.per_dim_constraining

        # Pass specific args to the model wrapper manually since they aren't in the default instantiation
        self.model.log_alpha_mean.data.fill_(cfg.train.init_log_alpha_mean)
        self.model.log_alpha_std.data.fill_(cfg.train.init_log_alpha_std)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(), lr=cfg.train.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(), lr=cfg.train.critic_lr
        )
        self.dual_optimizer = optim.Adam(
            [self.model.log_alpha_mean, self.model.log_alpha_std], lr=cfg.train.dual_lr
        )

        # Buffer
        self.replay_buffer = ReplayBuffer(
            self.obs_dim * self.n_cond_step,
            self.action_dim,
            cfg.train.buffer_size,
            self.device
        )

        # Total env steps counter
        self.total_env_steps = 0

        # temp for logging
        self.last_log_time = None

    def run(self):
        timer = Timer()

        # Initialize envs
        obs_venv = self.reset_env_all()

        while self.itr < self.n_train_itr:

            # --- 1. Evaluation Mode ---
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            if eval_mode:
                self.model.eval()
                # Run evaluation logic (simplified for brevity, usually involves separate rollout loop)
                pass
            else:
                self.model.train()

            # --- 2. Rollout Step (Collect Data) ---
            episode_rewards = np.zeros(self.n_envs).tolist()
            for step in range(self.n_steps):
                if step % 100 == 0:
                    log.info(f"Collecting step {step}/{self.n_steps}")

                # Select Action
                with torch.no_grad():
                    # Obs shape: [n_envs, n_cond_step, obs_dim] -> Flatten for MLP
                    # The MLP expects [n_envs, n_cond_step * obs_dim]
                    flat_obs = obs_venv['state'].reshape(self.n_envs, -1)
                    flat_obs_torch = torch.FloatTensor(flat_obs).to(self.device)
                    samples = self.model(flat_obs_torch, deterministic=False)
                    action_venv = samples.cpu().numpy()

                # Step Env
                next_obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Store in Buffer
                # We flatten observation for the buffer to match network input
                flat_next_obs = next_obs_venv['state'].reshape(self.n_envs, -1)

                # Expand rewards/dones for buffer dimensions [n_envs, 1]
                self.replay_buffer.add(
                    flat_obs,
                    action_venv,
                    reward_venv[:, None],
                    flat_next_obs,
                    done_venv[:, None]
                )

                # Update State
                obs_venv = next_obs_venv
                self.total_env_steps += self.n_envs

                # reward logging (keep original for logging)
                episode_rewards = episode_rewards + reward_venv

            # --- 3. Training Step (Update) ---
            if self.total_env_steps >= self.warmup_steps:
                update_stats = {}
                # Perform K updates per iteration (usually matches n_steps or defined ratio)
                # Here we do 1 update per env step collected, or batch it.
                # Let's do a fixed number of updates per iteration.
                n_updates = self.n_steps  # Simple 1-to-1 ratio

                for u in range(n_updates):
                    stats = self._update()

                    # Hard Target Update
                    # if (self.total_env_steps - self.n_steps + u) % self.target_update_period == 0:
                    #     self.model.hard_update_targets()

                    # Soft Target Update
                    self.model.soft_update_targets(tau=self.tau)

                    # Aggregate stats
                    if u == 0:
                        update_stats = stats
                    else:
                        for k, v in stats.items():
                            update_stats[k] += v

                # Average stats
                for k in update_stats.keys():
                    update_stats[k] /= n_updates

            else:
                update_stats = {}

            # --- 4. Logging & Saving ---
            avg_reward = np.mean(episode_rewards)

            if self.itr % self.log_freq == 0:
                time = timer()
                log.info(
                    f"Itr {self.itr}: Steps {self.total_env_steps} | "
                    f"Reward {avg_reward:.4f} | "
                    f"Total Loss {update_stats.get('wpo_loss', 0):.4f} | \n"
                    f"Drift Loss {update_stats.get('loss_drift', 0):.4f} | "
                    f"KL Loss {update_stats.get('loss_kl', 0):.4f} | "
                    f"Dual Loss {update_stats.get('loss_dual', 0):.4f} | "
                    f"Time {time:.2f}"
                )

                if self.use_wandb:
                    wandb_logs = {
                        "total_env_steps": self.total_env_steps,
                        "train/episode_reward": avg_reward,
                        **{f"train/{k}": v for k, v in update_stats.items()}
                    }
                    wandb.log(wandb_logs, step=self.itr)

            if self.itr % self.save_model_freq == 0:
                self.save_model()

            self.itr += 1

    def _update(self):
        # Sample Batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        # ==========================
        # A. Critic Update
        # ==========================
        with torch.no_grad(): # Target Q-Value Calculation
            N_TARGET_SAMPLES = 32
            # Get distribution from target actor
            flat_obs = obs.reshape(self.batch_size, -1)
            next_dist = self.model.target_actor(flat_obs)

            # [B, N, D]
            next_actions_sampled = next_dist.sample((N_TARGET_SAMPLES,)).transpose(0, 1)

            # Average Target Q-Values
            # Expand next_obs: [B, D] -> [B, N, D] -> [B*N, D]
            next_obs_expanded = next_obs.unsqueeze(1).expand(-1, N_TARGET_SAMPLES, -1).reshape(-1, obs.shape[1])
            flat_actions = next_actions_sampled.reshape(-1, self.action_dim)

            # Target Critic expects (cond, action)
            target_q1, target_q2 = self.model.get_q(next_obs_expanded, flat_actions, target=True)

            # Reshape back [B, N] and mean
            target_q1 = target_q1.reshape(self.batch_size, N_TARGET_SAMPLES).mean(dim=1, keepdim=True)
            target_q2 = target_q2.reshape(self.batch_size, N_TARGET_SAMPLES).mean(dim=1, keepdim=True)
            target_q_min = torch.min(target_q1, target_q2)

            y = rewards + self.gamma * (1 - dones) * target_q_min

        # Update Critic
        # q1 shape [256], y shape [256, 1] -> Unsqueeze q
        q1, q2 = self.model.get_q(obs, actions)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)

        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # ==========================
        # B. Actor Update
        # ==========================

        # 1. Distributions
        policy_dist = self.model.actor(obs)  # Current
        target_dist = self.model.target_actor(obs)  # Constraint Target

        # 2. Sample N actions from Current Actor
        N_ACTOR_SAMPLES = 128
        actions_sampled = policy_dist.sample((N_ACTOR_SAMPLES,)).transpose(0, 1)
        actions_sampled.requires_grad_(True)

        # 3. Calculate Q-Gradient (Drift)
        obs_expanded = obs.unsqueeze(1).expand(-1, N_ACTOR_SAMPLES, -1).reshape(-1, obs.shape[1])
        flat_sampled_actions = actions_sampled.reshape(-1, self.action_dim)

        # Use UPDATED critic
        q1, q2 = self.model.get_q(obs_expanded, flat_sampled_actions)
        Q = q1.sum() # Use Q1's gradients for flow calculation, whichever Q is not important
        # normalize Q with respect to the number of samples to keep scale consistent
        Q = Q / (N_ACTOR_SAMPLES/2)

        Q_grad_a = torch.autograd.grad(Q, actions_sampled, create_graph=False)[0]

        # 4. WPO Loss
        total_loss, stats = self.wpo_loss(
            policy_dist, target_dist, Q_grad_a.detach()
        )

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_([self.model.log_alpha_mean, self.model.log_alpha_std], max_norm=10.0)

        self.actor_optimizer.step()
        self.dual_optimizer.step()

        stats['q_loss'] = q_loss.item()
        return stats

    def wpo_loss(
            self,
            policy_dist,
            target_policy_dist,
            q_grad_wrt_actions
    ):
        # 1. Unpack
        mu = policy_dist.base_dist.loc
        sigma = policy_dist.base_dist.scale
        target_mu = target_policy_dist.base_dist.loc
        target_sigma = target_policy_dist.base_dist.scale

        # 2. Wasserstein Drift
        avg_q_grad = torch.mean(q_grad_wrt_actions, dim=1)
        # warning: q grad might be exploding, consider clipping
        # avg_q_grad = torch.clamp(avg_q_grad, -50.0, 50.0)

        # log the avg_q_grad every 10 seconds
        t = time.time()
        if self.last_log_time is None or (t - self.last_log_time) > 10.0:
            self.last_log_time = t
            avg_grad_norm = torch.mean(torch.norm(avg_q_grad, dim=-1)).item()
            log.info(f"Avg Q-Gradient Norm: {avg_grad_norm:.4f}")
            if self.use_wandb:
                wandb.log({"train/avg_q_grad_norm": avg_grad_norm}, step=self.itr)

        drift_target_mu = (sigma.pow(2) * avg_q_grad).detach()
        loss_policy_drift = -torch.sum(mu * drift_target_mu, dim=-1).mean()

        # 3. KL Constraints
        dist_fixed_std = Normal(loc=mu, scale=target_sigma.detach())
        dist_fixed_mean = Normal(loc=target_mu.detach(), scale=sigma)
        target_dist_base = Normal(loc=target_mu.detach(), scale=target_sigma.detach())

        kl_mean = kl_divergence(target_dist_base, dist_fixed_std)
        kl_std = kl_divergence(target_dist_base, dist_fixed_mean)

        if not self.per_dim_constraining:
            kl_mean = kl_mean.sum(dim=-1, keepdim=True)
            kl_std = kl_std.sum(dim=-1, keepdim=True)

        mean_kl_mean = kl_mean.mean(dim=0)
        mean_kl_std = kl_std.mean(dim=0)

        # 4. Dual Losses
        alpha_mean = self.model.get_alpha(self.model.log_alpha_mean)
        alpha_std = self.model.get_alpha(self.model.log_alpha_std)

        alpha_mean = torch.clamp(alpha_mean, max=100.0) # Clamp to prevent from dominating drift too much in the loss calculation
        alpha_std = torch.clamp(alpha_std, max=100.0)

        loss_kl_penalty = torch.sum(alpha_mean.detach() * mean_kl_mean) + \
                          torch.sum(alpha_std.detach() * mean_kl_std)

        loss_dual = torch.sum(alpha_mean * (self.epsilon_mean - mean_kl_mean.detach())) + \
                    torch.sum(alpha_std * (self.epsilon_std - mean_kl_std.detach()))

        total_loss = (
                self.policy_loss_scale * loss_policy_drift +
                self.kl_loss_scale * loss_kl_penalty +
                self.dual_loss_scale * loss_dual
        )

        return total_loss, {
            "wpo_loss": total_loss.item(),
            "loss_drift": loss_policy_drift.item(),
            "loss_kl": loss_kl_penalty.item(),
            "loss_dual": loss_dual.item(),
            "kl_mean": mean_kl_mean.mean().item(),
            "kl_std": mean_kl_std.mean().item(),
            "alpha_mean": alpha_mean.mean().item(),
            "alpha_std": alpha_std.mean().item()
        }