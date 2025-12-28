import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import logging
import wandb
import math
from copy import deepcopy

log = logging.getLogger(__name__)
from util.timer import Timer
# Inherit from your WORKING PPO Agent to keep the structure
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent 

class ReplayBuffer:
    """ Simple Buffer for Off-Policy Data """
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

class TrainWPODiffusionAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        # Initialize the PPO parent to get envs, dirs, and model setup
        super().__init__(cfg)

        # --- WPO Specific Initialization ---
        
        # 1. Replay Buffer (WPO is Off-Policy)
        # Assuming flat observation for buffer, adjust if image-based
        buffer_obs_dim = self.obs_dim * self.n_cond_step
        
        # Determine action storage dimension
        # DPPO usually generates [Horizon, ActDim] but executes only [ActSteps, ActDim]
        # WPO needs to store what was executed (or the full plan if doing plan-based RL).
        # Standard: Store the executed action.
        buffer_act_dim = self.action_dim * self.act_steps 
        # Note: If your Critic expects the FULL HORIZON action, change this to:
        # buffer_act_dim = self.action_dim * self.horizon_steps
            
        self.replay_buffer = ReplayBuffer(
            buffer_obs_dim,
            buffer_act_dim,
            max_size=cfg.train.get("buffer_size", 100000),
            device=self.device
        )

        # 2. Target Critic (Critical for stable Q-Learning)
        # Ensure self.model.critic is a Q-Critic (takes obs+act)
        self.target_critic = deepcopy(self.model.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
            
        # 3. WPO Dual Optimizers (Alphas)
        self.log_alpha_mean = torch.tensor(np.log(cfg.train.get("init_alpha_mean", 1.0)), requires_grad=True, device=self.device)
        self.dual_optimizer = torch.optim.Adam([self.log_alpha_mean], lr=cfg.train.get("dual_lr", 1e-4))
        
        self.target_update_tau = cfg.train.get("tau", 0.005)
        self.epsilon_mean = cfg.train.get("epsilon_mean", 0.05) # Constraint limit

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        # Initial Reset
        prev_obs_venv = self.reset_env_all()
        
        while self.itr < self.n_train_itr:
            
            # --- 1. Evaluation Mode ---
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Video paths (Same as PPO)
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            
            # --- FIX: Initialize Holders (Variables that were undefined) ---
            # We restore the trajectory holders so the logging code at the end doesn't crash
            obs_trajs = {
                "state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))
            }
            # Note: WPO doesn't *need* chains for training, so we might save zeros to save RAM
            # unless we explicitly ask for them. Here we initialize them.
            chains_trajs = np.zeros((
                self.n_steps, self.n_envs, self.model.ft_denoising_steps + 1,
                self.horizon_steps, self.action_dim
            ))
            
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                # If we just reset, firsts is 1
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack((obs_full_trajs, prev_obs_venv["state"][:, -1][None]))
            else:
                obs_full_trajs = None # Define it as None so logging doesn't crash

            # --- 2. Data Collection Loop ---
            for step in range(self.n_steps):
                if step % 100 == 0: print(f"Processing step {step}/{self.n_steps}")

                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                    # Standard Diffusion Sampling
                    # We only need chains if we are debugging/visualizing, otherwise set return_chain=False for speed
                    # But to keep structure identical to PPO logging, we get them:
                    samples = self.model(
                        cond=cond, 
                        deterministic=eval_mode,
                        return_chain=True 
                    )
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()
                
                action_venv = output_venv[:, :self.act_steps]

                # Step Environment
                next_obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                
                # --- WPO CHANGE: Store in Buffer ---
                if not eval_mode:
                    # Flatten obs for buffer
                    flat_obs = prev_obs_venv["state"].reshape(self.n_envs, -1)
                    flat_next = next_obs_venv["state"].reshape(self.n_envs, -1)
                    # Flatten act
                    flat_act = action_venv.reshape(self.n_envs, -1)
                    
                    self.replay_buffer.add(
                        flat_obs, flat_act, reward_venv[:, None], flat_next, done_venv[:, None]
                    )

                # --- Fill Holders for Logging (Restoring PPO structure) ---
                if self.save_full_observations:
                    obs_full_venv = np.array([info["full_obs"]["state"] for info in info_venv])
                    obs_full_trajs = np.vstack((obs_full_trajs, obs_full_venv.transpose(1, 0, 2)))
                
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step+1] = done_venv
                
                prev_obs_venv = next_obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # --- FIX: Calculate Reward Metrics (Copied from PPO) ---
            # This block was missing in your WPO code, causing "undefined success_rate"
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if self.furniture_sparse_reward:
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # --- 3. Update Step (WPO Logic) ---
            # Initialize stats holders for logging
            log_actor_loss = 0.0
            log_q_loss = 0.0
            log_alpha = 0.0
            total_loss = 0.0
            
            if not eval_mode and self.replay_buffer.size > self.batch_size:
                
                # Number of gradient updates matches PPO ratio
                num_updates = int(self.n_steps * self.n_envs / self.batch_size) * self.update_epochs
                
                for i in range(num_updates):
                    batch = self.replay_buffer.sample(self.batch_size)
                    stats = self.update_wpo(batch)
                    
                    # Accumulate
                    log_actor_loss += stats['actor_loss']
                    log_q_loss += stats['q_loss']
                    log_alpha += stats['alpha']
                    total_loss += (stats['actor_loss'] + stats['q_loss']) # Approximation
                
                # Average
                log_actor_loss /= num_updates
                log_q_loss /= num_updates
                log_alpha /= num_updates
                total_loss /= num_updates

            # --- 4. Logging & Saving ---
            # Update schedulers
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                # self.dual_optimizer_scheduler.step() # If you have one
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save Model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
            
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                else:
                    # --- FIX: Log WPO metrics instead of PPO metrics ---
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | "
                        f"loss {total_loss:8.4f} | "
                        f"actor loss {log_actor_loss:8.4f} | "
                        f"q loss {log_q_loss:8.4f} | "
                        f"alpha {log_alpha:8.4f} | "
                        f"reward {avg_episode_reward:8.4f} | "
                        f"t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "train/loss": total_loss,
                                "train/actor_loss": log_actor_loss,
                                "train/q_loss": log_q_loss,
                                "train/alpha": log_alpha,
                                "train/avg_episode_reward": avg_episode_reward,
                                "diffusion/min_sampling_std": diffusion_min_sampling_std,
                                "train/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                                "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1

    def wpo_loss(
            self,
            policy_dist,
            target_policy_dist,
            actions_sampled,
            q_grad_wrt_actions
    ):
        # 1. Unpack
        mu = policy_dist.base_dist.loc
        sigma = policy_dist.base_dist.scale
        target_mu = target_policy_dist.base_dist.loc
        target_sigma = target_policy_dist.base_dist.scale

        # 2. Wasserstein Drift
        avg_q_grad = torch.mean(q_grad_wrt_actions, dim=1)
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
            "loss_dual": loss_dual.item(),
            "kl_mean": mean_kl_mean.mean().item(),
            "kl_std": mean_kl_std.mean().item(),
            "alpha_mean": alpha_mean.mean().item()
        }

    # Sanırım artık bütün derdimiz tasamız burada
    def update_wpo(self, batch):
        """ 
        Stable WPO Update for Diffusion.
        1. Critic uses Base Policy for targets (Stable Bellman).
        2. Actor uses Action-Space Gradient Ascent + Supervised Denoising.
        """
        obs, actions, rewards, next_obs, dones = batch
        
        # Enforce [B, 1]
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        # Reshape obs
        obs_cond = {"state": obs.reshape(self.batch_size, self.n_cond_step, self.obs_dim)}
        next_obs_cond = {"state": next_obs.reshape(self.batch_size, self.n_cond_step, self.obs_dim)}
        
        # ===========================
        # 1. Critic Update
        # ===========================
        with torch.no_grad():
            # --- FIX 1: Use Base Policy for Target (Stabilizes Q-Learning) ---
            # Using the frozen base policy prevents the "Blind leading the blind" explosion
            next_samples = self.model(cond=next_obs_cond, use_base_policy=True, deterministic=True)
            next_act = next_samples.trajectories[:, :self.act_steps]
            next_act_flat = next_act.reshape(self.batch_size, -1)
            
            target_q = self.target_critic(next_obs_cond, next_act_flat)
            if isinstance(target_q, tuple):
                target_q = torch.min(target_q[0], target_q[1])
            target_q = target_q.view(-1, 1)
            
            y = rewards + self.gamma * (1 - dones) * target_q

        # Current Q
        actions_flat = actions.reshape(self.batch_size, -1)
        current_q = self.model.critic(obs_cond, actions_flat)
        
        if isinstance(current_q, tuple):
            q1, q2 = current_q
            q_loss = F.mse_loss(q1.view(-1,1), y) + F.mse_loss(q2.view(-1,1), y)
        else:
            q_loss = F.mse_loss(current_q.view(-1,1), y)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        # Clip Critic Gradients
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # Soft Update Target Critic
        for param, target_param in zip(self.model.critic.parameters(), self.target_critic.parameters()):
            target_param.data.mul_(1 - self.target_update_tau)
            torch.add(target_param.data, param.data, alpha=self.target_update_tau, out=target_param.data)

        # ===========================
        # 2. Actor Update
        # ===========================
        
        # A. Sample Current Actions (No Grad needed here)
        with torch.no_grad():
            # We determine the direction based on where the CURRENT policy is
            samples = self.model(cond=obs_cond, deterministic=False)
            current_actions = samples.trajectories[:, :self.act_steps]
        
        # B. Calculate Wasserstein Drift (Q-Ascent direction)
        current_actions.requires_grad_(True)
        current_actions_flat = current_actions.reshape(self.batch_size, -1)
        
        q_val_actor = self.model.critic(obs_cond, current_actions_flat)
        if isinstance(q_val_actor, tuple): q_val_actor = q_val_actor[0]
        
        q_sum = q_val_actor.sum()
        q_grad = torch.autograd.grad(q_sum, current_actions, create_graph=False)[0]
        
        # --- Normalize Gradient ---
        # Q-values are ~1200, so grads are huge. Normalize to prevent explosion.
        grad_norm = q_grad.norm(dim=-1, keepdim=True) + 1e-8
        q_grad_normalized = q_grad / grad_norm
        
        # C. Construct Target Action
        # eta is the step size in Action Space. 
        # Since we normalized grad, 0.5 means "move 0.5 units in the optimal direction"
        eta = 0.5 
        target_actions = (current_actions + eta * q_grad_normalized).detach()
        
        # D. Train Diffusion Model (Supervised) 
        # We assume 'target_actions' is the ground truth x_0.
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.model.denoising_steps, (self.batch_size,), device=self.device).long()
        
        # 2. Create noise
        noise = torch.randn_like(target_actions)
        
        # 3. Add noise (Forward Process)
        alpha_bar = self.model.alphas_cumprod[t]
        while alpha_bar.dim() < target_actions.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)
            
        noisy_target = torch.sqrt(alpha_bar) * target_actions + torch.sqrt(1 - alpha_bar) * noise
        
        # 4. Predict Noise (Backward Process)
        # --- FIX: USE self.model.actor_ft (The Trainable Network) ---
        noise_pred = self.model.actor_ft(noisy_target, t, cond=obs_cond)
        
        # 5. Loss
        actor_loss = F.mse_loss(noise_pred, noise)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Logging Metric
        with torch.no_grad():
            dist = F.mse_loss(target_actions, current_actions).item()

        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item(), "alpha": dist}