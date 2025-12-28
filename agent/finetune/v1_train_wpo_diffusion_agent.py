import numpy as np
import torch
import logging
import wandb
from copy import deepcopy
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from util.timer import Timer
from agent.finetune.train_wpo_agent import TrainWPOAgent, ReplayBuffer
from model.diffusion.sampling import make_timesteps, extract

log = logging.getLogger(__name__)

class TrainWPODiffusionAgent(TrainWPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # We need the actor_ft (fine-tuned) and potentially a separate target for WPO constraints
        # The WPO agent assumes self.model.target_actor exists. 
        # Since WPODiffusion keeps 'actor' and 'actor_ft', we map them:
        # self.model.actor -> The one being optimized
        # self.model.actor (frozen copy) -> used as the 'target' constraint usually
        self.q_grad_scale = cfg.train.get("q_grad_scale", 1.0)  # was 0.1
        self.max_inv_factor = cfg.train.get("max_inv_factor", 1e4)  # was 1e3
        self.grad_a_clip = cfg.train.get("grad_a_clip", 1e9)  # effectively no clip

        # Ensure target update parameters are set
        self.target_update_period = cfg.train.get("target_update_period", 1000)

        self._update_calls = 0
        self.target_update_period_updates = cfg.train.get("target_update_period_updates", 1000)


        # Create a separate copy for the target network and freeze if
        self.model.target_actor = deepcopy(self.model.actor).to(self.device)
        self.model.target_critic = deepcopy(self.model.critic).to(self.device)
        for p in self.model.target_actor.parameters():
            p.requires_grad = False
        for p in self.model.target_critic.parameters():
            p.requires_grad = False

        self.model.to(self.device)
        
        
    def run(self):
        """
        Combination of PPO Data Collection (Diffusion Sampling) 
        and WPO Replay Buffer storage.
        """
        timer = Timer()
        obs_venv = self.reset_env_all()
        
        # WPO is off-policy, so we collect data, store in buffer, then update
        while self.itr < self.n_train_itr:
            
            # --- 1. Evaluation / Mode Switch ---
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            if eval_mode:
                self.model.eval()
            else:
                self.model.train()

            # --- 2. Data Collection Loop ---
            episode_rewards = np.zeros(self.n_envs, dtype=float)
            
            # We collect n_steps of data (just like PPO), but add to buffer
            for step in range(self.n_steps):
                if step % 100 == 0:
                    log.info(f"Collecting step {step}/{self.n_steps}")

                # Select Action using Diffusion Model
                with torch.no_grad():
                    # Prepare condition
                    flat_obs = obs_venv['state'].reshape(self.n_envs, -1)
                    flat_obs_torch = torch.from_numpy(flat_obs).float().to(self.device)
                    cond = {"state": flat_obs_torch}

                    # Sample from Diffusion Actor
                    # Note: We use the 'trajectories' (final action) part of the Sample namedtuple
                    samples = self.model(cond, deterministic=eval_mode)
                    raw_action = samples.trajectories.cpu().numpy() # Shape: [10, 1, 3]
                    
                    # Take the first step of the horizon and remove the dimension
                    action_venv = raw_action[:, 0, :] # Shape: [10, 3]

                # Step Environment
                next_obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Prepare for Buffer
                flat_next_obs = next_obs_venv['state'].reshape(self.n_envs, -1)

                # Add to Replay Buffer
                # Note: We perform flattening similar to the WPO MLP agent
                action_buffer = raw_action.reshape(self.n_envs, -1)

                # Add to Replay Buffer
                self.replay_buffer.add(
                    flat_obs,
                    action_buffer,      # <--- CORRECT: This is shape (10, 12)
                    reward_venv[:, None],
                    flat_next_obs,
                    done_venv[:, None]
                )

                # Update current state
                obs_venv = next_obs_venv
                self.total_env_steps += self.n_envs
                reward_arr = np.asarray(reward_venv, dtype=float).reshape(self.n_envs,)
                episode_rewards += reward_arr

            # --- 3. Training Step (Update) ---
            update_stats = {}
            if self.total_env_steps >= self.warmup_steps and not eval_mode:
                # Perform K updates (matches n_steps or config ratio)
                n_updates = self.n_steps 
                
                for u in range(n_updates):
                    stats = self._update()
                    
                    # Soft Target Update (Optional: Diffusion usually doesn't update target actor often)
                    # For WPO, we usually update the Target Network slowly
                    # But here, 'target_actor' is actually the Constraint (Pretrained) model usually.
                    # If you have a separate target Q-network:
                    # Gemini deleted this
                    #self.model.soft_update_targets(tau=self.tau)

                    # Accumulate stats
                    if u == 0:
                        update_stats = stats
                    else:
                        for k, v in stats.items():
                            update_stats[k] += v
                
                    #self._update_calls += 1
                    #if (self._update_calls % self.target_update_period_updates) == 0:
                    #    # Use model.soft_update_targets if implemented; otherwise use helper below
                    #    if hasattr(self.model, "soft_update_targets"):
                    #        self.model.soft_update_targets(tau=self.tau)
                    #    else:
                    #        # simple helper
                    #        with torch.no_grad():
                    #            for p, tp in zip(self.model.actor.parameters(), self.model.target_actor.parameters()):
                    #                tp.data.mul_(1.0 - self.tau)
                    #                tp.data.add_(self.tau * p.data)
                    #            for p, tp in zip(self.model.critic.parameters(), self.model.target_critic.parameters()):
                    #                tp.data.mul_(1.0 - self.tau)
                    #                tp.data.add_(self.tau * p.data)

                #with torch.no_grad():
                #    for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                #        tp.data.mul_(1 - self.tau)
                #        tp.data.add_(self.tau * p.data)
                    
                #    for p, tp in zip(self.model.critic.parameters(), self.target_critic.parameters()):
                #        tp.data.mul_(1 - self.tau)
                #        tp.data.add_(self.tau * p.data)

                
                # Average stats
                # after update loop
                if update_stats:
                    for k in list(update_stats.keys()):
                        try:
                            update_stats[k] /= n_updates
                        except Exception:
                            # if an element is a tensor, convert safely
                            if isinstance(update_stats[k], torch.Tensor):
                                update_stats[k] = update_stats[k].item() / n_updates
                            else:
                                update_stats[k] = float(update_stats[k]) / n_updates
                else:
                    update_stats = {
                        "q_loss": 0.0,
                        "mean_abs_grad_eps": 0.0,
                        "wpo_loss": 0.0,
                        "critic_param_norm": 0.0,
                        "actor_grad_norm": 0.0,
                    }

                # logging helper
                def _safe_scalar(v):
                    try:
                        if isinstance(v, torch.Tensor):
                            return v.item()
                        return float(v)
                    except Exception:
                        return 0.0

                avg_reward = float(np.mean(episode_rewards))
                if self.itr % self.log_freq == 0:
                    time = timer()
                    log.info(
                        f"Itr {self.itr}: Steps {self.total_env_steps} | "
                        f"Reward {_safe_scalar(avg_reward):.4f} | "
                        f"Q {_safe_scalar(update_stats.get('q_loss', 0.0)):.3e} | "
                        f"Grad_a {_safe_scalar(update_stats.get('mean_abs_grad_a', 0.0)):.3e} | "
                        f"Actor_grad_norm {_safe_scalar(update_stats.get('actor_grad_norm', 0.0)):.3e} | "
                        f"Mean_abs_grad_eps {_safe_scalar(update_stats.get('mean_abs_grad_eps', 0.0)):.3e} | "
                        f"WPO Loss {_safe_scalar(update_stats.get('wpo_loss', 0.0)):.4f} | "
                        f"Time {time:.2f}"
                    )
                    if self.use_wandb:
                        wandb_logs = {"total_env_steps": self.total_env_steps,
                                    "train/episode_reward": float(avg_reward)}
                        for k, v in update_stats.items():
                            wandb_logs[f"train/{k}"] = _safe_scalar(v)
                        wandb.log(wandb_logs, step=self.itr)



            # --- 4. Logging & Saving ---
            avg_reward = np.mean(episode_rewards)
            
            if self.itr % self.log_freq == 0:
                time = timer()
                #log.info(
                #    f"Itr {self.itr}: Steps {self.total_env_steps} | "
                #    f"Reward {avg_reward:.4f} | "
                #    f"WPO Loss {update_stats.get('wpo_loss', 0):.4f} | "
                #    f"Time {time:.2f}"
                #)
                
                if self.use_wandb:
                    wandb_logs = {
                        "total_env_steps": self.total_env_steps,
                        "train/episode_reward": avg_reward,
                        **{f"train/{k}": v for k, v in update_stats.items()}
                    }
                    wandb.log(wandb_logs, step=self.itr)

            if self.itr % self.save_model_freq == 0:
                self.save_model()

            # Anneal diffusion parameters
            self.model.step()
            self.itr += 1

    def _update(self):
        #assert x_t.shape == x0.shape, f"x_t/x0 mismatch {x_t.shape} vs {x0.shape}"
        #assert eps_pred.shape == x_t.shape

        # 1. Sample Batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # ==========================
        # A. Critic Update (Bellman)
        # Toprak has been copy-pasted here!
        # ==========================
        N_TARGET_SAMPLES = 8
        # Prepare conditioning dict and ensure on device
        flat_obs = obs.reshape(self.batch_size, -1)
        cond = {"state": flat_obs.to(self.device)}

        # Sample N_TARGET_SAMPLES times from the target actor (diffusion wrapper).
        # Use deterministic=True if you want less stochasticity for target (common).
        N = N_TARGET_SAMPLES
        with torch.no_grad():
            samples_list = []
            for _ in range(N):
                s = self.model.sample_with_actor(self.model.target_actor, cond, deterministic=True, return_chain=False)
                samples_list.append(s.trajectories)

            # Stack -> [N, B, Ta, Da] -> transpose to [B, N, Ta, Da]
            stacked = torch.stack(samples_list, dim=0).transpose(0, 1).to(self.device)

        # Flatten horizon/time dims if your critic expects flattened action vector:
        # If horizon_steps * action_dim == action_dim used by critic (as in your buffer), flatten to [B*N, D]
        next_actions_sampled = stacked.reshape(self.batch_size, N, -1)  # [B, N, D_flat]
        flat_actions = next_actions_sampled.reshape(-1, next_actions_sampled.shape[-1])  # [B*N, D_flat]

        next_obs_expanded = (
            next_obs.unsqueeze(1)
            .expand(-1, N_TARGET_SAMPLES, -1)
            .reshape(-1, next_obs.shape[1])
        )

        # Target Critic expects (cond, action)
        target_q_values = self.model.get_q(next_obs_expanded, flat_actions, target=True)

        if isinstance(target_q_values, tuple):
            q1, q2 = target_q_values
            target_q_values = torch.min(q1, q2)

        target_q_mean = (
            target_q_values
            .view(self.batch_size, N_TARGET_SAMPLES)
            .mean(dim=1, keepdim=True)
        )

        y = rewards + self.gamma * (1 - dones) * target_q_mean

        # Update Critic
        # current_q shape [256], y shape [256, 1] -> Unsqueeze current_q
        current_q = self.model.get_q(obs, actions)
        if isinstance(current_q, tuple):
            current_q = torch.min(current_q[0], current_q[1])
        current_q = current_q.unsqueeze(1)

        q_loss = F.mse_loss(current_q, y)
        #log.debug(f"q_loss={q_loss.item():.6f}")

        self.critic_optimizer.zero_grad()
        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=10.0)

        self.critic_optimizer.step()

        def param_norm(module):
            s = 0.0
            for p in module.parameters():
                s += (p.data.norm().item())
            return s

        #log.debug(f"critic_param_norm={param_norm(self.model.critic):.6e} actor_param_norm={param_norm(self.model.actor):.6e}")


        # ==========================
        # B. Actor Update (noise-space WPO) - REPLACEMENT
        # ==========================
        # We'll use buffer actions (x0) to sample x_t via q_sample, predict eps, reconstruct action,
        # compute grad wrt eps, and pass grad_eps into wpo_loss.

        # --------------- prepare cond ---------------
        flat_obs = obs.reshape(self.batch_size, -1).to(self.device)
        cond_curr = {"state": flat_obs}
        cond_target = {"state": flat_obs}

        # --------------- get x0 from buffer (reshape) ---------------
        # 'actions' from replay buffer are flattened [B, Ta*Da] -> reshape to [B, Ta, Da]
        horizon = self.model.horizon_steps
        act_dim = self.model.action_dim
        x0 = actions.reshape(self.batch_size, horizon, act_dim).to(self.device)  # [B, Ta, Da]

        # --------------- choose timestep t (scalar or per-sample) ---------------
        # stable start: use the last finetuned denoising step (ft_denoising_steps - 1)
        max_t = max(2, int(self.model.ft_denoising_steps))
        t = torch.randint(low=1, high=max_t, size=(self.batch_size,), device=self.device)


        # --------------- forward q_sample to get x_t and true eps ---------------
        # Use model.q_sample so x_t is consistent with forward process of this diffusion model
        with torch.no_grad():
            x_t, eps_true = self.model.q_sample(x_start=x0, t=t)  # [B, Ta, Da], [B, Ta, Da]

        # --------------- predict eps (policy mean) ---------------
        # eps_pred: [B, Ta, Da]
        eps_pred = self.model.predict_eps(x_t, t, cond_curr)

        # --------------- build policy / target distributions over flattened eps ---------------
        # Flatten epsilon dims to shape [B, D_flat]
        eps_pred_flat = eps_pred.view(self.batch_size, -1)
        # fixed std for epsilon policy: use 1.0 (standard normal) or small stable min
        sigma_eps_flat = torch.ones_like(eps_pred_flat, device=self.device)

        #policy_dist = Independent(Normal(eps_pred_flat, sigma_eps_flat), 1)

        with torch.no_grad():
            eps_pred_tgt = self.model.target_actor(x_t, t, cond_target)
        #eps_pred_tgt_flat = eps_pred_tgt.view(self.batch_size, -1).detach()

        # ----------------- MULTI-SAMPLE EPS→ACTION GRADIENT (CORRECTED) -----------------
        # Choose number of eps samples per state to lower variance
        N_SAMPLES = 2  # reduce if OOM; try 2 or 4

        # eps_pred: [B, Ta, Da]
        # Expand mean eps to [B*N, Ta, Da] and sample noise (we compute gradients wrt eps_samples)
        eps_pred_det = eps_pred.detach()
        eps_pred_exp = eps_pred_det.unsqueeze(1).expand(-1, N_SAMPLES, -1, -1) \
                                        .reshape(-1, horizon, act_dim)  # [B*N, Ta, Da]
        noise = torch.randn_like(eps_pred_exp, device=self.device)
        eps_samples = (eps_pred_exp + noise).requires_grad_(True) # [B*N, Ta, Da]

        # Expand x_t and t to match B*N
        x_t_exp = x_t.unsqueeze(1).expand(-1, N_SAMPLES, -1, -1).reshape(-1, horizon, act_dim)  # [B*N, Ta, Da]
        t_repeat = t.unsqueeze(1).expand(-1, N_SAMPLES).reshape(-1).to(self.device)  # [B*N]

        # Reconstruct actions for each eps sample
        a_recon_multi = self.model.eps_to_x0(x_t_exp, eps_samples, t_repeat)  # [B*N, Ta, Da]
        a_flat_multi = a_recon_multi.view(-1, horizon * act_dim)              # [B*N, D_flat]

        # Expand obs for critic and evaluate Q for each sample
        obs_expanded = obs.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, obs.shape[1]).to(self.device)  # [B*N, obs_dim]
        q_vals_multi = self.model.get_q(obs_expanded, a_flat_multi)  # [B*N] or [B*N,1]
        if isinstance(q_vals_multi, tuple):
            q_vals_multi = torch.min(q_vals_multi[0], q_vals_multi[1])
        q_vals_multi = q_vals_multi.view(self.batch_size, N_SAMPLES)  # [B, N]

        # IMPORTANT: average across samples before summing across batch to stabilize scale
        q_mean_per_batch = q_vals_multi.mean(dim=1)  # [B]
        q_mean_sum = q_mean_per_batch.sum()          # scalar

        # compute gradient wrt eps_samples (produces [B*N, Ta, Da])
        grad_eps_multi = torch.autograd.grad(q_mean_sum, eps_samples, retain_graph=False)[0]

        # reshape back to [B, N, Ta, Da] and average sample-axis -> [B, Ta, Da]
        grad_eps_multi = grad_eps_multi.view(self.batch_size, N_SAMPLES, horizon, act_dim)
        avg_grad_eps = grad_eps_multi.mean(dim=1)  # [B, Ta, Da]

        # scale by q_grad_scale
        avg_grad_eps = avg_grad_eps * (self.q_grad_scale)

        # --- Convert averaged ∂Q/∂eps -> ∂Q/∂a (action-space) analytically ---
        sqrt_alpha_cum = extract(self.model.sqrt_alphas_cumprod, t, x_t.shape).to(self.device)
        sqrt_one_minus_alpha = extract(self.model.sqrt_one_minus_alphas_cumprod, t, x_t.shape).to(self.device)

        eps_den = sqrt_one_minus_alpha.clamp(min=1e-6)   # avoid tiny denominator
        inv_factor = -(sqrt_alpha_cum / eps_den)         # [B, Ta, Da]

        # Prevent enormous values — clip the inv_factor magnitude
        max_inv = getattr(self, "max_inv_factor", 1e3)    # tune between 1e2..1e4
        inv_factor = inv_factor.clamp(min=-max_inv, max=max_inv)

        grad_a = avg_grad_eps * inv_factor

        # Optional: clip action-space gradients to a safe range
        grad_clip = getattr(self, "grad_a_clip", 1.0)
        grad_a = grad_a.clamp(min=-grad_clip, max=grad_clip)

        # --- Build action-space policy & target dists using eps mean (policy mean) ---
        sigma_eps = 1.0
        scale_factor = (sqrt_one_minus_alpha / (sqrt_alpha_cum + 1e-12))
        action_mu = (x_t / (sqrt_alpha_cum + 1e-12)) - (scale_factor * eps_pred)   # [B, Ta, Da]
        action_sigma = (scale_factor * sigma_eps)
        B = action_mu.shape[0]

        action_mu_flat = action_mu.view(B, -1)
        action_sigma_flat = action_sigma.view(B, -1).clamp(min=1e-3)
        assert not torch.isnan(action_mu_flat).any(), "NaN in action_mu"
        assert not torch.isinf(action_mu_flat).any(), "Inf in action_mu"
        assert (action_sigma_flat > 0).all(), "Non-positive action sigma"

        # after computing grad_a:
        if torch.isnan(grad_a).any():
            log.warning("NaN in grad_a; inspect q/value scales.")

        policy_action_dist = Independent(Normal(action_mu_flat, action_sigma_flat), 1)

        with torch.no_grad():
            eps_pred_tgt = self.model.target_actor(x_t, t, cond_target)  # if not computed earlier
            target_action_mu = (x_t / (sqrt_alpha_cum + 1e-12)) - (scale_factor * eps_pred_tgt)
            target_action_mu_flat = target_action_mu.view(B, -1)
            target_action_sigma_flat = action_sigma_flat.detach()
            target_action_dist = Independent(Normal(target_action_mu_flat, target_action_sigma_flat), 1)

        # Flatten grad_a to [B, 1, D] for wpo_loss
        grad_a_flat = grad_a.view(B, -1).unsqueeze(1)          # [B, 1, D]
        action_mu_flat_unsq = action_mu_flat.unsqueeze(1)     # [B, 1, D]

        # Call WPO loss in action-space
        total_loss, stats = self.wpo_loss(
            policy_action_dist,
            target_action_dist,
            action_mu_flat_unsq,
            grad_a_flat.detach()
        )

        # diagnostics
        stats['mean_abs_grad_a'] = float(grad_a.view(self.batch_size, -1).abs().mean().item())
        stats['mean_abs_grad_eps'] = float(avg_grad_eps.view(self.batch_size, -1).abs().mean().item())
        # -------------------------------------------------------------------------


        # --------------- optimize actor and duals (same as before) ---------------
        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        total_loss.backward()
        some_grad_found = any((p.grad is not None and p.grad.abs().sum() > 0) for p in self.model.actor.parameters())
        assert some_grad_found, "No grad on actor! check wpo_loss or detachings"


        actor_grad_norm = sum(p.grad.norm().item() for p in self.model.actor.parameters() if p.grad is not None)
        #log.debug(f"actor_grad_norm={actor_grad_norm:.6e}")

        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_([self.model.log_alpha_mean, self.model.log_alpha_std], max_norm=10.0)

        #alpha_mean = self.model.get_alpha(self.model.log_alpha_mean).detach()
        #alpha_std  = self.model.get_alpha(self.model.log_alpha_std).detach()
        #log.info(f"alpha_mean_mean={alpha_mean.mean().item():.6e}, alpha_std_mean={alpha_std.mean().item():.6e}")

        # grad_eps summary
        gn = avg_grad_eps.view(self.batch_size, -1).abs().mean().item()
        #log.info(f"mean_abs_grad_eps={gn:.6e}")

        self.actor_optimizer.step()
        self.dual_optimizer.step()

        stats['q_loss'] = q_loss.item()
        stats['critic_param_norm'] = param_norm(self.model.critic)
        stats['actor_grad_norm'] = actor_grad_norm
        stats['mean_abs_grad_eps'] = gn
        return stats

    
    
