"""
WPO fine-tuning for Diffusion Policies.
Inherits from TrainPPODiffusionAgent but replaces the PPO update with WPO.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from torch.distributions import Normal, Independent

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from util.scheduler import CosineAnnealingWarmupRestarts

from wpo_loss import WPOLoss


class TrainWPODiffusionAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # --- WPO Specific Configurations ---
        self.wpo_cfg = cfg.train.get("wpo", {})

        # Instantiate WPO Loss
        self.wpo_loss_module = WPOLoss(
            action_dim=self.action_dim,
            epsilon_mean=self.wpo_cfg.get("epsilon_mean", 0.001),
            epsilon_std=self.wpo_cfg.get("epsilon_std", 0.00001),
            init_log_alpha_mean=self.wpo_cfg.get("init_log_alpha_mean", 5.0),
            init_log_alpha_std=self.wpo_cfg.get("init_log_alpha_std", 5.0),
            policy_loss_scale=self.wpo_cfg.get("policy_loss_scale", 1.0),
            kl_loss_scale=self.wpo_cfg.get("kl_loss_scale", 1.0),
            dual_loss_scale=self.wpo_cfg.get("dual_loss_scale", 1.0),
            per_dim_constraining=self.wpo_cfg.get("per_dim_constraining", True)
        ).to(self.device)

        # WPO Dual Optimizer (for the Lagrangian multipliers alpha)
        self.dual_lr = self.wpo_cfg.get("dual_lr", 1e-2)
        self.dual_optimizer = torch.optim.Adam(
            [self.wpo_loss_module.log_alpha_mean, self.wpo_loss_module.log_alpha_std],
            lr=self.dual_lr
        )

        log.info("Initialized TrainWPODiffusionAgent with WPO Loss.")

    def get_distribution_params(self, obs, chain_prev, step_indices):
        """
        Helper to extract Mu and Sigma from the diffusion model for WPO.

        Assumes self.model has a method or logic to return (loc, scale)
        given the inputs.

        NOTE: You might need to adjust this depending on how your
        DiffusionActor retrieves the mean/std.
        In DDPM, the 'action' is usually the noise prediction.
        """
        # We assume self.model.actor_ft or a similar method returns the predicted noise (mean)
        # and the scheduler returns the variance (std).

        # Example implementation assuming standard DDPM parameterization:
        # Mu = Predicted Noise (or Denoised Action)
        # Sigma = Fixed or Learned Variance scheduler

        # This is a placeholder call - adapt to your specific Model class
        if hasattr(self.model, "get_distribution_params"):
            return self.model.get_distribution_params(obs, chain_prev, step_indices)

        # Fallback/Manual extraction if get_distribution_params doesn't exist
        # This assumes self.model.p_mean_variance or similar exists, typical in diffusers
        # For simplicity here, we assume the model output IS the mean
        model_output = self.model.actor_ft(obs, chain_prev, step_indices)

        # Get variance from scheduler (usually fixed in DPPO)
        # extracting variance for the specific time steps
        logvar = self.model.extract(self.model.posterior_log_variance_clipped, step_indices, chain_prev.shape)
        std = torch.exp(0.5 * logvar)

        return model_output, std

    def compute_q_grads(self, obs, chain_prev, action_sample, step_indices):
        """
        Computes grad_a Q(s, a).

        CRITICAL: WPO requires a Q-Critic Q(obs, action).
        Standard DPPO uses V(obs).

        If you only have V(obs), this gradient will be zero and WPO will not work.
        """
        # Enable gradient calculation w.r.t the action sample
        action_sample.requires_grad_(True)

        # Calculate Q-value
        # If your critic is V(s), this line needs to be: self.model.critic(obs)
        # But then grad will be None/Zero.
        # Assuming you have updated self.model.critic to take (obs, action, step)
        try:
            # Try calling with actions and steps
            q_values = self.model.critic(obs, action_sample, step_indices)
        except TypeError:
            # Fallback to standard DPPO critic (V-function) and warn
            # Note: This effectively breaks the 'Drift' part of WPO unless you use
            # an advantage approximation here.
            q_values = self.model.critic(obs)
            # Warning: This gradient will likely be zero w.r.t action_sample

        # Sum q_values to get a scalar for backward
        q_sum = q_values.sum()

        # Compute gradient w.r.t action_sample
        q_grads = torch.autograd.grad(q_sum, action_sample, create_graph=True)[0]

        return q_grads

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))

        while self.itr < self.n_train_itr:
            # --- Standard DPPO Data Collection (Copied from Parent) ---

            # Prepare video paths
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            # Storage
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Collect Trajectories
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()

                action_venv = output_venv[:, : self.act_steps]

                # Step Env
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)

                done_venv = terminated_venv | truncated_venv

                if self.save_full_observations:
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )

                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize Rewards (Standard Logic)
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
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                num_episode_finished = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # --- WPO UPDATE LOGIC STARTS HERE ---
            if not eval_mode:
                # Prepare Tensors
                obs_k = {
                    "state": einops.rearrange(
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device),
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )

                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps

                # Metrics trackers
                wpo_stats_acc = {}

                for update_epoch in range(self.update_epochs):
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)

                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]

                        # Unravel indices to get batch and time-step specific data
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )

                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b] # x_t (Input)
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1] # x_{t-1} (Target/Output)

                        # 1. Get Target Distribution (Fixed Reference)
                        # We use torch.no_grad() and the current model weights (assuming trust region is valid over small updates)
                        # Ideally, this should be a frozen copy of the model from the start of the iteration.
                        with torch.no_grad():
                            target_mu, target_sigma = self.get_distribution_params(
                                obs_b, chains_prev_b, denoising_inds_b
                            )
                            target_dist = Independent(
                                Normal(loc=target_mu, scale=target_sigma),
                                reinterpreted_batch_ndims=1
                            )

                        # 2. Get Current Distribution (With Gradients)
                        # Re-run forward pass to track gradients on parameters
                        mu, sigma = self.get_distribution_params(
                            obs_b, chains_prev_b, denoising_inds_b
                        )
                        policy_dist = Independent(
                            Normal(loc=mu, scale=sigma),
                            reinterpreted_batch_ndims=1
                        )

                        # 3. Compute Q-Gradient (Wasserstein Drift)
                        # We need gradients of Q(s, a) w.r.t a.
                        # In diffusion, 'a' is the sample generated by the distribution (chains_next_b or new sample).
                        # WPO typically samples from the current policy.

                        # Sample new actions for gradient estimation (reparameterization trick)
                        actions_sampled = policy_dist.rsample() # [B, D]

                        # We need [B, N, D] format for WPO loss, here N=1
                        actions_sampled_expanded = actions_sampled.unsqueeze(1)

                        # Compute Grad Q w.r.t actions
                        q_grads = self.compute_q_grads(
                            obs_b, chains_prev_b, actions_sampled, denoising_inds_b
                        )
                        q_grads_expanded = q_grads.unsqueeze(1) # [B, 1, D]

                        # 4. WPO Loss Calculation
                        loss, stats = self.wpo_loss_module(
                            policy_dist=policy_dist,
                            target_policy_dist=target_dist,
                            actions_sampled=actions_sampled_expanded,
                            q_grad_wrt_actions=q_grads_expanded
                        )

                        # 5. Optimization Step
                        self.actor_optimizer.zero_grad()
                        self.dual_optimizer.zero_grad()

                        loss.backward()

                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(), self.max_grad_norm
                            )

                        self.actor_optimizer.step()
                        self.dual_optimizer.step()

                        # Accumulate stats
                        for k, v in stats.items():
                            wpo_stats_acc[k] = wpo_stats_acc.get(k, 0) + v

                        # Log occasionally
                        if batch % 10 == 0:
                            log.info(f"Epoch {update_epoch} Batch {batch}: WPO Loss {loss.item():.4f}")

            # --- End WPO Update ---

            # Plotting (Standard)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Schedulers
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save Model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Logging
            run_results.append({"itr": self.itr, "step": cnt_train_step})

            # WPO Logging Average
            avg_stats = {k: v / (self.update_epochs * num_batch) for k, v in wpo_stats_acc.items()} if not eval_mode else {}

            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(f"eval: success rate {success_rate:.4f} | avg reward {avg_episode_reward:.4f}")
                    if self.use_wandb:
                        wandb.log({
                            "success rate - eval": success_rate,
                            "avg episode reward - eval": avg_episode_reward,
                        }, step=self.itr, commit=False)
                else:
                    log.info(f"{self.itr}: WPO Loss {avg_stats.get('wpo_loss', 0):.4f} | reward {avg_episode_reward:.4f}")
                    if self.use_wandb:
                        wandb_log_dict = {
                            "total env step": cnt_train_step,
                            "train_episode_reward": avg_episode_reward,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                        }
                        # Add WPO stats to wandb
                        wandb_log_dict.update(avg_stats)
                        wandb.log(wandb_log_dict, step=self.itr, commit=True)

            with open(self.result_path, "wb") as f:
                pickle.dump(run_results, f)

            self.itr += 1