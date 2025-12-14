"""
WDPO fine-tuning agent.
Inherits from DPPO Agent but replaces PPO optimization with WPO.
"""

import copy
import logging
import math
import numpy as np
import torch
import wandb
import einops

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
# Assuming you saved the previous file as wpo_loss.py in agent/finetune/
from agent.finetune.wpo_loss import WPOLoss

log = logging.getLogger(__name__)

class TrainWDPOAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        # 1. Initialize Parent (DPPO)
        # This sets up envs, model (actor_ft, critic), wandb, etc.
        super().__init__(cfg)

        log.info("Initializing WDPO Agent extensions...")

        # 2. Setup WPO Specifics
        # WPO requires a Target Policy Network to anchor the KL constraint.
        # We assume self.model.actor_ft is the policy network being fine-tuned.
        self.target_actor = copy.deepcopy(self.model.actor_ft)
        self.target_actor.to(self.device)
        self.target_actor.eval() # Target network is always in eval mode

        # Soft update parameter (tau)
        self.tau = cfg.train.get("target_update_tau", 0.005)

        # WPO Sampling settings
        self.wpo_num_samples = cfg.train.get("wpo_num_samples", 20)

        # 3. Initialize WPO Loss Module
        self.wpo_loss = WPOLoss(
            action_dim=self.action_dim,
            epsilon_mean=cfg.train.get("epsilon_mean", 0.001),
            epsilon_std=cfg.train.get("epsilon_std", 1e-5),
            init_log_alpha_mean=cfg.train.get("init_log_alpha_mean", 5.0),
            init_log_alpha_std=cfg.train.get("init_log_alpha_std", 5.0),
            policy_loss_scale=cfg.train.get("policy_loss_scale", 1.0),
            kl_loss_scale=cfg.train.get("kl_loss_scale", 1.0),
            dual_loss_scale=cfg.train.get("dual_loss_scale", 1.0)
        ).to(self.device)

        # 4. Re-initialize Actor Optimizer to include WPO Dual Parameters
        # The parent class init created an optimizer for actor_ft only.
        # We need to optimize [actor_ft params] + [wpo_loss dual params]
        self.actor_optimizer = torch.optim.AdamW(
            [
                {'params': self.model.actor_ft.parameters()},
                {'params': self.wpo_loss.parameters(), 'lr': cfg.train.get("dual_lr", 1e-2)}
            ],
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )

        # We need to recreate the scheduler because the optimizer changed
        # Assuming you have the import available for CosineAnnealingWarmupRestarts
        from util.scheduler import CosineAnnealingWarmupRestarts
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

    def soft_update_target(self):
        """Soft update target actor parameters."""
        with torch.no_grad():
            for param, target_param in zip(self.model.actor_ft.parameters(), self.target_actor.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def run(self):
        """
        Main Loop. Most logic is identical to PPODiffusion, but the optimization loop is overridden.
        """
        # ... (Reuse the setup logic from TrainPPODiffusionAgent until the update loop)
        # To avoid copying 300 lines of code, I will focus on the Override part.
        # In a real file, you would copy the `run` method from `TrainPPODiffusionAgent`
        # and replace the "Update models" block.

        # BELOW IS THE LOGIC THAT GOES INSIDE THE LOOP, REPLACING THE PPO UPDATE BLOCK

        # ... [Data collection happens here, stored in obs_k, chains_k, etc.] ...
        # ... [Value bootstrapping happens here, stored in advantages_k, etc.] ...

        # --- START OF WPO UPDATE BLOCK ---

        # Note: We still use 'update_epochs' and 'batch_size' from config
        total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps

        for update_epoch in range(self.update_epochs):
            inds_k = torch.randperm(total_steps, device=self.device)
            num_batch = max(1, total_steps // self.batch_size)

            for batch in range(num_batch):
                start = batch * self.batch_size
                end = start + self.batch_size
                inds_b = inds_k[start:end]

                # Unravel indices to get batch and denoising step
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                )

                # Prepare Batch Data
                # WPO needs: Obs, Denoising Step, and Current Chains
                obs_b = {"state": obs_k["state"][batch_inds_b]}

                # We need gradients w.r.t actions.
                # In Diffusion, the "action" is the output of the denoising step (mu or epsilon).
                # We need to query the actor to get distributions.

                # ----------------------------------------------------
                # A. Get Distributions (Online and Target)
                # ----------------------------------------------------
                # We assume self.model.get_distribution(obs, step) returns Independent(Normal)
                # You might need to add this wrapper to your model class if not present.
                # Assuming the model has a way to return the distribution object:

                # Current Policy
                current_dist = self.model.actor_ft.get_distribution(
                    obs_b,
                    denoising_inds_b
                )

                # Target Policy
                with torch.no_grad():
                    target_dist = self.target_actor.get_distribution(
                        obs_b,
                        denoising_inds_b
                    )

                # ----------------------------------------------------
                # B. Sample Actions for Q-Gradient Estimation
                # ----------------------------------------------------
                # [Batch, N, Dim]
                actions_sampled = current_dist.sample((self.wpo_num_samples,))
                actions_sampled = einops.rearrange(actions_sampled, "n b d -> b n d")
                actions_sampled.requires_grad = True

                # ----------------------------------------------------
                # C. Compute Q-Gradients (The Critic)
                # ----------------------------------------------------
                # CRITICAL: WPO needs Q(s, a). DPPO usually has V(s).
                # You must ensure self.model.critic can handle (obs, action) input.
                # If your critic is only V(s), WPO cannot be mathematically applied directly.
                # Assuming your Critic structure supports Q-evaluation:

                # Expand obs for N samples
                obs_expanded = obs_b["state"].unsqueeze(1).expand(-1, self.wpo_num_samples, -1)
                flat_obs = {"state": einops.rearrange(obs_expanded, "b n d -> (b n) d")}
                flat_actions = einops.rearrange(actions_sampled, "b n d -> (b n) d")

                # Evaluate Q
                flat_q_values = self.model.critic(flat_obs, flat_actions) # Expecting [B*N, 1]
                q_values = flat_q_values.view(self.batch_size, self.wpo_num_samples)

                # Compute Gradients dQ/da
                q_grads = torch.autograd.grad(
                    outputs=q_values.sum(),
                    inputs=actions_sampled,
                    create_graph=False,
                    retain_graph=False
                )[0]

                # ----------------------------------------------------
                # D. Compute WPO Loss
                # ----------------------------------------------------
                wpo_loss_val, wpo_stats = self.wpo_loss(
                    policy_dist=current_dist,
                    target_policy_dist=target_dist,
                    actions_sampled=actions_sampled,
                    q_grad_wrt_actions=q_grads
                )

                # Add auxiliary losses from parent (e.g. BC loss if enabled)
                # We might need to manually calc BC loss if not using model.loss()
                total_loss = wpo_loss_val
                if self.use_bc_loss:
                    # Retrieve expert action (chains_next_b is roughly the target in DPPO)
                    # Implementation depends on how BC is defined in your `model.loss`
                    pass

                # ----------------------------------------------------
                # E. Update Steps
                # ----------------------------------------------------

                # Actor Update
                self.actor_optimizer.zero_grad()
                total_loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.actor_ft.parameters(), self.max_grad_norm
                    )
                self.actor_optimizer.step()

                # Critic Update (Standard MSE / PPO style)
                # We use the computed returns/advantages from GAE (calculated outside loop)
                # This part remains similar to PPO, using V_loss
                values_pred = self.model.critic(obs_b) # V(s)
                v_loss = 0.5 * ((values_pred - returns_b) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                self.critic_optimizer.step()

                # Eta Update (if learnable)
                if self.learn_eta:
                    # This logic remains from DPPO
                    pass

            # Soft Update Target Network at end of epoch (or batch)
            self.soft_update_target()

        # --- END OF WPO UPDATE BLOCK ---

        # ... (Resume standard logging logic from TrainPPODiffusionAgent)
        # Log wpo_stats to wandb