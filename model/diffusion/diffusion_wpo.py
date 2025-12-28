"""
Wasserstein policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import torch
import logging
import hydra

log = logging.getLogger(__name__)
import torch.nn.functional as F
import torch.nn as nn

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
from torch.distributions import Normal, Independent, kl_divergence


class WPODiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        horizon_steps,
        device,
        action_dim,
        ft_denoising_steps, # Not in Toprak
        init_log_alpha_mean=0.0,
        init_log_alpha_std=0.0,
        per_dim_constraining=True,
        # Other above are same with Toprak

        #horizon_steps=1, # This was in Toprak
        #action_dim=1, # This was in Toprak

        # Below are not in Toprak, these are diffusion-based parameters
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        eta=None,
        learn_eta=False,
        #epsilon_mean: float = 0.001,
        #epsilon_std: float = 0.00001,
        #policy_loss_scale: float = 1.0,
        #kl_loss_scale: float = 1.0,
        #dual_loss_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            **kwargs,
        )
        #self.horizon_steps = horizon_steps
        self.action_dim = action_dim

        # Re-name network to actor
        self.actor = self.network
        
        # By default do not freeze actor (actor is trainable)
        # Leave actor_ft frozen until you explicitly want to train it
        #self.actor_ft = copy.deepcopy(self.actor).to(device)
        #for p in self.actor_ft.parameters():
        #    p.requires_grad = False
        
        self.critic = critic.to(device)

        # 3. Target Networks
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.target_actor.eval()
        self.target_critic.eval()

        # Dual Parameters (Lagrangian Multipliers)
        # Use full flattened event-dimension when constraining per-dim in noise-space:
        event_dim = self.horizon_steps * self.action_dim
        shape = (event_dim,) if per_dim_constraining else (1,)
        self.log_alpha_mean = nn.Parameter(torch.full(shape, init_log_alpha_mean))
        self.log_alpha_std  = nn.Parameter(torch.full(shape, init_log_alpha_std))


        # Diffusion-related initializations (these are not in Toprak)
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        # Number of denoising steps to use with fine-tuned model. Thus denoising_step - ft_denoising_steps is the number of denoising steps to use with original model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                logging.info("Turned off gradients for eta")

        ########## WPO-specific parts ########## 
        #self.epsilon_mean = epsilon_mean
        #self.epsilon_std = epsilon_std
        #self.policy_loss_scale = policy_loss_scale
        #self.kl_loss_scale = kl_loss_scale
        #self.dual_loss_scale = dual_loss_scale
        self.per_dim_constraining = per_dim_constraining

        # At end of WPODiffusion.__init__ (temporary debug)
        assert tuple(self.log_alpha_mean.shape) == ((self.horizon_steps * self.action_dim,) if self.per_dim_constraining else (1,))

    # Diffusion forwarding, naturally different from Toprak
    # Probably copy-pasted from diffusion_vpg code
    # Btw I am not sure whether we should use torch_no_grad
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        #use_base_policy=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Get updated minimum sampling denoising std
        #min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                #use_base_policy=use_base_policy,
                #deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = torch.clip(std, min=1e-3)
                else:  # use higher minimum noise
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)
    
    # override
    # Probably copy-pasted from diffusion_vpg code
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.target_actor

        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)  # always one for DDPM
        return mu, logvar, etas
    
    
    # Below are copy-pasted from Toprak
    def get_q(self, obs, action, target=False): 
        """
        Helper to handle input formatting for CriticObsAct
        """
        cond = {'state': obs} # CriticObsAct expects cond={'state': ...}
        net = self.target_critic if target else self.critic
        return net(cond, action) # CriticObsAct returns (q1, q2) if double_q=True, else q1
    def get_alpha(self, log_alpha): 
        return F.softplus(log_alpha) + 1e-8
    def hard_update_targets(self): 
        """Hard copy actor/critic weights into internal targets."""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    def soft_update_targets(self, tau=0.005): 
        """
        Polyak averaging: target = tau * current + (1-tau) * target
        Update internal target_actor and target_critic.
        """
        with torch.no_grad():
            # Update Actor Targets
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

            # Update Critic Targets
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
