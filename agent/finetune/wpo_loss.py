import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence

class WPOLoss(nn.Module):
    """
    PyTorch implementation of Wasserstein Policy Optimization (WPO) Loss.
    """
    def __init__(
        self,
        action_dim: int,
        epsilon_mean: float = 0.001,
        epsilon_std: float = 0.00001,
        init_log_alpha_mean: float = 5.0,
        init_log_alpha_std: float = 5.0,
        policy_loss_scale: float = 1.0,
        kl_loss_scale: float = 1.0,
        dual_loss_scale: float = 1.0,
        per_dim_constraining: bool = True
    ):
        super().__init__()
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.policy_loss_scale = policy_loss_scale
        self.kl_loss_scale = kl_loss_scale
        self.dual_loss_scale = dual_loss_scale
        self.per_dim_constraining = per_dim_constraining

        # Dual Parameters (Lagrangian Multipliers)
        shape = (action_dim,) if per_dim_constraining else (1,)
        self.log_alpha_mean = nn.Parameter(torch.full(shape, init_log_alpha_mean))
        self.log_alpha_std = nn.Parameter(torch.full(shape, init_log_alpha_std))

    def get_alpha(self, log_alpha):
        return F.softplus(log_alpha) + 1e-4

    def forward(
        self,
        policy_dist: Independent,
        target_policy_dist: Independent,
        q_grad_wrt_actions: torch.Tensor, # [B, N, D]
    ):
        # 1. Unpack Distributions (Assume Normal)
        mu = policy_dist.base_dist.loc
        sigma = policy_dist.base_dist.scale
        target_mu = target_policy_dist.base_dist.loc
        target_sigma = target_policy_dist.base_dist.scale

        # 2. Wasserstein Drift (Policy Gradient)
        # Direction = sigma^2 * E[grad_a Q]
        avg_q_grad = torch.mean(q_grad_wrt_actions, dim=1) # [B, D]
        drift_target_mu = (sigma.pow(2) * avg_q_grad).detach()

        # Surrogate Loss: L = - (mu * drift) -> grad L = -drift -> grad descent = +drift
        loss_policy_drift = -torch.sum(mu * drift_target_mu, dim=-1).mean()

        # 3. KL Constraints (MPO Decomposition)
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
        alpha_mean = self.get_alpha(self.log_alpha_mean)
        alpha_std = self.get_alpha(self.log_alpha_std)

        # Penalty term (for policy)
        loss_kl_penalty = torch.sum(alpha_mean.detach() * mean_kl_mean) + \
                          torch.sum(alpha_std.detach() * mean_kl_std)

        # Dual term (for alpha update)
        loss_dual = torch.sum(alpha_mean * (self.epsilon_mean - mean_kl_mean.detach())) + \
                    torch.sum(alpha_std * (self.epsilon_std - mean_kl_std.detach()))

        # Total
        total_loss = (
            self.policy_loss_scale * loss_policy_drift +
            self.kl_loss_scale * loss_kl_penalty +
            self.dual_loss_scale * loss_dual
        )

        stats = {
            "wpo_loss": total_loss.item(),
            "loss_policy": loss_policy_drift.item(),
            "loss_kl_penalty": loss_kl_penalty.item(),
            "loss_dual": loss_dual.item(),
            "kl_mean": mean_kl_mean.mean().item(),
            "kl_std": mean_kl_std.mean().item(),
            "alpha_mean": alpha_mean.mean().item(),
            "alpha_std": alpha_std.mean().item()
        }

        return total_loss, stats