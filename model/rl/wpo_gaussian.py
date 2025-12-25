import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import hydra
from torch.distributions import Normal, Independent


# --- Custom MLP Actor ---
class GaussianActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=[256, 256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.414)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.trunk(x)
        mu = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return Independent(Normal(mu, std), 1)


class WPO_Gaussian(nn.Module):
    def __init__(
            self,
            actor,
            critic,
            horizon_steps,
            device,
            action_dim,
            init_log_alpha_mean=0.0,
            init_log_alpha_std=0.0,
            per_dim_constraining=True,
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        # 1. Instantiate Custom Actor
        self.actor = GaussianActor(
            input_dim=actor.get('cond_dim'),
            action_dim=action_dim,
            hidden_dims=actor.get('mlp_dims', [256, 256, 256])
        ).to(device)

        # 2. Instantiate Repository Critic (CriticObsAct)
        # We pass arguments required by CriticObsAct
        self.critic = hydra.utils.instantiate(
            critic,
            action_dim=action_dim,
            action_steps=1
        ).to(device)

        # 3. Target Networks
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.target_actor.eval()
        self.target_critic.eval()

        # 4. Dual Parameters
        shape = (action_dim,) if per_dim_constraining else (1,)
        self.log_alpha_mean = nn.Parameter(torch.full(shape, init_log_alpha_mean, device=device))
        self.log_alpha_std = nn.Parameter(torch.full(shape, init_log_alpha_std, device=device))

    def forward(self, cond, deterministic=False):
        dist = self.actor(cond)
        if deterministic:
            return dist.base_dist.loc
        else:
            return dist.sample()

    def get_q(self, obs, action, target=False):
        """
        Helper to handle input formatting for CriticObsAct
        """
        # CriticObsAct expects cond={'state': ...}
        cond = {'state': obs}

        net = self.target_critic if target else self.critic

        # CriticObsAct returns (q1, q2) if double_q=True, else q1
        # We assume double_q=False for now to match your WPO algorithm 1-to-1
        return net(cond, action)

    def get_alpha(self, log_alpha):
        return F.softplus(log_alpha) + 1e-8

    def hard_update_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())