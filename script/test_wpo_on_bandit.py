import torch
import wandb
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from agent.finetune.wpo_loss import WPOLoss
from agent.finetune.train_wpo_agent import WPOAgent
from env.bandit import ContinuousBanditEnv

# =============================================================================
# 3. Main Benchmark Function
# =============================================================================
def train_benchmark(total_steps=1000):
    """
    Trains WPO, PPO, and SAC alongside each other and logs to Wandb.
    """
    wandb.init(project="wpo-vs-baselines", name="bandit_benchmark")

    env_bandit = ContinuousBanditEnv()

    # 1. Train WPO
    print("--- Training WPO ---")
    wpo_agent = WPOAgent(
        env=env_bandit,
        hidden_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    wpo_agent.learn(total_timesteps=total_steps)

    # 2. Train PPO (Stable Baselines 3)
    print("--- Training PPO ---")
    # Monitor wrapper is needed for SB3 to track stats properly
    env_ppo = gym.wrappers.RecordEpisodeStatistics(env_bandit)

    model_ppo = PPO(
        "MlpPolicy",
        env_ppo,
        learning_rate=3e-4,
        gamma=0.0,  # Crucial for Bandits (no future)
        policy_kwargs=dict(net_arch=[64, 64])  # Match WPO size
    )

    # Custom Callback to log PPO specific keys to same Wandb run
    class WandbLogger(BaseCallback):
        def _on_step(self):
            # Log reward from the wrapper
            if 'episode' in self.locals['infos'][0]:
                wandb.log({
                    "PPO/reward": self.locals['infos'][0]['episode']['r'],
                    "PPO/step": self.num_timesteps
                })
            return True

    model_ppo.learn(total_timesteps=total_steps, callback=WandbLogger())

    # 3. Train SAC (Stable Baselines 3)
    print("--- Training SAC ---")
    env_sac = gym.wrappers.RecordEpisodeStatistics(env_bandit)

    model_sac = SAC(
        "MlpPolicy",
        env_sac,
        learning_rate=3e-4,
        gamma=0.0,  # Crucial for Bandits
        policy_kwargs=dict(net_arch=[64, 64]),
        ent_coef='auto'  # Auto-tune entropy (Standard for SAC)
    )

    class WandbLoggerSAC(BaseCallback):
        def _on_step(self):
            if 'episode' in self.locals['infos'][0]:
                wandb.log({
                    "SAC/reward": self.locals['infos'][0]['episode']['r'],
                    "SAC/step": self.num_timesteps
                })
            return True

    model_sac.learn(total_timesteps=total_steps, callback=WandbLoggerSAC())

    wandb.finish()
    print("Benchmark Complete. Check Wandb for 'WPO/reward' vs 'PPO/reward' vs 'SAC/reward'.")


if __name__ == "__main__":

    for _ in range(5):  # Run 3 times for robustness
        train_benchmark()