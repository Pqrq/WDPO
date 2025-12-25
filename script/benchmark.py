import torch
import wandb
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

# Adjust imports to match your file structure
from agent.finetune.train_wpo_agent import WPOAgent
from env.bandit import ContinuousBanditEnv


# =============================================================================
# 1. Training Function (Logs to WandB)
# =============================================================================

def train_benchmark(
        env,
        seed=42,
        total_steps=1000,
        wandb_project_name="wpo-vs-baselines",
        wandb_run_name="bandit_benchmark",
        wandb_group="bandit_comparison_experiment"
):
    """
    Trains Multiple Methods sequentially in the same WandB run context.
    """
    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        group=wandb_group,
        reinit=True  # Important for loop execution
    )

    # --- 1. Train WPO ---
    print(f"--- Training WPO (Seed {seed}) ---")
    wpo_agent = WPOAgent(
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        warmup_steps=0,
        epsilon_mean=0.0001,  # Fix for overshoot
        epsilon_std=0.0001,
    )
    torch.manual_seed(seed)
    # Custom loop to log WPO steps to WandB
    # (Assuming your WPOAgent logs internally, but we ensure keys match)
    wpo_agent.learn(total_timesteps=total_steps)

    # --- 2. Train PPO ---
    print(f"--- Training PPO (Seed {seed}) ---")
    env_ppo = gym.wrappers.RecordEpisodeStatistics(env)
    model_ppo = PPO(
        "MlpPolicy",
        env_ppo,
        seed=seed,
        learning_rate=3e-4,
        gamma=0.0,
        n_steps=32,
        batch_size=32,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64])
    )

    class WandbLoggerPPO(BaseCallback):
        def _on_step(self):
            if 'episode' in self.locals['infos'][0]:
                wandb.log({
                    "PPO/reward": self.locals['infos'][0]['episode']['r'],
                    "PPO/step": self.num_timesteps
                })
            return True

    model_ppo.learn(total_timesteps=total_steps, callback=WandbLoggerPPO())

    # --- 3. Train SAC ---
    print(f"--- Training SAC (Seed {seed}) ---")
    env_sac = gym.wrappers.RecordEpisodeStatistics(env)
    model_sac = SAC(
        "MlpPolicy",
        env_sac,
        seed=seed,
        learning_rate=3e-4,
        gamma=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),
        ent_coef='auto'
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


# =============================================================================
# 2. Data Fetching & Plotting
# =============================================================================

def fetch_and_plot(project_name, group_name, methods=["WPO", "PPO", "SAC"]):
    print("\nDownloading data from WandB...")
    api = wandb.Api()

    # Fetch runs that match the group
    runs = api.runs(path=project_name, filters={"group": group_name})

    if len(runs) == 0:
        print("No runs found! Check your project/group names.")
        return

    data = {m: [] for m in methods}

    # Iterate through all runs (seeds)
    for run in runs:
        # Download history (all logged metrics)
        # Using a large sample size to get all points
        hist = run.history(samples=10000)

        for method in methods:
            # Filter for rows where this method logged data
            # Key format: "Method/reward" and "Method/step"
            cols = [f"{method}/reward", f"{method}/step"]
            if cols[0] in hist.columns:
                df_method = hist[cols].dropna()
                # Sort by step to be safe
                df_method = df_method.sort_values(f"{method}/step")

                # Convert to simple numpy array [steps]
                # We assume steps are 0..1000. If logging skips, we might need interpolation.
                # For this simple case, taking the values is usually sufficient.
                rewards = df_method[f"{method}/reward"].values

                # Truncate or pad if lengths slightly differ due to logging
                data[method].append(rewards)

    # Plotting
    print("Generating Plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    colors = {"WPO": "#1f77b4", "PPO": "#ff7f0e", "SAC": "#2ca02c"}

    for method in methods:
        runs_np = np.array(data[method])

        if runs_np.size == 0:
            print(f"No data found for {method}")
            continue

        # Handle potentially ragged arrays (if some runs crashed or length differs)
        # Here we truncate to the minimum length found across seeds
        min_len = min(len(r) for r in runs_np)
        runs_np = np.stack([r[:min_len] for r in runs_np])


        # Statistics
        mean_reward = np.mean(runs_np, axis=0)
        min_reward = np.min(runs_np, axis=0)
        max_reward = np.max(runs_np, axis=0)

        mean_smoothed = pd.Series(mean_reward).rolling(window=10, min_periods=1).mean().values
        min_smoothed = pd.Series(min_reward).rolling(window=10, min_periods=1).mean().values
        max_smoothed = pd.Series(max_reward).rolling(window=10, min_periods=1).mean().values

        steps = np.arange(len(mean_reward))

        # Plot Mean
        plt.plot(steps, mean_smoothed, label=method, color=colors[method], linewidth=2)

        # Plot Variance Area
        plt.fill_between(
            steps,
            min_smoothed,
            max_smoothed,
            color=colors[method],
            alpha=0.2
        )

    plt.title("Method Comparison (Mean ± Range over Seeds)")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("final_comparison.png")
    plt.show()
    print("Plot saved to final_comparison.png")


# =============================================================================
# 3. Execution Block
# =============================================================================

if __name__ == "__main__":
    PROJECT_NAME = "wpo-vs-baselines"
    GROUP_NAME = "bandit_comparison_experiment_v3"

    # A. Run Training (Logs to cloud)
    num_seeds_to_test = 1
    for _ in range(num_seeds_to_test):
        seed = np.random.randint(0, 10000)
        print(f"\n=== Starting Benchmark for Seed {seed} ===")
        env = ContinuousBanditEnv()
        env.reset(seed=seed)

        run_name = f"bandit_seed_{seed}"

        train_benchmark(
            env=env,
            seed=seed,
            total_steps=1000,
            wandb_project_name=PROJECT_NAME,
            wandb_run_name=run_name,
            wandb_group=GROUP_NAME
        )

    # B. Download & Plot
    # Ensure your WPOAgent logic logs keys: "WPO/reward" and "WPO/step"
    fetch_and_plot(
        project_name=f"{wandb.api.default_entity}/{PROJECT_NAME}",  # Uses your default wandb username
        group_name=GROUP_NAME
    )