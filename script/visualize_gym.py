import os
import torch
import imageio
import numpy as np
import gym
import d4rl
from omegaconf import OmegaConf
import datetime

# --- RESOLVERS SETUP ---
try:
    OmegaConf.register_new_resolver("now", lambda pattern: datetime.datetime.now().strftime(pattern))
except ValueError: pass 

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError: pass

# --- IMPORT ---
from agent.eval.eval_diffusion_agent import EvalDiffusionAgent as Agent

# ==========================================
# 1. PASTE YOUR PATHS HERE
# ==========================================
CHECKPOINT_PATH = "/home/toprak/dppo/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/my_pretrain-2025-11-21/checkpoint/state_200.pt" # from my 200 step pretrain
# CHECKPOINT_PATH = "/home/toprak/dppo/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt" # from downloaded 3000 step pretrain
# CHECKPOINT_PATH = "/home/toprak/dppo/log/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/my_finetune_from_3000-2025-11-21/checkpoint/state_999.pt" # from my finetuning of 3000 step
# CHECKPOINT_PATH = "/home/toprak/dppo/log/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/my_finetune_from_my_200-2025-11-22/checkpoint/state_999.pt" # from my finetuning of my 200 step
CONFIG_PATH = "/home/toprak/dppo/cfg/gym/eval/hopper-v2/eval_diffusion_mlp.yaml"
OUTPUT_VIDEO = "demo_videos/test.gif"  # Changed from .mp4 to .gif
# ==========================================

def load_trained_policy():
    print(f"Loading config from {CONFIG_PATH}...")
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.device = "cuda:0"
    if 'wandb' in cfg: cfg.wandb.mode = 'offline'
    cfg.base_policy_path = CHECKPOINT_PATH

    # 1. Initialize the Container Agent
    agent = Agent(cfg)
    
    # 2. Extract the actual Policy (The Neural Network)
    # Based on TrainDiffusionAgent, it inherits from PreTrainAgent which sets self.model
    if hasattr(agent, 'model'):
        policy = agent.model
    else:
        raise AttributeError("Cannot find 'model' inside the Agent.")

    # # 3. Load Weights directly into the Policy
    # print(f"Loading weights from {CHECKPOINT_PATH}...")
    # state_dict = torch.load(CHECKPOINT_PATH, map_location='cuda')
    #
    # # Handle different checkpoint structures
    # if 'state_dict' in state_dict:
    #     # If the checkpoint is the agent state, we need to be careful about keys
    #     # Usually agent checkpoints have "model.network..." keys
    #     # But if we are loading into policy directly, we might need to adjust keys
    #     # Let's try loading into the agent first if possible, it handles prefixing
    #     agent.load_state_dict(state_dict['state_dict'])
    # elif 'model' in state_dict:
    #     policy.load_state_dict(state_dict['model'])
    # else:
    #     policy.load_state_dict(state_dict)
        
    policy.eval()
    return policy, cfg

def run_eval():
    # Load Policy
    policy, cfg = load_trained_policy()
    
    # Setup Environment
    env_name = cfg.env.name
    print(f"Detected environment: {env_name}")

    env = gym.make(env_name)
    
    video_writer = imageio.get_writer(OUTPUT_VIDEO, fps=30)
    
    num_resets_to_record = 1
    
    print("Starting rendering...")
    
    with torch.no_grad():
        for _ in range(num_resets_to_record):
            obs = env.reset()
            done = False
            step = 0
            while not done:
                # 1. Prepare state Tensor
                state_tensor = torch.from_numpy(obs).float().to("cuda").unsqueeze(0)

                # 2. Wrap in Dictionary (THE FIX)
                # The model expects a dict with "state" key
                cond = {"state": state_tensor}

                # 3. Generate Trajectory (THE FIX)
                # Calls DiffusionModel.forward(cond)
                output = policy(cond)

                # 4. Extract Action
                # Output is a named tuple (trajectories, chains)
                # Trajectories shape: (Batch, Horizon, ActionDim)
                # We take the first action of the predicted horizon (MPC style)
                trajectory = output.trajectories
                action = trajectory[0, 0, :]

                # Convert to Numpy
                action = action.cpu().detach().numpy()

                # Step
                obs, reward, done, info = env.step(action)

                # Render
                frame = env.render(mode='rgb_array')
                video_writer.append_data(frame)

                step += 1
                if step % 50 == 0: print(f"Step {step}...")

            env.seed() # change seed for the next take

    video_writer.close()
    env.close()
    print(f"Done! Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_eval()
