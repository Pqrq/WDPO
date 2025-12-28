import numpy as np
import torch
import logging
import einops
import wandb
import math
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

        # ==========================
        # Initialization stuff
        # ==========================



        pass

    def run(self):
        """
        Combination of PPO Data Collection (Diffusion Sampling) 
        and WPO Replay Buffer storage.
        """
        prev_obs_venv = self.reset_env_all()
        # Initialization stuff

        # Main loop
        while self.itr < self.n_train_itr:
            # (Topragin kodunda eval mode muhabbetleri var burada, simdilik pas gectim)
            
            
            # ==========================
            # 1. Collect data
            # ==========================
            # 1.1: Reset your environment
            #prev_obs_venv = self.reset_env_all()

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

            # 1.2: Create holders
            obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
            chains_trajs = np.zeros((self.n_steps, self.n_envs,self.model.ft_denoising_steps + 1,self.horizon_steps,self.action_dim,))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            #if self.save_full_observations:  # state-only
            #    obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            #    obs_full_trajs = np.vstack((obs_full_trajs, prev_obs_venv["state"][:, -1][None]))

            # 1.3: Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 100 == 0:
                    log.info(f"Collecting step {step}/{self.n_steps}")
                
                # 1.3.1: Select action 
                with torch.no_grad():
                    flat_prev_obs = prev_obs_venv['state'].reshape(self.n_envs, -1)
                    
                    cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                    samples = self.model(cond=cond,
                        #deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (samples.trajectories.cpu().numpy())  # n_env x horizon x act
                    chains_venv = (samples.chains.cpu().numpy())  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # 1.3.2: Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # 1.3.3: Store in Buffer 
                # We flatten observation for the buffer to match network input
                flat_obs = obs_venv['state'].reshape(self.n_envs, -1)

                # Expand rewards/dones for buffer dimensions [n_envs, 1]
                self.replay_buffer.add(
                    flat_prev_obs,
                    output_venv,
                    chains_venv,
                    action_venv,
                    reward_venv[:, None],
                    flat_obs,
                    done_venv[:, None]
                )

                # 1.3.4: Update for next step
                prev_obs_venv = obs_venv

            # 1.4: Collect rewards (copy-pasted from DPPO training)
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
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
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
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            
            # ==========================
            # 2. Training Step (Update)
            # ==========================

            # ==========================
            # 3. Logging and Saving
            # ==========================
            # Might be directly copy pasted from one of the proposed models
            # (like either dppo or wpo, will depend on our models)
  
            
            
            # Back to the lab again...
            self.itr += 1 
        
    # Gozumun nuru
    def _update(self, obs_trajs, chains_trajs):
        # Sample Batch mi gelecek buraya
        #obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        # ==========================
        # A. Critic Update
        # ==========================
        with torch.no_grad():
            obs_trajs["state"] = (torch.from_numpy(obs_trajs["state"]).float().to(self.device))

            # A.1: Calculate q_value and logprobs
            # (split into batches to prevent out of memory)
            num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
            obs_ts = [{} for _ in range(num_split)]
            obs_k = einops.rearrange(obs_trajs["state"],"s e ... -> (s e) ...",)
            obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
            
            for i, obs_t in enumerate(obs_ts_k):
                obs_ts[i]["state"] = obs_t
            
            q_values_trajs = np.empty((0, self.n_envs))
            for obs in obs_ts:
                q_values = self.model.critic(obs).cpu().numpy().flatten() # Sampling fron critic model???
                q_values_trajs = np.vstack((q_values_trajs, q_values.reshape(-1, self.n_envs)))
            chains_t = einops.rearrange(torch.from_numpy(chains_trajs).float().to(self.device),"s e t h d -> (s e) t h d",)
            chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
            logprobs_trajs = np.empty((0,self.model.ft_denoising_steps,self.horizon_steps,self.action_dim,))
            for obs, chains in zip(obs_ts, chains_ts):
                logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                logprobs_trajs = np.vstack((logprobs_trajs,logprobs.reshape(-1, *logprobs_trajs.shape[1:]),))
            # DPPO code calculated advantage function here, I think mine should be
            # somewhat different

        # k for environment step
        obs_k = {"state": einops.rearrange(obs_trajs["state"],"s e ... -> (s e) ...",)}
        chains_k = einops.rearrange(torch.tensor(chains_trajs, device=self.device).float(),"s e t h d -> (s e) t h d",)
        q_values_k = (torch.tensor(q_values_trajs, device=self.device).float().reshape(-1))
        # DPPO also had returns_trajs and advantages_trajs here, mine will probably be different
        logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

        # ==========================
        # B. Actor Update
        # ==========================




        pass

    def wpo_loss():
        pass
