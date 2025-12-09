"""
WDPO fine-tuning.

"""

import numpy as np
import torch
import logging
import wandb

log = logging.getLogger(__name__)

from agent.finetune.train_ppo_agent import TrainPPOAgent
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent


class TrainWDPOAgent(TrainPPODiffusionAgent): # PPODiffusion can be thought of standard DPPO, on which WDPO is based
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        pass