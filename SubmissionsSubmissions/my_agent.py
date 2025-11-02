import os
import numpy as np
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO

class SubmittedAgent(Agent):
    """
    Submission-ready agent using your trained PPO model.
    """
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        # If no path is provided, use the local trained model
        self.file_path = file_path or "shaped_imitation_checkpoints_selfplay/ppo_shaped_selfplay_final.zip"
        self.model = None

    def _initialize(self) -> None:
        # Load your trained PPO model
        self.model = PPO.load(self.file_path)

    def predict(self, obs):
        obs = np.array(obs, dtype=np.float32)
        # Add batch dimension if necessary
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return action[0]

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
