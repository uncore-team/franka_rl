import sys
import os
sys.path.append(os.path.abspath("../../"))

from typing import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_spin_decoupler.spindecoupler import RLSide
from task import Task, TaskReach1, TaskReach2


class PandaEnv(gym.Env):

    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, task: Task):
        super().__init__()

        self.task = task

        self.action_space = self.task.RLActionSpace()

        self.observation_space = self.task.RLObservationSpace()

        # Comunicación con panda_side.py
        self._commstopanda = RLSide(49054)

        # CONSTANTS
        #self.max_force2 = self.max_force/2
        #self.min_dist3 = self.min_dist*3
        #self.last_dist = None


    def step(self, action):
        self.task.RLStep()
        # Transform the format of the action
        action = self.task.RLActionToComm(action)

        # Send action to Panda and receive observation
        # lat, observation, reward, agenttime
        _, observation, _, _ = self._commstopanda.stepSendActGetObs(action)
        observation = self.task.RLCommToObservation(observation)
        #print(observation)

        # Calculate terminated and truncated
        terminated = self.task.RLTerminated(observation)
        truncated = self.task.RLTruncated()

        # Calculate reward
        reward = self.task.RLReward(observation, terminated, truncated)

        # End of learning
        #if steps > learning_steps:
        #    self._commstopanda.stepExpFinished()

        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.task.RLReset()
        #self.last_dist = None

        observation, _ = self._commstopanda.resetGetObs()
        observation = self.task.RLCommToObservation(observation)

        info = {}
        return observation, info
    


if __name__ == '__main__':
    task = TaskReach2(mode=TaskReach2.TaskMode.LEARN)
    env = PandaEnv(task=task)
    model = SAC("MultiInputPolicy", env, verbose=1)
    #model = SAC.load("checkpoints_1/reach_18000_steps.zip",env)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=2000, 
        save_path="./checkpoints_1/",
        name_prefix="reach")
    model.learn(total_timesteps=50000,callback=checkpoint_callback)
    model.save("reach.zip")


