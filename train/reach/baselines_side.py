import sys
import os
sys.path.append(os.path.abspath("../../"))

from typing import Dict
#import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_spin_decoupler.spindecoupler import RLSide


class PandaEnv(gym.Env):

    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            #"pos": spaces.Box(low=-1.5, high=1.5, shape=(3,), dtype=float), 
            "rel_pos": spaces.Box(-1.5,+1.5, shape=(3,), dtype=float),
            "goal_dist": spaces.Box(low=0, high=3, shape=(1,), dtype=float),
            "force_mag": spaces.Box(low=0, high=100.0, shape=(1,), dtype=float)   
            })
  
        # Comunicación con panda_side.py
        self._commstopanda = RLSide(49054)

        self.max_steps = 300
        self.steps = 0

        self.max_force = 40
        #self.max_force2 = self.max_force/2

        self.min_dist = 0.2
        #self.min_dist3 = self.min_dist*3
        #self.last_dist = None


    def step(self, action):
        self.steps += 1
        # Transform the format of the action
        # action = self._taskdef.BaselinesActToComm(action)
        action = self.act_to_comm(action)

        # Send action to Panda and receive observation
        lat, observation, reward, agenttime = self._commstopanda.stepSendActGetObs(action)
        observation = self.comm_to_obs(observation)
        #print(observation)

        # Calculate terminated and truncated
        terminated = self.is_terminated(observation)
        truncated = self.is_truncated(observation)

        # Calculate reward
        reward = self.reward(observation, terminated, truncated)

        # End of learning
        #if steps > learning_steps:
        #    self._commstopanda.stepExpFinished()

        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_dist = None

        observation = self._commstopanda.resetGetObs()
        observation = self.comm_to_obs(observation)

        info = {}
        #print(observation[0])
        return observation[0], info
    
    def is_terminated(self, observation):
        #force = np.linalg.norm(observation["force"])
        #goal_distance = np.linalg.norm(observation["goal_pos"] - observation["pose"][0:3])
        if observation["force_mag"] > self.max_force:
            return True
        if observation["goal_dist"] < self.min_dist:
            return True
        return False
    
    def is_truncated(self, observation):
        if self.steps >= self.max_steps:
            return True
        return False
    
    def reward(self, observation, terminated, truncated):
        force_mag = observation["force_mag"]
        goal_dist = observation["goal_dist"]

        if force_mag > self.max_force: # and finished
            reward = -100
        elif goal_dist < self.min_dist: # and finished
            reward = 100 * (self.min_dist - goal_dist)
        elif truncated: # and truncated
            reward = -500
        else: # just one more step
            reward = -goal_dist * 10 
        return reward
    """reward = 0

        if truncated:
            return -10

        if observation["force_mag"] > self.max_force2:
            k = 10
            reward = reward - min(k*(observation["force_mag"] - self.max_force2)/self.max_force2,k)

        if self.last_dist!=None:
            inc = observation["goal_dist"]-self.last_dist
            if inc >= 0:
                reward = reward -10*inc
            else:
                reward = reward +10*inc
            
        self.last_dist = observation["goal_dist"]
        return reward"""

    
    def get_info(self):
        return {}
    
    def act_to_comm(self, action) -> Dict:
        return {"action":action}

    def comm_to_obs(self, observation) -> Dict:
        return observation
    

if __name__ == '__main__':
    env = PandaEnv()
    model = SAC("MultiInputPolicy", env, verbose=1)
    #model = SAC.load("checkpoints_1/reach_18000_steps.zip",env)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,  # Guarda cada 5000 timesteps
        save_path="./checkpoints_1/",  # Carpeta de guardado
        name_prefix="reach")
    model.learn(total_timesteps=100000,callback=checkpoint_callback)#40000
    model.save("reach.zip")


