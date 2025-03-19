import sys
import os
sys.path.append(os.path.abspath("../../"))

from typing import Dict
#import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C

from rl_spin_decoupler.spindecoupler import RLSide


class PandaEnv(gym.Env):

    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        """
        Action spec: BoundedArray(shape=(6,), 
        dtype=dtype('float32'), 
        name='panda_twist0\tpanda_twist1\tpanda_twist2\tpanda_twist3\tpanda_twist4\tpanda_twist5', 
        minimum=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5], 
        maximum=[0.5 0.5 0.5 0.5 0.5 0.5])
        """
        # El espacio de acciones: vector de 7 float32 [-0.5,0.5]
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(7,), dtype=np.float32)
        """
        Observation spec:
        panda_tcp_pos: Array(shape=(3,), 
        dtype=dtype('float32'), 
        name='panda_tcp_pos')
        """
        #self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        # Es espacio de observaciones: 3 panda_tcp_pos + 3 goal_pose
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
  
        # Comunicación con panda_side.py
        #self._commstopanda = BaselinesSide(49054)
        self._commstopanda = RLSide(49054)

        self.target_position = None

        self.max_steps = 100
        self.steps = 0


    def step(self, action):
        self.steps += 1
        # Transform the format of the action
        # action = self._taskdef.BaselinesActToComm(action)
        action = self.act_to_comm(action)

        # Send action to Panda and receive observation
        #observation = self._commstopanda.stepGetObsSendAct(action)
        lat, observation, reward = self._commstopanda.stepSendActGetObs(action)
        observation = self.comm_to_obs(observation)
        observation = np.concatenate([observation, self.goal_pos])
        #print(observation)#

        # Calculate terminated and truncated
        terminated = self.is_terminated(observation)
        truncated = self.is_truncated(observation)

        # Calculate reward
        reward = self.reward(observation, terminated)

        # End of episode
        if terminated or truncated:
            self._commstopanda.stepExpFinished()

        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.steps = 0

        observation = self._commstopanda.resetGetObs()
        observation = self.comm_to_obs(observation)

        self.goal_pos = np.random.uniform(low=-10.0, high=10.0, size=(3,))  # Nueva posición aleatoria
        observation = np.concatenate([observation, self.goal_pos])

        info = {}
        return observation, info
    
    def is_terminated(self, observation):
        return False
    
    def is_truncated(self, observation):
        if self.steps >= self.max_steps:
            return True
        return False
    
    def reward(self, observation, terminated):
        goal_distance = np.linalg.norm(observation[0:2] - observation[3:5])
        return np.clip(1.0 - goal_distance, 0, 1)

    
    def get_info(self):
        return {}
    
    def act_to_comm(self, action) -> Dict:
        return {"action":action}

    def comm_to_obs(self, observation) -> Dict:
        return observation["observation"]
    

if __name__ == '__main__':
    env = PandaEnv()
    model = A2C("MlpPolicy",env)
    model.learn(total_timesteps=500)
    model.save("reach.zip")


    """
    observation, info = env.reset(seed=42)
    #print(observation)
    for _ in range(10000):
        action = env.action_space.sample()
        #action, _state = model.predict(observation, deterministic=True)
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
    env.close()"""
