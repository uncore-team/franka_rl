import sys
import os

sys.path.append(os.path.abspath("../../"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3 import ...

#from rl_spin_decoupler import BaselinesSide
from rl_spin_decoupler import RLSide
from typing import Dict

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
        # Es espacio de observaciones: muchos arrays... float32 ¿?
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
  
        # Comunicación con panda_side.py
        #self._commstopanda = BaselinesSide(49054)
        self._commstopanda = RLSide(49054)


    def step(self, action):
        # Transform the format of the action
        # action = self._taskdef.BaselinesActToComm(action)
        action = self.act_to_comm(action)

        # Send action to Panda and receive observation
        #observation = self._commstopanda.stepGetObsSendAct(action)
        lat, observation, reward = self._commstopanda.stepSendActGetObs(action)
        observation = self.comm_to_obs(observation)
        #print(observation)#

        # Calculate terminated and truncated
        terminated, truncated = self.is_terminated_truncated(observation)

        # Calculate reward
        #reward = self.reward(observation, terminated)

        # End of episode
        if terminated or truncated:
            self._commstopanda.stepExpFinished()

        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = self._commstopanda.resetGetObs()
        observation = self.comm_to_obs(observation)
        info = None
        return observation, info
    
    def is_terminated_truncated(self, observation):
        return False, False
    
    """def reward(self, observation, terminated):
        return 0"""
    
    def get_info(self):
        return None
    
    def act_to_comm(self, action) -> Dict:
        return {"action":action}

    def comm_to_obs(self, observation) -> Dict:
        return observation["observation"]
    

if __name__ == '__main__':
    env = PandaEnv()
    # model = ...

    L = 0.15
    V = 0.1
    SIDE = False
    FIRST = True
    action = np.zeros(shape=(7,), dtype=np.float32)
    
    observation, info = env.reset(seed=42)
    #print(observation)
    for _ in range(10000):
        #action = env.action_space.sample()
        #action, _state = model.predict(obs, deterministic=True)
        #print(action)

        """Line y-axis"""
        
        if observation[1]>L and not(SIDE):
            action[1] = -V
            SIDE = True
        elif observation[1]<-L and SIDE:
            action[1] = V
            SIDE = False
        elif FIRST:
            action[1] = V
            FIRST = False
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
