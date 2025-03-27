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
        """panda_force: Array(shape=(3,), dtype=dtype('float32'), name='panda_force')"""
        #self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        # Es espacio de observaciones: 3 panda_tcp_pos + 3 panda_force + 3 goal_pos 
        #self.observation_space = spaces.Box(low=-10, high=10, shape=(9,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "pose": spaces.Box(low=np.array([0.1, -0.3, 0.1, -1, -1, -1, -1]), high=np.array([0.5, 0.3, 0.7, 1, 1, 1, 1]), shape=(7,), dtype=np.float32), 
            "vel": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
            "force": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),   
            "goal_pos": spaces.Box(low=np.array([0.1, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.7]), shape=(3,), dtype=np.float32)
        })
  
        # Comunicación con panda_side.py
        #self._commstopanda = BaselinesSide(49054)
        self._commstopanda = RLSide(49054)

        self.goal_pos = None

        self.max_steps = 250
        self.steps = 0

        self.max_force = 40
        self.min_dist = 0.2


    def step(self, action):
        self.steps += 1
        # Transform the format of the action
        # action = self._taskdef.BaselinesActToComm(action)
        action = self.act_to_comm(action)

        # Send action to Panda and receive observation
        lat, observation, reward = self._commstopanda.stepSendActGetObs(action)
        observation = self.comm_to_obs(observation)
        #print(observation)

        # Calculate terminated and truncated
        terminated = self.is_terminated(observation)
        truncated = self.is_truncated(observation)

        # Calculate reward
        reward = self.reward(observation, terminated)

        # End of learning
        #if steps > learning_steps:
        #    self._commstopanda.stepExpFinished()

        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.goal_pos = np.random.uniform(low=[0.1,-0.3,0.1], high=[0.5,0.3,0.7], size=(3,))  # Nueva posición aleatoria

        observation = self._commstopanda.resetGetObs()
        observation = self.comm_to_obs(observation)

        #observation = np.concatenate([observation, self.goal_pos])

        info = {}
        return observation, info
    
    def is_terminated(self, observation):
        force = np.linalg.norm(observation["force"])
        goal_distance = np.linalg.norm(observation["goal_pos"] - observation["pose"][0:3])
        if force > self.max_force:
            return True
        if goal_distance < self.min_dist:
            return True
        return False
    
    def is_truncated(self, observation):
        if self.steps >= self.max_steps:
            return True
        return False
    
    def reward(self, observation, terminated):
        # Corregir por separado la posición y la orientación
        goal_distance = np.linalg.norm(observation["goal_pos"] - observation["pose"][0:3]) # Distancia al objetivo
        force = np.linalg.norm(observation["force"])
        #vel = np.linalg.norm(observation["vel"][0:3])

        if force > self.max_force:
            return -100
        if goal_distance < self.min_dist:
            return 100
        
        reward = 100*(self.min_dist/goal_distance)-10   
        reward = reward-0.1 # Penalización por tiempo   
        return reward

    
    def get_info(self):
        return {}
    
    def act_to_comm(self, action) -> Dict:
        return {"action":action}

    def comm_to_obs(self, observation) -> Dict:
        
        observation["goal_pos"]=self.goal_pos
        #print(observation)
        return observation#["observation"]
    

if __name__ == '__main__':
    env = PandaEnv()
    #model = A2C("MultiInputPolicy",env,learning_rate=1e-4,gamma=0.95,verbose=1,)
    model = SAC("MultiInputPolicy", env, 
                gamma=0.99, 
                learning_rate=1e-4, 
                buffer_size=20000, 
                #batch_size=200, 
                #tau=0.005, 
                train_freq=(1, "episode"), 
                verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Guarda cada 5000 timesteps
        save_path="./checkpoints_1/",  # Carpeta de guardado
        name_prefix="reach")
    model.learn(total_timesteps=40000,callback=checkpoint_callback)#40000
    model.save("reach.zip")


