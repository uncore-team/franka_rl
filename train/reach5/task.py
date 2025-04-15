from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import gymnasium as gym
import random

class Task(ABC):
    """
    This class template contains methods that define the task done by the robot
    """

    # DEFINE CONSTANT USED BY MULTIPLE METHODS
    @abstractmethod
    def __init__(self):
        super().__init__()

    ###########
    # rl_side #
    ###########

    # METHODS CONCERNING REINFORCEMENT LEARNING
    @abstractmethod
    def RLActionSpace(self):
        pass

    @abstractmethod
    def RLObservationSpace(self):
        pass

    @abstractmethod
    def RLTerminated(self):
        pass

    @abstractmethod
    def RLTruncated(self):
        pass

    @abstractmethod
    def RLReward(self, observation, terminated, truncated):
        pass

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    @abstractmethod
    def RLActionToComm(self, action):
        pass

    @abstractmethod
    def RLCommToObservation(self,observation):
        pass


    ##############
    # panda_side #
    ##############

    # NULL ACTION FOR THE ROBOT
    @abstractmethod
    def PandaNullAct(self):
        pass

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    @abstractmethod
    def PandaObservationToComm(self, timestep, momaenv):
        pass

    @abstractmethod
    def PandaCommToAction(self):
        pass

class TaskReach5(Task):
    """
    TaskReach4 with improvements:
    Random Goal is cylindrical to avoid corners that the robot can't reach
    """

    class TaskMode(Enum):
        LEARN = "learn"
        TEST = "test"
        TEST_GUI = "test_gui"	

    def __init__(self, mode: TaskMode):
        """
        mode should be passed in panda_side
        """
        super().__init__()
        self.max_steps = 100
        self.steps_near_goal_min = 30

        self.min_dist = 0.25
        self.goal_dist_max = 3
        self.rel_pos_max = 1.5

        self.max_force = 40
        self.fmax = 100.0

        self.vmax = 0.4

        self.last_vel=[0,0,0]

        if mode == self.TaskMode.LEARN:
            self.PandaObservationToComm = self.PandaObservationToCommLearn
            self.RLCommToObservation = self.RLCommToObservationLearn
            print("Learning mode")
        if mode == self.TaskMode.TEST:
            self.PandaObservationToComm = self.PandaObservationToCommLearn
            self.RLCommToObservation = self.RLCommToObservationLearn
            print("Testing mode")
        elif mode == self.TaskMode.TEST_GUI:
            self.PandaObservationToComm = self.PandaObservationToCommTestGUI
            self.RLCommToObservation = self.RLCommToObservationTestGUI
            self.goal_pos = None
            print("Testing with GUI mode")

        self.RLReset()

    ###########
    # rl_side #
    ###########

    # METHODS CONCERNING REINFORCEMENT LEARNING
    def RLActionSpace(self):
        # CARTESIAN VELOCITY CONTROL OF THE END EFFECTOR
        return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float)

    def RLObservationSpace(self):
        
        return gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            "force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

    def RLTerminated(self,observation):
        if observation["force_mag"][0] > self.max_force:
            return True
        if self.steps_near_goal > self.steps_near_goal_min:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        force_mag = observation["force_mag"][0]
        goal_dist = observation["goal_dist"][0]
        vel = observation["vel"]

        k_dist = 89
        k_force = 5
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max
        reward -= k_force*force_mag/self.max_force
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_force+k_v+k_time)
        
        if goal_dist < self.min_dist:
            self.steps_near_goal += 1
        else:
            self.steps_near_goal = 0

        self.last_vel = vel
        return reward
    

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def RLActionToComm(self,action):
        return {"action":action}
    
    def RLCommToObservation(self,observation):
        pass

    def RLCommToObservationLearn(self,observation):
        return observation

    def RLCommToObservationTestGUI(self,observation):
        diff = self.goal_pos - observation["pos"]

        observation["rel_pos"] = [diff]
        observation["goal_dist"] = [np.linalg.norm(diff)]

        observation.pop("pos")
        return observation
    
    # RECORDER
    def RLReset(self):
        self.steps = 0
        self.steps_near_goal = 0

        self.last_vel=[0,0,0]

    def RLStep(self):
        self.steps += 1


    ##############
    # panda_side #
    ##############

    # NULL ACTION FOR THE ROBOT
    def PandaNullAct(self):
        return np.zeros((7,))

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def PandaObservationToComm(self, timestep, momaenv):
        pass
    
    def PandaObservationToCommLearn(self, timestep, momaenv):
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]

        obs = {
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            "force_mag": [force_mag],
            "vel": vel}
        return obs

    def PandaObservationToCommTestGUI(self, timestep, momaenv):
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]

        obs = {
            "pos": pos, # Add position of TCP
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            "force_mag": [force_mag],
            "vel": vel
            }
        return obs
    
    def PandaCommToAction(self, actrec):
        act = self.PandaNullAct()
        act[0:3] = actrec["action"]
        return act
    
    def PandaReset(self):
        self.goal_pos = self.PandaRandomGoalPos()

    def PandaRandomGoalPos(self):
        # BOUNDS
        r_bounds = (0.3,0.7)
        z_bounds = (0.2,0.7)

        # GET RANDOM POINT
        r = random.uniform(*r_bounds)
        z = random.uniform(*z_bounds)
        theta = random.uniform(0,2*np.pi)

        # COORDINATES TF
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        return [x,y,z]

   
class TaskReach5_2(Task):
    """
    TaskReach5 without terminating the episode due to force.
    The goal is to avoid "suicide"
    """

    class TaskMode(Enum):
        LEARN = "learn"
        TEST = "test"
        TEST_GUI = "test_gui"	

    def __init__(self, mode: TaskMode):
        """
        mode should be passed in panda_side
        """
        super().__init__()
        self.max_steps = 100
        self.steps_near_goal_min = 30

        self.min_dist = 0.25
        self.goal_dist_max = 3
        self.rel_pos_max = 1.5

        self.max_force = 40
        self.fmax = 100.0

        self.vmax = 0.4

        self.last_vel=[0,0,0]

        if mode == self.TaskMode.LEARN:
            self.PandaObservationToComm = self.PandaObservationToCommLearn
            self.RLCommToObservation = self.RLCommToObservationLearn
            print("Learning mode")
        if mode == self.TaskMode.TEST:
            self.PandaObservationToComm = self.PandaObservationToCommLearn
            self.RLCommToObservation = self.RLCommToObservationLearn
            print("Testing mode")
        elif mode == self.TaskMode.TEST_GUI:
            self.PandaObservationToComm = self.PandaObservationToCommTestGUI
            self.RLCommToObservation = self.RLCommToObservationTestGUI
            self.goal_pos = None
            print("Testing with GUI mode")

        self.RLReset()

    ###########
    # rl_side #
    ###########

    # METHODS CONCERNING REINFORCEMENT LEARNING
    def RLActionSpace(self):
        # CARTESIAN VELOCITY CONTROL OF THE END EFFECTOR
        return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float)

    def RLObservationSpace(self):
        
        return gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            "force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

    def RLTerminated(self,observation):
        #if observation["force_mag"][0] > self.max_force:
        #    return True
        if self.steps_near_goal > self.steps_near_goal_min:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        force_mag = observation["force_mag"][0]
        goal_dist = observation["goal_dist"][0]
        vel = observation["vel"]

        k_dist = 89
        k_force = 5
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max
        reward -= k_force*force_mag/self.max_force
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_force+k_v+k_time)
        
        if goal_dist < self.min_dist:
            self.steps_near_goal += 1
        else:
            self.steps_near_goal = 0

        self.last_vel = vel
        return reward
    

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def RLActionToComm(self,action):
        return {"action":action}
    
    def RLCommToObservation(self,observation):
        pass

    def RLCommToObservationLearn(self,observation):
        return observation

    def RLCommToObservationTestGUI(self,observation):
        diff = self.goal_pos - observation["pos"]

        observation["rel_pos"] = [diff]
        observation["goal_dist"] = [np.linalg.norm(diff)]

        observation.pop("pos")
        return observation
    
    # RECORDER
    def RLReset(self):
        self.steps = 0
        self.steps_near_goal = 0

        self.last_vel=[0,0,0]

    def RLStep(self):
        self.steps += 1


    ##############
    # panda_side #
    ##############

    # NULL ACTION FOR THE ROBOT
    def PandaNullAct(self):
        return np.zeros((7,))

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def PandaObservationToComm(self, timestep, momaenv):
        pass
    
    def PandaObservationToCommLearn(self, timestep, momaenv):
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]

        obs = {
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            "force_mag": [force_mag],
            "vel": vel}
        return obs

    def PandaObservationToCommTestGUI(self, timestep, momaenv):
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]

        obs = {
            "pos": pos, # Add position of TCP
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            "force_mag": [force_mag],
            "vel": vel
            }
        return obs
    
    def PandaCommToAction(self, actrec):
        act = self.PandaNullAct()
        act[0:3] = actrec["action"]
        return act
    
    def PandaReset(self):
        self.goal_pos = self.PandaRandomGoalPos()

    def PandaRandomGoalPos(self):
        # BOUNDS
        r_bounds = (0.3,0.7)
        z_bounds = (0.2,0.7)

        # GET RANDOM POINT
        r = random.uniform(*r_bounds)
        z = random.uniform(*z_bounds)
        theta = random.uniform(0,2*np.pi)

        # COORDINATES TF
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        return [x,y,z]

             