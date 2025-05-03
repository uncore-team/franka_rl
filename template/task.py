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
    def RLTerminated(self) -> bool:
        pass

    @abstractmethod
    def RLTruncated(self) -> bool:
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

    @abstractmethod
    def PandaReset(self):
        pass

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    @abstractmethod
    def PandaObservationToComm(self, timestep, momaenv):
        pass

    @abstractmethod
    def PandaCommToAction(self):
        pass

class TaskReach(Task):
    """
    Implementation of a Task
    """

    class TaskMode(Enum):
        LEARN = "learn"
        TEST = "test"
        TEST_GUI = "test_gui"	

    def __init__(self, mode: TaskMode):
        super().__init__()
        # DEFINE PARAMETERS HERE
        # ...
        # self.vmax = 0.4

        # SELECT THE TASKMODE (chosen by the user in CONFIG.py)
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
        """
        Action space of the robot. Defined using gym.spaces.
        """
        pass
        #return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(7,), dtype=float)

    def RLObservationSpace(self):
        """
        Observation space of the robot. Defined using gym.spaces.
        """
        pass
        # return gym.spaces.Dict({
        #     "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
        #     "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
        #     "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
        #     "rel_orient": gym.spaces.Box(-1, 1, shape=(4,), dtype=float),
        #     "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
        #     })

    def RLTerminated(self,observation):
        """
        Checks if the episode is terminated.
        """
        pass
        #if self.steps_near_goal > self.steps_near_goal_min:
        #    return True
        #return False

    def RLTruncated(self):
        """
        Cheks if the episode is truncated
        """
       # if self.steps >= self.max_steps:
       #     return True
       # return False
        pass
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        """
        Returns a scalar reward for the agent.
        """
        pass

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def RLActionToComm(self,action):
        """
        Wraps the action in a dictionary to send it via rl_spin_decoupler
        """
        return {"action":action}
    
    def RLCommToObservation(self,observation):
        """
        Modifies the observation received (if needed)
        """
        return observation 

    def RLCommToObservationLearn(self,observation):
        """
        Version of RLCommToObservation used when mode is LEARN
        """
        return observation

    def RLCommToObservationTestGUI(self,observation):
        """
        Version of RLCommToObservation used when mode is TEST_GUI
        It adds the position of the end effector in the observation to calculate and plot the error vector.
        """
        # Calculate the relative position based on the GUI position 
        self.diff = self.goal_pos - observation["pos"]
        
        observation["rel_pos"] = [self.diff]
        observation["goal_dist"] = [np.linalg.norm(self.diff)]

        observation.pop("pos")
        return observation
    
    # RECORDER
    def RLReset(self):
        """
        Method called at the start of each episode.
        """
        #self.steps = 0
        #self.steps_near_goal = 0

        #self.last_vel=[0,0,0]
        pass
    def RLStep(self):
        """
        Counts the steps of the episode.
        """
        self.steps += 1


    ##############
    # panda_side #
    ##############

    # NULL ACTION FOR THE ROBOT
    def PandaNullAct(self):
        """
        Returns an action composed of 0
        """
        return np.zeros((8,))

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def PandaObservationToComm(self, timestep, momaenv):
        """
        Reshapes the dm_robotics_panda timestep into the observation required by RLSide.
        """
        pass
    
    def PandaObservationToCommLearn(self, timestep, momaenv):
        """
        Version of PandaObservationToComm for LEARN mode
        """
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]

        # orientation
        orient = timestep.observation["panda_tcp_pose"][3:7] # (x,y,z,w)
        orient_inv = orient*np.array([-1, -1, -1, 1])
        orient_err = self.quat_multiply([0,0,1,0],orient_inv) # (pi,0,0) ypr, vertical

        obs = {
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "rel_orient": orient_err,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            #"force_mag": [force_mag],
            "vel": vel}
        return obs

    def PandaObservationToCommTestGUI(self, timestep, momaenv):
        """
        Version of PandaObservationToComm for TEST_GUI mode. Adds the position of the end effector.
        """
        goal_pos = self.goal_pos
        pos = timestep.observation["panda_tcp_pose"][0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)
        vel = timestep.observation["panda_tcp_vel_world"][0:3]
        joint_pos = timestep.observation["panda_joint_pos"]
        
        # orientation
        orient = timestep.observation["panda_tcp_pose"][3:7] # (x,y,z,w)
        orient_inv = orient*np.array([-1, -1, -1, 1])
        orient_err = self.quat_multiply([0,0,1,0],orient_inv) # (pi,0,0) ypr, vertical

        obs = {
            "pos": pos, # Add position of TCP
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "rel_orient": orient_err,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            #"force_mag": [force_mag],
            "vel": vel
            }
        return obs
    
    def PandaCommToAction(self, actrec):
        """
        Reshapes the action received from de RL model into the format required by the dm_robotics_panda controller.
        """
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
        act[7] = 0 
        return act
    
    def PandaReset(self):
        """
        Called when a new episode begins.
        """
        self.goal_pos = self.PandaRandomGoalPos()

    def PandaRandomGoalPos(self):
        """
        Returns a random (X,Y,Z) position in a cylinder centered in the robot base.
        """
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
    
    #########################

    def quat_multiply(self, q1, q2):
        """
        Multiplies two quaternions. Useful for orientation.
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]

                 