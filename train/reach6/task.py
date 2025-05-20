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

class TaskReach6_1(Task):
    """
    Reach based on joint controller.
    Controlls both position and orientation.
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

        self.vmax = 1 #rad/s

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
        return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(7,), dtype=float)

    def RLObservationSpace(self):
        
        return gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "rel_orient": gym.spaces.Box(-1, 1, shape=(4,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            #"force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

    def RLTerminated(self,observation):
        if self.steps_near_goal > self.steps_near_goal_min:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        goal_dist = observation["goal_dist"][0]
        angle = 2 * np.arccos(np.clip(np.abs(observation["rel_orient"][0]), -1.0, 1.0))
        vel = observation["vel"]
        
        k_dist = 69
        k_orient = 25
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max # Distance penalisation
        reward -= k_orient*angle/np.pi # orientation penalisation
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_orient+k_v+k_time)
        
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
        return np.zeros((8,))

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

        # orientation
        orient = timestep.observation["panda_tcp_pose"][3:7]
        orient_inv = orient*np.array([1, -1, -1, -1])
        orient_err = self.quat_multiply([0,0,0,1],orient_inv) # (pi,0,0) ypr, vertical

        obs = {
            "joint_pos": joint_pos,
            "rel_pos": rel_pos,
            "rel_orient": orient_err,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            #"force_mag": [force_mag],
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
        
        # orientation
        orient = timestep.observation["panda_tcp_pose"][3:7] # (x,y,z,w)
        orient_inv = orient*np.array([1, -1, -1, -1])
        orient_err = self.quat_multiply([0,0,0,1],orient_inv) # (pi,0,0) ypr, vertical

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
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
        act[7] = 0 
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
    
    #########################

    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]

class TaskReach6_2(Task):
    """
    Reach based on joint controller.
    Controlls both position and orientation.
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

        self.vmax = 1 #rad/s

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
        return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(7,), dtype=float)

    def RLObservationSpace(self):
        
        return gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "rel_orient": gym.spaces.Box(-1, 1, shape=(4,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            #"force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

    def RLTerminated(self,observation):
        if self.steps_near_goal > self.steps_near_goal_min:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        goal_dist = observation["goal_dist"][0]
        angle = 2 * np.arccos(np.clip(np.abs(observation["rel_orient"][0]), -1.0, 1.0))
        vel = observation["vel"]
        
        k_dist = 69
        k_orient = 25
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max # Distance penalisation
        reward -= k_orient*angle/np.pi # orientation penalisation
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_orient+k_v+k_time)
        
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
       return observation 

    def RLCommToObservationLearn(self,observation):
        return observation

    def RLCommToObservationTestGUI(self,observation):
        # Calculate the relative position based on the GUI position 
        self.diff = self.goal_pos - observation["pos"]
        
        observation["rel_pos"] = [self.diff]
        observation["goal_dist"] = [np.linalg.norm(self.diff)]

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
        return np.zeros((8,))

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
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
        act[7] = 0 
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
    
    #########################

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]

class TaskReach6_3(Task):
    """
    Reach based on joint controller.
    Controlls both position and orientation.
    """

    class TaskMode(Enum):
        LEARN = "learn"
        TEST = "test"
        TEST_GUI = "test_gui"	

    def __init__(self, mode: TaskMode, has_hand: bool = False):
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

        self.vmax = 1 #rad/s

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

        if has_hand == True:
            self.PandaNullAct = self.PandaNullActHand
            self.PandaCommToAction = self.PandaCommToActionHand
            print("has_hand = True")
        else:
            self.PandaNullAct = self.PandaNullActNoHand
            self.PandaCommToAction = self. PandaCommToActionNoHand
            print("has_hand = False")

        self.RLReset()

    ###########
    # rl_side #
    ###########

    # METHODS CONCERNING REINFORCEMENT LEARNING
    def RLActionSpace(self):
        return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(7,), dtype=float)

    def RLObservationSpace(self):
        
        return gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(7,), dtype=float),
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=float), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "rel_orient": gym.spaces.Box(-1, 1, shape=(4,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            #"force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

    def RLTerminated(self,observation):
        if self.steps_near_goal > self.steps_near_goal_min:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False
    
    # REWARD
    def RLReward(self,observation, terminated, truncated):
        goal_dist = observation["goal_dist"][0]
        angle = 2 * np.arccos(np.clip(np.abs(observation["rel_orient"][0]), -1.0, 1.0))
        vel = observation["vel"]
        
        k_dist = 69
        k_orient = 25
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max # Distance penalisation
        reward -= k_orient*angle/np.pi # orientation penalisation
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_orient+k_v+k_time)
        
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
       return observation 

    def RLCommToObservationLearn(self,observation):
        return observation

    def RLCommToObservationTestGUI(self,observation):
        # Calculate the relative position based on the GUI position 
        self.diff = self.goal_pos - observation["pos"]
        
        observation["rel_pos"] = [self.diff]
        observation["goal_dist"] = [np.linalg.norm(self.diff)]

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
        return np.zeros((8,))
    
    def PandaNullActHand(self):
        return np.zeros((8,))
    
    def PandaNullActNoHand(self):
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
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
        act[7] = 0 
        return act
    
    def PandaCommToActionHand(self, actrec):
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
        act[7] = 0 # Add gripper null action
        return act
    
    def PandaCommToActionNoHand(self, actrec):
        act = self.PandaNullAct()
        act[0:7] = actrec["action"]
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
    
    #########################

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]