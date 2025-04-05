from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym

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


class TaskReach1(Task):

    def __init__(self):
        super().__init__()
        self.max_steps = 300
        self.min_dist = 0.2
        self.max_force = 40

        self.RLReset()

    ###########
    # rl_side #
    ###########

    # METHODS CONCERNING REINFORCEMENT LEARNING
    def RLActionSpace(self):
        # CARTESIAN VELOCITY CONTROL OF THE END EFFECTOR
        vmax = 0.5
        return gym.spaces.Box(low=-vmax, high=vmax, shape=(3,), dtype=np.float32)

    def RLObservationSpace(self):
        rel_pos_max = 1.5
        goal_dist_max = 3
        fmax = 100.0
        return gym.spaces.Dict({
            "rel_pos": gym.spaces.Box(-rel_pos_max,rel_pos_max, shape=(3,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=goal_dist_max, shape=(1,), dtype=float),
            "force_mag": gym.spaces.Box(low=0, high=fmax, shape=(1,), dtype=float)   
            })

    def RLTerminated(self,observation):
        #force = np.linalg.norm(observation["force"])
        #goal_distance = np.linalg.norm(observation["goal_pos"] - observation["pose"][0:3])
        if observation["force_mag"][0] > self.max_force:
            return True
        if observation["goal_dist"][0] < self.min_dist:
            return True
        return False

    def RLTruncated(self):
        if self.steps >= self.max_steps:
            return True
        return False

    def RLReward(self,observation, terminated, truncated):
        force_mag = observation["force_mag"][0]
        goal_dist = observation["goal_dist"][0]

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

    # TRANSLATION FOR COMMUNICATION VIA rl_spin_decoupler 
    def RLActionToComm(self,action):
        return {"action":action}
    
    # RECORDER
    def RLReset(self):
        self.steps = 0

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
        goal_pos = momaenv.physics.named.data.xpos['unnamed_model/']
        #pose = timestep.observation["panda_tcp_pose"]
        pos = timestep.observation["panda_tcp_pose"][0:3]
        #rel_pos = goal_pos - pose[0:3]
        rel_pos = goal_pos - pos
        goal_dist = np.linalg.norm(rel_pos)
        #vel = timestep.observation["panda_tcp_vel_relative"]
        force = timestep.observation["panda_force"]
        force_mag = np.linalg.norm(force)

        #obs = {"pose":pose, "force":force, "vel":vel, "goal_pos":goal_pos}
        obs = {#"pose": pose,
            #"pos": pos,
            "rel_pos": rel_pos,
            "goal_dist": [goal_dist], # Must be list for gym.spaces.Box to work wehen using model.predict
            "force_mag": [force_mag]}
        return obs # Position x,y,z of end-effector

    def PandaCommToAction(self, actrec):
        act = self.PandaNullAct()
        act[0:3] = actrec["action"]
        return act

    