import sys
import os

sys.path.append(os.path.abspath("../"))

import numpy as np
import dm_env
from dm_env import specs
from dm_control import mjcf
from dm_control.composer.variation import distributions, rotations
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import rewards, timestep_preprocessor

from dm_robotics.geometry import pose_distribution

from dm_robotics.moma import entity_initializer, prop
from dm_robotics.moma.sensors import prop_pose_sensor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils
from dm_robotics.panda import arm_constants

from rl_spin_decoupler.spindecoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint
from enum import Enum

from CONFIG import *

class Agent():
  """
  Agent that controls the dm_robotics_panda robot manipulator.
  * Receives the dm_robotics_panda environment observation and sends it to PandaEnv (rl_side) 
  * Receives the action chosen by a RL model (rl_side) and gives it to the dm_robotics_panda environment.
  """
  class StepState(Enum):
    """
    States of the agent when enters its step() method
    """
    READYFORRLCOMMAND = 0	# Ready for a new RL command
    EXECUTINGLASTACTION = 1	# Executing the last action
    AFTERRESET = 2	# After a previous (immediate) reset

  def __init__(self, env, task: Task) -> None:
    self.env = env
    self._spec = self.env.action_spec()
    self._random_state = np.random.RandomState(42)

    # Definition of the RL task
    self.task = task
    # Beggining of the communication with rl_side server
    self._commstoRL = AgentSide(BaseCommPoint.get_ip(),PORT)

    # Check that the timesteps are right 
    self._control_timestep = env.task.control_timestep
    self._rltimestep = 0.1
    if self._rltimestep <= self._control_timestep:
      raise(ValueError("RL timestep must be > control timestep"))

    # initialize info
    self._stepstate = Agent.StepState.READYFORRLCOMMAND
    self._lastaction = self.task.PandaNullAct()
    self._lastactiont0 = 0.0
    self._starttimecurepisode = 0.0

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Function called repeatedly in dm_robotics_panda simulation loop
    """
    action = self._lastaction
    curtime = self.env.physics.data.time 

    if self._stepstate == Agent.StepState.EXECUTINGLASTACTION: 
      # not waiting new commands, executing last action
      if (curtime - self._lastactiont0 >= self._rltimestep): 
        # last action time's up
        observation = self.task.PandaObservationToComm(timestep,self.env) 
        self._commstoRL.stepSendObs(observation,curtime) 
        self._stepstate = Agent.StepState.READYFORRLCOMMAND
    elif self._stepstate == Agent.StepState.READYFORRLCOMMAND: 
      # waiting for new RL command 
      whattodo = self._commstoRL.readWhatToDo()
      if whattodo is not None: 
        if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
          actrec = whattodo[1]
          action = self.task.PandaCommToAction(actrec)

          lat = curtime - self._lastactiont0
          self._lastactiont0 = curtime
          self._commstoRL.stepSendLastActDur(lat)
          self._stepstate = Agent.StepState.EXECUTINGLASTACTION 
          # from now on, we are executing that action
        elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
          # reset the agent scenario 
          finish = False
          while not finish:
            try:
              self.env.task.initialize_episode(self.env.physics,self._random_state) # put the robot at a given pose instantaneously
              finish = True
            except ValueError as e: 
              print("\tInitializing error: {}. Repeating...".format(str(e)))

          action = self.task.PandaNullAct() 
          self._starttimecurepisode = curtime
          self._stepstate = Agent.StepState.AFTERRESET 
          # prepare to send an observation right after this
        elif whattodo[0] == AgentSide.WhatToDo.FINISH:
          raise RuntimeError("Experiment finished")
        else:
          raise(ValueError("Unknown indicator data"))
    elif self._stepstate == Agent.StepState.AFTERRESET: 
      # must send the pending observation after the last reset
      self.task.PandaReset()
      observation = self.task.PandaObservationToComm(timestep,self.env) 
      self._commstoRL.resetSendObs(observation,curtime)
      self._stepstate = Agent.StepState.READYFORRLCOMMAND
        
    self._lastaction = action
    return action	 

def init_random(panda_env,robotname):
  """Randomly initializes the scenario (after creation)"""
  gripper_pose_dist = pose_distribution.UniformPoseDistribution(
    min_pose_bounds=np.array([0.5, -0.3, 0.7, np.pi - np.pi/3, - np.pi/3, - np.pi/3]),
    max_pose_bounds=np.array([0.2, 0.3,  0.3, np.pi + np.pi/3, + np.pi/3, + np.pi/3]))

  initialize_arm = entity_initializer.PoseInitializer(
    panda_env.robots[robotname].position_gripper,
    gripper_pose_dist.sample_pose)

  panda_env.add_entity_initializers([initialize_arm])

if __name__ == '__main__':
  # Initialize
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Environment
  robot_params = params.RobotParams(robot_ip=args.robot_ip,has_hand=HAS_HAND, actuation=arm_constants.Actuation.JOINT_VELOCITY) # actuation=arm_constants.Actuation.JOINT_VELOCITY
  panda_env = environment.PandaEnvironment(robot_params,
                                           control_timestep=0.05,
                                           physics_timestep=0.002)

  init_random(panda_env,robot_params.name)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    #utils.full_spec(env)

    # Initialize the agent
    task = TASK(mode=MODE, has_hand=HAS_HAND)
    agent = Agent(env, task=task)
    
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=100000000, real_time=True)
