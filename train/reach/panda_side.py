import sys
import os

sys.path.append(os.path.abspath("../../"))

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

#from dm_robotics.moma.

from rl_spin_decoupler.spindecoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint
from typing import Dict
from enum import Enum

import threading
import time


class Agent:

  class StepState(Enum):
    """
    States of the agent when enters its step() method
    """
    READYFORRLCOMMAND = 0	# Ready for a new RL command
    EXECUTINGLASTACTION = 1	# Executing the last action
    AFTERRESET = 2	# After a previous (immediate) reset

  def __init__(self, env) -> None:
    self.env = env
    self._spec = self.env.action_spec()
    self._random_state = np.random.RandomState(42)

    self._commstoRL = AgentSide(BaseCommPoint.get_ip(),49054)
    
    self._control_timestep = env.task.control_timestep
    self._rltimestep = 0.1
    if self._rltimestep <= self._control_timestep:
      raise(ValueError("RL timestep must be > control timestep"))

    self._stepstate = Agent.StepState.READYFORRLCOMMAND
    self._lastaction = self.null_act()#None
    self._lastactiont0 = 0.0
    self._starttimecurepisode = 0.0

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    action = self._lastaction
    curtime = self.env.physics.data.time # get agent current time

    if self._stepstate == Agent.StepState.EXECUTINGLASTACTION: 
      # --- not waiting new commands from RL, just executing last action
      if (curtime - self._lastactiont0 >= self._rltimestep): 
        # last action is finished by now
      
        observation = self.obs_to_comm(timestep) # gather observation
        self._commstoRL.stepSendObs(observation,curtime) 
        self._stepstate = Agent.StepState.READYFORRLCOMMAND

    elif self._stepstate == Agent.StepState.READYFORRLCOMMAND: 
      # --- waiting for new RL step() or reset() command from RL
      
      # read the last (pending) step()/reset() indicator 
      whattodo = self._commstoRL.readWhatToDo()

      if whattodo is not None: # otherwise, no command available yet
          
        if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:

          actrec = whattodo[1]
          action = self.comm_to_act(actrec)

          lat = curtime - self._lastactiont0
          self._lastactiont0 = curtime
          self._commstoRL.stepSendLastActDur(lat)
          self._stepstate = Agent.StepState.EXECUTINGLASTACTION 
          # from now on, we are executing that action

        elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:

          # do reset the agent scenario / episode
          finish = False
          while not finish:
            try:
              # block step somehow???
              self.env.task.initialize_episode(self.env.physics,self._random_state) # put the robot at a given pose instantaneously
              finish = True
            except ValueError as e: # may occur that the arm is initialized at a pose out of its workspace
              print("\tInitializing error: {}. Repeating...".format(str(e)))

          action = self.null_act() # null action
          self._starttimecurepisode = curtime
          self._stepstate = Agent.StepState.AFTERRESET 
          # prepare to send an observation right after this

        elif whattodo[0] == AgentSide.WhatToDo.FINISH:
        
          raise RuntimeError("Experiment finished")
          
        else:
          raise(ValueError("Unknown indicator data"))

    elif self._stepstate == Agent.StepState.AFTERRESET: 
      # --- must send the pending observation after the last reset

      observation = self.obs_to_comm(timestep) # gather observation
      self._commstoRL.resetSendObs(observation,curtime)
      self._stepstate = Agent.StepState.READYFORRLCOMMAND
        
    self._lastaction = action
    return action	 # to be executed now by Panda
  

  def obs_to_comm(self, timestep) -> Dict:
    goal_pos = self.env.physics.named.data.xpos['unnamed_model/']
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
           "goal_dist": goal_dist,
           "force_mag": force_mag}
    return obs # Position x,y,z of end-effector
  
  def comm_to_act(self, actrec) -> np.array:
    act = self.null_act()
    act[0:3] = actrec["action"]
    return act

  def null_act(self) -> np.array:
    return np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
  


class Ball(prop.Prop):
	"""Simple ball prop that consists of a MuJoco sphere geom."""

	def _build(self, *args, **kwargs):
		del args, kwargs
		mjcf_root = mjcf.RootElement()
		# Props need to contain a body called prop_root
		body = mjcf_root.worldbody.add('body', name='prop_root')
		body.add('geom',
				 type='box', # a cube to reduce rolling
				 size=[0.04, 0.04, 0.04],
				 solref=[0.01, 0.5],
				 mass=1,
				 rgba=(1, 0, 0, 1))
		super()._build('ball', mjcf_root)

def init_random(panda_env,robotname,props):
  """Randomly initializes the scenario (after creation)"""

  gripper_pose_dist = pose_distribution.UniformPoseDistribution(
    min_pose_bounds=np.array([0.5, -0.3, 0.7, .75 * np.pi, -.25 * np.pi,    -.25 * np.pi]),
    max_pose_bounds=np.array([0.2, 0.3,  0.1, 1.25 * np.pi, .25 * np.pi / 2, .25 * np.pi]))

  initialize_arm = entity_initializer.PoseInitializer(
    panda_env.robots[robotname].position_gripper,
    gripper_pose_dist.sample_pose)

  initialize_props = entity_initializer.prop_initializer.PropPlacer(
    props,
    position=distributions.Uniform(-.5, .5),
    quaternion=rotations.UniformQuaternion())

  panda_env.add_entity_initializers([initialize_arm, initialize_props])
  

if __name__ == '__main__':
  # Initialize
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Environment
  robot_params = params.RobotParams(robot_ip=args.robot_ip,gripper=False)
  panda_env = environment.PandaEnvironment(robot_params,
                                           control_timestep=0.05,
                                           physics_timestep=0.002)

  # props
  ball = Ball()
  props = [ball]
  panda_env.add_props(props)
  #goal_sensor = prop_pose_sensor.PropPoseSensor(ball, 'goal')
  #panda_env.add_extra_sensors([goal_sensor])

  init_random(panda_env,robot_params.name,props)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    #utils.full_spec(env)

    # Initialize the agent
    agent = Agent(env)
    
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=100000000, real_time=True)
