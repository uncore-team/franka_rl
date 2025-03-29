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

import threading
import time


class Agent:

  def __init__(self, env) -> None:
    self.env = env
    self._spec = self.env.action_spec()
    self._random_state = np.random.RandomState(42)

    self.timestep = None
    self.observation = None
    self.actrec = None
    self.action = self.null_act()

    self.reset = False

    self._commstoRL = AgentSide(BaseCommPoint.get_ip(),49054)
    # THREADING TO RECEIVE INFO
    self.thread = threading.Thread(target=self.comms_thread)
    self.timestep_ready = threading.Event()
    #self.lock = threading.Lock()
    self.thread.start()

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    #with self.lock:
    self.timestep = timestep
    self.timestep_ready.set()
    if self.reset:
      self.reset=False
      finish = False
      while not finish:
        try:
          # block step somehow???
          self.env.task.initialize_episode(self.env.physics,self._random_state) # put the robot at a given pose instantaneously
          finish = True
        except ValueError as e: # may occur that the arm is initialized at a pose out of its workspace
          print("\tInitializing error: {}. Repeating...".format(str(e)))
      
    return self.action
  
  def comms_thread(self):
    self.timestep_ready.wait()
    while True:
      whattodo = self._commstoRL.readWhatToDo()
      #print(whattodo)

      if whattodo is not None:
        if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
          self.actrec = whattodo[1]
          #print(actrec)

          #with self.lock:
          # CHANGE THE FORMAT OF THE ACTION
          self.action = self.comm_to_act(self.actrec)
          # GET OBSERVATION
          self.observation = self.obs_to_comm(self.timestep)

          # SEND OBSERVATION AND RECEIVE ACTION
          self._commstoRL.stepSendLastActDur(0)
          self._commstoRL.stepSendObs(self.observation)
          
        elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
          # do reset the physics
          """finish = False
          while not finish:
            try:
              # block step somehow???
              self.env.task.initialize_episode(self.env.physics,self._random_state) # put the robot at a given pose instantaneously
              finish = True
            except ValueError as e: # may occur that the arm is initialized at a pose out of its workspace
              print("\tInitializing error: {}. Repeating...".format(str(e)))
          """
          self.reset = True
          
          #with self.lock:
          # Null action
          self.action = self.null_act()
          # GET OBSERVATION
          self.observation = self.obs_to_comm(self.timestep)

          print("Episode finished")
          # SEND OBSERVATION
          self._commstoRL.resetSendObs(self.observation)
        elif whattodo[0] == AgentSide.WhatToDo.FINISH:
          #print("Experiment finished")
          raise RuntimeError("Experiment finished")
        else:
          raise(ValueError("Unknown indicator data"))

    
  def obs_to_comm(self, timestep) -> Dict:
    pose = timestep.observation["panda_tcp_pose"]
    vel = timestep.observation["panda_tcp_vel_relative"]
    force = timestep.observation["panda_force"]
    #goal_pos = self.env.physics.named.data.xpos['unnamed_model/']

    return {"pose":pose, "force":force, "vel":vel, "goal_pos":[0,0,0]} # Position x,y,z of end-effector
  
  def comm_to_act(self, actrec) -> np.array:
    #return actrec
    return actrec["action"]

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
    max_pose_bounds=np.array([0.1, 0.3,  0.1, 1.25 * np.pi, .25 * np.pi / 2, .25 * np.pi]))

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
      run_loop.run(env, agent, [], max_steps=100000, real_time=True)
