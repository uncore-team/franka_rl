# python3 panda_side.py --gui
import sys
import os

sys.path.append(os.path.abspath("../../"))

import dm_env
import numpy as np
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import rewards, timestep_preprocessor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils

from rl_spin_decoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint
from typing import Dict

goal_pose = [0.4,0.1,0.4] # Definir aquí la pose objetivo




class Agent:
  """Reactive agent that follows a goal."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec
    self._commstobaselines = AgentSide(BaseCommPoint.get_ip(),49054)

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    
    self._commstobaselines.startComms()
    whattodo = self._commstobaselines.readWhatToDo()

    if whattodo == AgentSide.WhatToDo.SEND_OBS_REC_ACTION:
      # GET OBSERVATION
      #observation = self._taskdef.PandaObsToComm(timestep.observation,self._env)
      observation = self.obs_to_comm(timestep)

      # SEND OBSERVATION AND RECEIVE ACTION
      actrec = self._commstobaselines.sendObsRecAct(observation)

      # CHANGE THE FORMAT OF THE ACTION
      #act = self._taskdef.PandaCommToAct(actrec)
      action = self.comm_to_act(actrec)
    elif whattodo == AgentSide.WhatToDo.RESET_SEND_OBS:
      # do reset the physics
      finish = False
      while not finish:
        try:
          # put the robot at a given pose instantaneously
          finish = True
        except ValueError as e: # may occur that the arm is initialized at a pose out of its workspace
          print("\tInitializing error: {}. Repeating...".format(str(e)))
      # GET OBSERVATION
      #observation = self._taskdef.PandaObsToComm(timestep.observation,self._env) 
      observation = self.obs_to_comm(timestep)
      
      # SEND OBSERVATION
      self._commstobaselines.resetSendObs(observation)
      
      # Null action
      #action = self._taskdef.PandaNullAct()
      action = self.null_act()
    elif whattodo == AgentSide.WhatToDo.FINISH:
      raise RuntimeError("Experiment finished")
    else:
      raise(ValueError("Unknown indicator data"))

    self._commstobaselines.stopComms()
    return action
  
  def obs_to_comm(self, timestep) -> Dict:
    return {"observation":timestep.observation["panda_tcp_pos"]} # Position x,y,z of end-effector
    #return {}
  
  def comm_to_act(self, actrec) -> np.array:
    return actrec["action"]
    #return np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)

  def null_act(self) -> np.array:
    return np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)


def goal_reward(observation: spec_utils.ObservationValue):
  """Computes a normalized reward based on distance between end-effector and goal."""
  goal_distance = np.linalg.norm(goal_pose -
                                 observation['panda_tcp_pos'])
  return np.clip(1.0 - goal_distance, 0, 1)


if __name__ == '__main__':
  # Initialize
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Environment
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env = environment.PandaEnvironment(robot_params)

  # Reward
  reward = rewards.ComputeReward(goal_reward)
  panda_env.add_timestep_preprocessors([reward])


  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    #utils.full_spec(env)

    # Initialize the agent
    agent = Agent(env.action_spec())
    
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=100000, real_time=True)
