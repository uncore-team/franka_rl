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

from rl_spin_decoupler.spindecoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint
from typing import Dict

import threading
import time

goal_pos = [0.4,0.1,0.4] # Definir aquí la posición objetivo




class Agent:
  """Reactive agent that follows a goal."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec
    self._commstoRL = AgentSide(BaseCommPoint.get_ip(),49054)

    """# Frequencies of physical control and decision making
    self._control_T = 1/20
    self._rl_T = 1/10
    if self._control_T >= self._rl_T:
      raise(ValueError("RL timestep must be > control timestep"))
    self._waitingforrlcommands = True
    #self._lastaction = None
		#self._lastactiont0 = 0.0"""

    # THREADING TO RECEIVE INFO
    self.timestep = None
    self.observation = None
    self.actrec = None
    self.action = self.null_act()
    self.thread = threading.Thread(target=self.comms_thread)
    self.timestep_ready = threading.Event()
    self.thread.start()

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    self.timestep = timestep
    self.timestep_ready.set()
    return self.action
  
  def comms_thread(self):
    self.timestep_ready.wait()
    while True:
      #self._commstobaselines.startComms()
      whattodo = self._commstoRL.readWhatToDo()
      #print(whattodo)#

      if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
        # GET OBSERVATION
        #observation = self._taskdef.PandaObsToComm(timestep.observation,self._env)
        self.observation = self.obs_to_comm(self.timestep)

        # SEND OBSERVATION AND RECEIVE ACTION
        #self.actrec = self._commstoRL.sendObsRecAct(self.observation)
        self._commstoRL.stepSendLastActDur(0)
        self._commstoRL.stepSendObs(self.observation)
        self.actrec = whattodo[1]
        #print(actrec)

        # CHANGE THE FORMAT OF THE ACTION
        #act = self._taskdef.PandaCommToAct(actrec)
        self.action = self.comm_to_act(self.actrec)
      elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
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
        self.observation = self.obs_to_comm(self.timestep)
        
        # SEND OBSERVATION
        self._commstoRL.resetSendObs(self.observation)
        
        # Null action
        #action = self._taskdef.PandaNullAct()
        self.action = self.null_act()
      elif whattodo[0] == AgentSide.WhatToDo.FINISH:
        print("Experiment finished")
        #raise RuntimeError("Experiment finished")
      else:
        raise(ValueError("Unknown indicator data"))

      #self._commstobaselines.stopComms()
    
  def obs_to_comm(self, timestep) -> Dict:
    return {"observation":timestep.observation["panda_tcp_pos"]} # Position x,y,z of end-effector
  
  def comm_to_act(self, actrec) -> np.array:
    #return actrec
    return actrec["action"]

  def null_act(self) -> np.array:
    return np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)

if __name__ == '__main__':
  # Initialize
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Environment
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env = environment.PandaEnvironment(robot_params)

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
