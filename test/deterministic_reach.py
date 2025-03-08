# python3 deterministic_reach.py --gui

import dm_env
import numpy as np
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import rewards, timestep_preprocessor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils




goal_pose = [0.4,0.1,0.4] # Definir aquí la pose objetivo




class Agent:
  """Reactive agent that follows a goal."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes end-effector velocities in direction of goal."""
    observation = timestep.observation
    #v = 0
    v = goal_pose - observation['panda_tcp_pos']
    #v = 0.1 * v / np.linalg.norm(v)
    v = min(np.linalg.norm(v), 0.1) * v / np.linalg.norm(v)
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[:3] = v
    action[6] = 1

    return action


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
      run_loop.run(env, agent, [], max_steps=1000, real_time=True)
