# python3 deterministic_reach.py --gui

import dm_env
import numpy as np
import math
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import rewards, timestep_preprocessor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils

import threading


# GUI for pose input

import tkinter as tk

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sliders XYZ")

        self.x_slider = tk.Scale(self.root, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="X")
        self.y_slider = tk.Scale(self.root, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Y")
        self.z_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Z")

        self.yaw_slider = tk.Scale(self.root, from_=-180, to=180, resolution=0.1, orient=tk.HORIZONTAL, label="Yaw (°)")
        self.pitch_slider = tk.Scale(self.root, from_=-90, to=90, resolution=0.1, orient=tk.HORIZONTAL, label="Pitch (°)")
        self.roll_slider = tk.Scale(self.root, from_=-180, to=180, resolution=0.1, orient=tk.HORIZONTAL, label="Roll (°)")


        self.button = tk.Button(self.root, text="Actualizar", command=self.update_values)
        self.label = tk.Label(self.root, text="x: 0.00, y: 0.00, z: 0.00")

        self.x_slider.pack()
        self.y_slider.pack()
        self.z_slider.pack()
        self.yaw_slider.pack()
        self.pitch_slider.pack()
        self.roll_slider.pack()
        self.button.pack()
        self.label.pack()
        
        self.update_values()

        #self.root.mainloop()

    def update_values(self):
        self.x_val = self.x_slider.get()
        self.y_val = self.y_slider.get()
        self.z_val = self.z_slider.get()
        self.yaw = math.radians(self.yaw_slider.get())  # Convert to radians
        self.pitch = math.radians(self.pitch_slider.get())  # Convert to radians
        self.roll = math.radians(self.roll_slider.get())  # Convert to radians
        quaternion = self.euler_to_quaternion(self.yaw, self.pitch, self.roll)

        self.goal_pose = [self.x_val, self.y_val, self.z_val] + quaternion
        self.label.config(text=f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")

    def euler_to_quaternion(self, yaw, pitch, roll):
        # Compute the quaternion components from Euler angles (yaw, pitch, roll)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy

        return [q_w, q_x, q_y, q_z]
    





###############################################################################33

class Agent:
  """Reactive agent that follows a goal."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec
    self.window = None

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes end-effector velocities in direction of goal."""
    observation = timestep.observation
    v = self.window.goal_pose - observation['panda_tcp_pose']
    #v = min(np.linalg.norm(v), 0.1) * v / np.linalg.norm(v)
    q = v[3:]
    print(q)
    yaw, pitch, roll = self.quaternion_to_euler(q)

    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[0:3] = v[0:3]
    action[3:6] = [0,0,0] #[yaw,pitch,roll]
    #action[7] = 1 # gripper??

    return action

  def quaternion_to_euler(self,q):
    # Extraemos los valores del cuaternión
    w, x, y, z = q

    # Calculamos los ángulos de Euler
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x**2 + y**2))
    pitch = math.asin(2.0 * (w * y - z * x))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (y**2 + z**2))

    # Convertimos de radianes a grados (opcional)
    yaw = math.degrees(yaw)
    pitch = math.degrees(pitch)
    roll = math.degrees(roll)

    return yaw, pitch, roll
    


def goal_reward(observation: spec_utils.ObservationValue):
  return 0

############################################################################################

def main():
  ############################33333
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
    agent.window = window
    
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=1000, real_time=True)

if __name__ == '__main__':

  window = Gui()

  t2 = threading.Thread(target=main)
  t2.start()  

  window.root.mainloop()