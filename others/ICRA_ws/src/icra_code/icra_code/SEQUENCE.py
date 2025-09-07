#!/home/diego/tfg_ws/franka_rl/.venv/bin/python
import sys
import time
import multiprocessing as mp
from typing import List, Tuple
from math import sqrt
import numpy as np
from stable_baselines3 import SAC

from icra_code.rl_side import PandaEnv
from icra_code.CONFIG import *

##########
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray


class GoalPublisher(Node):

    def __init__(self, shared_goal_pos):
        super().__init__('goal_publisher')
        self.publisher_pos = self.create_publisher(Float32MultiArray, 'goal_pos', 10)
        self.publisher_vel = self.create_publisher(Float32MultiArray, 'goal_vel', 10)
        self.shared_goal_pos = shared_goal_pos

    def publish(self, pos, vel):
        msg_pos = Float32MultiArray()
        msg_pos.data = pos
        self.publisher_pos.publish(msg_pos)

        msg_vel = Float32MultiArray()
        msg_vel.data = list(vel)
        self.publisher_vel.publish(msg_vel)

##########

class Point():
    """
    3D-Point and time t
    """
    def __init__(self, x: float, y: float, z: float, t: float):
        self.x = x
        self.y = y
        self.z = z
        self.t = t


class Sequence():
    """
    ...
    """

    points: List[Point]

    def __init__(self):
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)

    def start(self, shared_position):
        for p in self.points:
            print("NEXT POINT")
            shared_position[0] = p.x
            shared_position[1] = p.y
            shared_position[2] = p.z
            time.sleep(p.t)


def main():
    mp.set_start_method('spawn')  
    manager = mp.Manager()

    """
    ##################################################
    # OPTION 1
    # Sequence of the experiment
    X = 0.2
    Z = 0.35 # Probar una secuencia en el plano vertical, empezando ya desde la esquina
    T = 5.0 # Ajustar el tiempo al minimo para evitar el tiempo estacionario
    L = 0.2
    sequence = Sequence()
    sequence.add_point(Point(X,   0.0, Z, 4*T)) # Initial point
    sequence.add_point(Point(X+L, L,   Z, T)) # 1
    sequence.add_point(Point(X-L, L,   Z, T)) # 2
    sequence.add_point(Point(X-L, -L,  Z, T)) # 3
    sequence.add_point(Point(X+L, -L,  Z, T)) # 4
    sequence.add_point(Point(X,   0.0, Z, T)) # Final point

    # Shared memory to store goal position and error vector
    shared_goal_pos = manager.list([X, 0.0, Z])
    ##################################################
    """
    ##################################################
    # OPTION 2
    # Sequence of the experiment
    X = 0.3
    Z = 0.4 # Probar una secuencia en el plano vertical, empezando ya desde la esquina
    T = 5.0 # Ajustar el tiempo al minimo para evitar el tiempo estacionario
    L = 0.15
    sequence = Sequence()
    sequence.add_point(Point(X, L,   Z+L, 4*T)) # 1
    sequence.add_point(Point(X, L,   Z-L, T)) # 2
    sequence.add_point(Point(X, -L,  Z-L, T)) # 3
    sequence.add_point(Point(X, -L,  Z+L, T)) # 4
    sequence.add_point(Point(X, L,   Z+L, T)) # 1

    # Shared memory to store goal position and error vector
    shared_goal_pos = manager.list([X, L,   Z+L])
    ##################################################

    # RL
    rl_process = mp.Process(target=main2, args=(shared_goal_pos, None,))
    rl_process.start()

    sequence.start(shared_goal_pos) 

    rl_process.join()

# Main function with multiprocessing and RL logic
def main2(shared_goal_pos, args=None):
    # Initialize the RL task and environment
    task = TASK(mode=MODE)  
    task.goal_pos = list(shared_goal_pos)
    env = PandaEnv(task=task)  
    model = SAC.load(MODEL, env)  

    # ROS2 SETUP
    rclpy.init(args=args)

    publisher = GoalPublisher(shared_goal_pos)

    observation, info = env.reset(seed=42)
    for _ in range(100000):
        # Update task goal position from the shared memory
        task.goal_pos = list(shared_goal_pos)

        # RL agent action prediction
        action, _state = model.predict(observation, deterministic=True)
        print(action.tolist())
        #print(action.tolist().shape)
        print(type(action.tolist()))

        # ROS2
        publisher.publish(task.goal_pos, action.tolist()[0])

        observation, reward, terminated, truncated, info = env.step(action)
    env.close()

    publisher.destroy_node()
    rclpy.shutdown()


# Launching the GUI and RL logic using multiprocessing
if __name__ == '__main__':
    main()
