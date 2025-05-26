from dm_robotics.panda import arm_constants
from task import Task, TaskReach, TaskReach2, TaskReach4, TaskReach5, TaskReach5_2, TaskReach6_3

PORT = 49054 # 49054-65535 are free ports

TASK = TaskReach5_2
MODE = TASK.TaskMode.TEST_GUI # LEARN/TEST/TEST_GUI

CONTROLLER = arm_constants.Actuation.CARTESIAN_VELOCITY # JOINT_VELOCITY, CARTESIAN_VELOCITY

HAS_HAND = True

CONTROL_TIMESTEP = 0.05
PHYSICS_TIMESTEP = 0.002

# TEST/TEST_GUI
MODEL = "models/reach5_2.zip" # Load an existing model

# LEARN
LOG_PATH = "./logs/1" # "./logs/xxx"
CHECKPOINTS_PATH = "./checkpoints/1" # "./checkpoints/xxx"
SAVE_FREQ = 2000 # steps before saving a new checkpoint
NAME_PREFIX = "reach" # name prefix for the checkpoints
TOTAL_TIMESTEPS = 100000

MODEL_NAME = "reach.zip" # Name for the final model
