from task import Task, TaskReach6_1, TaskReach6_2, TaskReach6_3

PORT = 49054 # 49054-65535 are free ports

TASK = TaskReach6_3
MODE = TASK.TaskMode.TEST_GUI # LEARN/TEST/TEST_GUI

HAS_HAND = True

# TEST/TEST_GUI
MODEL = "reach.zip" # Load an existing model

# LEARN
LOG_PATH = "./logs/nuevo" # "./logs/xxx"
CHECKPOINTS_PATH = "./checkpoints/nuevo" # "./checkpoints/xxx"
SAVE_FREQ = 2000 # steps before saving a new checkpoint
NAME_PREFIX = "reach" # name prefix for the checkpoints
TOTAL_TIMESTEPS = 100000

MODEL_NAME = "reach6_nuevo.zip" # Name for the final model



