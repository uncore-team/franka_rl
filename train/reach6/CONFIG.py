from task import Task, TaskReach6_1, TaskReach6_2

PORT = 49054 # 49054-65535 are free ports

TASK = TaskReach6_2
MODE = TASK.TaskMode.TEST_GUI # LEARN/TEST/TEST_GUI

# TEST/TEST_GUI
MODEL = "reach.zip" # Load an existing model

# LEARN
LOG_PATH = "./logs/3" # "./logs/xxx"
CHECKPOINTS_PATH = "./checkpoints/3" # "./checkpoints/xxx"
SAVE_FREQ = 2000 # steps before saving a new checkpoint
NAME_PREFIX = "reach" # name prefix for the checkpoints
TOTAL_TIMESTEPS = 100000

MODEL_NAME = "reach.zip" # Name for the final model



