# reach2

Improved version of *reach1*. *TaskReach2* has 3 different modes:
* LEARN: *rl_side.py* is for training.
* TEST: *test.py* simulates the robot to test the model. The goal position is random.
* TEST_GUI: *test_gui.py* has a simple user interface made with *tkinter* that lets the user choose the (X,Y,Z) coordinates of the goal.

The mode must be changed in *panda_side.py* when necessary.

Reinforcement Learning parameters (action and observation spaces, reward function...) are still the same as in *reach1* and *reach*.