# reach5

The goal of this model is ...

Next things to fix:
* Self collision and cortorsionism -> add joint position information.
* Orientation must be controlled by the model.
* The random position generator should be cylindrical, not a squared region

1. The 7 dimensions of the joint position are passed in the observation, hoping that it will fix contorsionism without being too slow in training.

It looks like it has learnt to "commit suicide" sometimes. Maybe the goal is too far and the accumulated penalty is smaller when ending prematurely? Maybe it should not be truncated when excessive force is applied, and emergency stop should be implemented only for the real robot.

2. Don't terminate the episode due to force 


3. Try joint controller???