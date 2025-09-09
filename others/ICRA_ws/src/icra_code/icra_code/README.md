# ICRA 2026 PAPER

Experiment using model reach4 to follow a sequence of points

* *panda_side.py* has now a logger to save the EE (end-effector) trajectory and the desired trajectory in a *.csv* file.

* *SEQUENCE.py* is based on *test_gui.py*, but instead of a UI, it enforces a sequence of points for the experiment. Works for TEST_GUI (although there is no UI) in simulation and real robot.

Both programs publish data using ROS2-Foxy, and the data can be stored in a ros bag. The messages are Float32MultiArray.

---

    python SEQUENCE.py 

    python panda_side.py --gui

    ros2 bag record -o name /goal_pos /goal_vel /real_pos /real_vel

    ros2 bag play name