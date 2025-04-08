from rl_side import PandaEnv
from stable_baselines3 import SAC
import numpy as np
from task import TaskReach3

if __name__ == '__main__':
    task = TaskReach3(mode=TaskReach3.TaskMode.TEST)
    env = PandaEnv(task=task)
    #model = SAC.load("checkpoints/reach_1000_steps.zip")
    model = SAC.load("reach.zip")
    #model = SAC.load("reach.zip")

    observation, info = env.reset(seed=42)
    #print(observation)
    for _ in range(10000):
        #action = env.action_space.sample()
        #observation["goal_dist"] = np.array([observation["goal_dist"]]).reshape((1,))
        #observation["force_mag"] = np.array([observation["force_mag"]]).reshape((1,))
        action, _state = model.predict(observation, deterministic=True)
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            #dist = env.goal_pos-observation["pose"][0:3]
            #print(np.linalg.norm(dist))
            observation, info = env.reset()
            #print(env.goal_pos)
    env.close()
