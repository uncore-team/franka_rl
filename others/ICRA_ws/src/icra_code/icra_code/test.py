from rl_side import PandaEnv
from stable_baselines3 import SAC

from CONFIG import *

if __name__ == '__main__':
    task = TASK(mode=TASK.TaskMode.TEST)
    env = PandaEnv(task=task)
    model = SAC.load(MODEL, env)

    observation, info = env.reset(seed=42)
    for _ in range(10000):
        action, _state = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
