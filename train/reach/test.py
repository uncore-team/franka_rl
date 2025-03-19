from baselines_side import PandaEnv
from stable_baselines3 import A2C

if __name__ == '__main__':
    env = PandaEnv()
    model = A2C.load("reach.zip")

    observation, info = env.reset(seed=42)
    #print(observation)
    for _ in range(10000):
        #action = env.action_space.sample()
        action, _state = model.predict(observation, deterministic=True)
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
