from rl_side import PandaEnv
from stable_baselines3 import SAC
import threading
import tkinter as tk

from task import TaskReach1, TaskReach2

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sliders XYZ")

        self.x_slider = tk.Scale(self.root, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="X")
        self.y_slider = tk.Scale(self.root, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Y")
        self.z_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Z")

        self.button = tk.Button(self.root, text="Actualizar", command=self.update_values)
        self.label = tk.Label(self.root, text="x: 0.00, y: 0.00, z: 0.00")

        self.x_slider.pack()
        self.y_slider.pack()
        self.z_slider.pack()
        self.button.pack()
        self.label.pack()
        
        #self.update_values()
        self.goal_pos = [0.5, 0, 0.5]

        #self.root.mainloop()

    def update_values(self):
        self.x_val = self.x_slider.get()
        self.y_val = self.y_slider.get()
        self.z_val = self.z_slider.get()

        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.label.config(text=f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")



def main():
    task = TaskReach2(mode=TaskReach2.TaskMode.TEST_GUI)
    #task.max_steps = 100000000
    task.goal_pos = window.goal_pos
    env = PandaEnv(task=task)
    
    model = SAC.load("reach.zip")

    observation, info = env.reset(seed=42)
    for _ in range(10000):
        task.goal_pos = window.goal_pos # nop
        #action = env.action_space.sample()
        action, _state = model.predict(observation, deterministic=True)
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation["pose"][0:3])

        # If the episode has ended then we can reset to start a new episode
        #if terminated or truncated:
        #    observation, info = env.reset()
    env.close()



if __name__ == '__main__':

    window = Gui()

    t2 = threading.Thread(target=main)
    t2.start()  

    window.root.mainloop()


    
