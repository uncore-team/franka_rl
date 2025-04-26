from rl_side import PandaEnv
from stable_baselines3 import SAC
import threading
import tkinter as tk
from math import sqrt

from CONFIG import *

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Position XYZ")

        self.max_radius = 0.7

        # Circle Canvas for XY
        self.canvas_size = 300
        self.radius = self.canvas_size // 2 - 10  # leave some margin
        self.center = self.canvas_size // 2

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        # Draw circle
        self.canvas.create_oval(
            self.center - self.radius, self.center - self.radius,
            self.center + self.radius, self.center + self.radius,
            outline="black"
        )

        self.canvas.bind("<B1-Motion>", self.update_from_canvas)  # drag
        self.canvas.bind("<Button-1>", self.update_from_canvas)   # click

        # Z slider
        self.z_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Z", length=300)
        self.z_slider.pack()

        self.button = tk.Button(self.root, text="Actualizar", command=self.update_values)
        self.button.pack()
        
        self.label = tk.Label(self.root, text="x: 0.00, y: 0.00, z: 0.00")
        self.label.pack()

        self.x_val = 0.4
        self.y_val = 0.0
        self.z_val = 0.4
        self.goal_pos = [self.x_val, self.y_val, self.z_val]

        # Current XY marker
        self.marker = self.canvas.create_oval(0,0,0,0,fill="red")

    def update_from_canvas(self, event):
        # Compute (x, y) relative to center
        dx_canvas = event.x - self.center
        dy_canvas = event.y - self.center
        dist = sqrt(dx_canvas**2 + dy_canvas**2)
        

        if dist > self.radius:
            # Clamp to circle edge
            dx_canvas = dx_canvas * self.radius / dist
            dy_canvas = dy_canvas * self.radius / dist

        # Update marker position
        marker_radius = 5
        self.canvas.coords(
            self.marker,
            self.center + dx_canvas - marker_radius,
            self.center + dy_canvas - marker_radius,
            self.center + dx_canvas + marker_radius,
            self.center + dy_canvas + marker_radius
        )


        # Take into account the axes of the canvas and the robot
        dx = -dy_canvas
        dy = -dx_canvas
        
        # Normalize to [-self.max_radius, self.max_radius]
        self.x_val = dx / self.radius * self.max_radius
        self.y_val = dy / self.radius * self.max_radius
        self.update_label()

    def update_values(self):
        self.z_val = self.z_slider.get()
        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.update_label()

    def update_label(self):
        # Update Z value live too
        self.z_val = self.z_slider.get()
        self.label.config(text=f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")


def main():
    task = TASK(mode=TASK.TaskMode.TEST_GUI)
    task.goal_pos = window.goal_pos
    env = PandaEnv(task=task)
    
    model = SAC.load(MODEL, env)

    observation, info = env.reset(seed=42)
    for _ in range(100000):
        task.goal_pos = window.goal_pos 
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


    
