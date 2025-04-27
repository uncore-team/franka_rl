from rl_side import PandaEnv
from stable_baselines3 import SAC
import threading
from math import sqrt
import numpy as np

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from CONFIG import *

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Position XYZ")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.observation_frame = tk.Frame(self.main_frame)
        self.observation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


        # Circle Canvas for XY
        self.max_radius = 0.7

        self.canvas_size = 300
        self.radius = self.canvas_size // 2 - 10  # leave some margin
        self.center = self.canvas_size // 2

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(in_=self.control_frame)

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
        self.z_slider.pack(in_=self.control_frame)

        self.button = tk.Button(self.root, text="Actualizar", command=self.update_values)
        self.button.pack(in_=self.control_frame)
        
        self.label = tk.Label(self.root, text="x: 0.00, y: 0.00, z: 0.00")
        self.label.pack(in_=self.control_frame)

        # Current XY marker
        self.marker = self.canvas.create_oval(0,0,0,0,fill="red")

        # Goal Position
        self.x_val = 0.4
        self.y_val = 0.0
        self.z_val = 0.4
        self.goal_pos = [self.x_val, self.y_val, self.z_val]

        # Error vector
        self.diff = [0, 0, 0]

        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([0, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.canvas3d = FigureCanvasTkAgg(self.fig, master=self.observation_frame)
        self.canvas3d.get_tk_widget().pack()

        self.vector_mod = tk.Label(self.root, text="ERROR DISTANCE: 0 (m)")
        self.vector_mod.pack(in_=self.observation_frame)

        self.root.after(100, self.update_loop)  # 20 FPS


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
        self.update_3D_vector()

    def update_values(self):
        self.z_val = self.z_slider.get()
        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.update_label()

    def update_label(self):
        # Update Z value live too
        self.z_val = self.z_slider.get()
        self.label.config(text=f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")

    def update_loop(self):
        self.update_3D_vector()
        self.root.after(100, self.update_loop)  # call again after 50ms

    def update_3D_vector(self):
        L = 0.3
        self.ax.cla()
        self.ax.set_xlim([-L, L])
        self.ax.set_ylim([-L, L])
        self.ax.set_zlim([-L, L])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.quiver(0, 0, 0, self.diff[0], self.diff[1], self.diff[2], color='red')

        self.canvas3d.draw()
        self.vector_mod.config(text=f"ERROR DISTANCE: {round(np.linalg.norm(self.diff), 3)} (m)\n \
                                X: {self.diff[0]} \n \
                                Y: {self.diff[1]} \n \
                                Z: {self.diff[2]}")

def main():
    task = TASK(mode=TASK.TaskMode.TEST_GUI)
    # Set the goal position
    task.goal_pos = window.goal_pos
    env = PandaEnv(task=task)
    
    model = SAC.load(MODEL, env)

    observation, info = env.reset(seed=42)
    for _ in range(100000):
        task.goal_pos = window.goal_pos 
        # Get the error vector for the GUI
        window.diff = task.diff
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


    
