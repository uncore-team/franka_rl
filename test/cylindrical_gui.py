import tkinter as tk
from math import sqrt

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sliders XYZ")

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
        self.z_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Z")
        self.z_slider.pack()

        self.button = tk.Button(self.root, text="Actualizar", command=self.update_values)
        self.button.pack()
        
        self.label = tk.Label(self.root, text="x: 0.00, y: 0.00, z: 0.00")
        self.label.pack()

        self.goal_pos = [0.0, 0.0, 0.0]

        # Current XY marker
        self.marker = self.canvas.create_oval(0,0,0,0,fill="red")

    def update_from_canvas(self, event):
        # Compute (x, y) relative to center
        dx = event.x - self.center
        dy = event.y - self.center
        dist = sqrt(dx**2 + dy**2)

        if dist > self.radius:
            # Clamp to circle edge
            dx = dx * self.radius / dist
            dy = dy * self.radius / dist

        # Update marker position
        marker_radius = 5
        self.canvas.coords(
            self.marker,
            self.center + dx - marker_radius,
            self.center + dy - marker_radius,
            self.center + dx + marker_radius,
            self.center + dy + marker_radius
        )

        # Normalize to [-1, 1]
        self.x_val = dx / self.radius
        self.y_val = dy / self.radius
        self.update_label()

    def update_values(self):
        self.z_val = self.z_slider.get()
        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.update_label()

    def update_label(self):
        # Update Z value live too
        self.z_val = self.z_slider.get()
        self.label.config(text=f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")

if __name__ == "__main__":
    gui = Gui()
    gui.root.mainloop()
