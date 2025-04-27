from rl_side import PandaEnv
from stable_baselines3 import SAC
import threading
from math import sqrt
import numpy as np

from CONFIG import *

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QFrame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class CircleCanvas(QFrame):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self.callback = callback
        self.max_radius = 0.7
        self.center = self.width() // 2
        self.radius = self.center - 10

        self.marker_x = self.center
        self.marker_y = self.center

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw circle
        pen = QPen(QColor("black"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(self.center - self.radius, self.center - self.radius, self.radius * 2, self.radius * 2)

        # Draw marker
        painter.setBrush(QColor("red"))
        painter.drawEllipse(self.marker_x - 5, self.marker_y - 5, 10, 10)

    def mousePressEvent(self, event):
        self.update_marker(event.x(), event.y())

    def mouseMoveEvent(self, event):
        self.update_marker(event.x(), event.y())

    def update_marker(self, x, y):
        dx = x - self.center
        dy = y - self.center
        dist = sqrt(dx ** 2 + dy ** 2)

        if dist > self.radius:
            dx = dx * self.radius / dist
            dy = dy * self.radius / dist

        self.marker_x = self.center + dx
        self.marker_y = self.center + dy
        self.update()

        # Flip axes as in original code
        dx_robot = -dy
        dy_robot = -dx

        x_val = dx_robot / self.radius * self.max_radius
        y_val = dy_robot / self.radius * self.max_radius

        self.callback(x_val, y_val)

class Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Position XYZ")

        self.x_val = 0.4
        self.y_val = 0.0
        self.z_val = 0.4
        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.diff = [0, 0, 0]

        main_layout = QHBoxLayout(self)

        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)

        observation_layout = QVBoxLayout()
        main_layout.addLayout(observation_layout)

        # Circle Canvas
        self.canvas = CircleCanvas(self, self.update_from_canvas)
        control_layout.addWidget(self.canvas)

        # Z Slider
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 100)
        self.z_slider.setValue(40)
        self.z_slider.valueChanged.connect(self.update_values)
        control_layout.addWidget(self.z_slider)

        # Button
        self.button = QPushButton("Actualizar")
        self.button.clicked.connect(self.update_values)
        control_layout.addWidget(self.button)

        # Label
        self.label = QLabel("x: 0.00, y: 0.00, z: 0.00")
        control_layout.addWidget(self.label)

        # 3D Plot
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-0.3, 0.3])
        self.ax.set_ylim([-0.3, 0.3])
        self.ax.set_zlim([-0.3, 0.3])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.canvas3d = FigureCanvas(self.fig)
        observation_layout.addWidget(self.canvas3d)

        # Error Vector Label
        self.vector_mod = QLabel("ERROR DISTANCE: 0 (m)")
        observation_layout.addWidget(self.vector_mod)

        # Timer to update plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_3D_vector)
        self.timer.start(100)  # every 100 ms

    def update_from_canvas(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        self.update_label()
        self.update_3D_vector()

    def update_values(self):
        self.z_val = self.z_slider.value() / 100
        self.goal_pos = [self.x_val, self.y_val, self.z_val]
        self.update_label()

    def update_label(self):
        self.z_val = self.z_slider.value() / 100
        self.label.setText(f"x: {self.x_val:.2f}, y: {self.y_val:.2f}, z: {self.z_val:.2f}")

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

        self.vector_mod.setText(f"ERROR DISTANCE: {round(np.linalg.norm(self.diff), 3)} (m)\n \
                                X: {round(self.diff[0],3)} \n \
                                Y: {round(self.diff[1],3)} \n \
                                Z: {round(self.diff[2],3)}")

def main():
    task = TASK(mode=TASK.TaskMode.TEST_GUI)
    task.goal_pos = window.goal_pos
    env = PandaEnv(task=task)
    
    model = SAC.load(MODEL, env)

    observation, info = env.reset(seed=42)
    for _ in range(100000):
        task.goal_pos = window.goal_pos 
        window.diff = task.diff
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
    env.close()

if __name__ == '__main__':
    app = QApplication([])

    window = Gui()
    window.show()

    t2 = threading.Thread(target=main)
    t2.start()

    app.exec()
