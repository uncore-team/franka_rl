# reach4

The goal of this model is to be able to stay in a fixed point. Instead of terminating the episode instantly when the end effector reaches the goal, force it to wait there until a counter gets to a threshold. Also, changes in velocity vector are penalised to avoid huge oscillations.

The red cube dissappears to avoid collision problems. Instead, the function *PandaRandomGoalPos()* is used.

Also, to speed up training, consider setting real_time=False in the last line of *panda_side.py* and running without --gui. (It looks like it has not worked properly, maybe the panda_side runs too fast and it always crashes without learning)

1. Dense reward instead of Sparse reward, continuous and differentiable reward function. Normalise the reward.

Observation space:

        gym.spaces.Dict({
            "vel": gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(3,), dtype=np.float32), 
            "rel_pos": gym.spaces.Box(-self.rel_pos_max,self.rel_pos_max, shape=(3,), dtype=float),
            "goal_dist": gym.spaces.Box(low=0, high=self.goal_dist_max, shape=(1,), dtype=float),
            "force_mag": gym.spaces.Box(low=0, high=self.fmax, shape=(1,), dtype=float),
            })

Reward function:

        k_dist = 89
        k_force = 5
        k_v = 5
        k_time = 1

        reward = 0
        reward -= k_dist*goal_dist/self.goal_dist_max
        reward -= k_force*force_mag/self.max_force
        reward -= k_v*np.linalg.norm(self.last_vel-vel)/(self.vmax*np.sqrt(3)) # Penalises changes in velocity
        reward -= k_time

        reward = reward/(k_dist+k_force+k_v+k_time)
        
        if goal_dist < self.min_dist:
            self.steps_near_goal += 1
        else:
            self.steps_near_goal = 0

        self.last_vel = vel
        return reward

Random position generator. (Maybe it should be in cylindrical coordinates to avoid xy extremes)

    def PandaRandomGoalPos(self):
        limit = 0.7
        x_bounds = (-limit,limit)
        y_bounds = (-limit,limit)
        z_bounds = (0.2,limit)
        R = 0.3 # Safety radius for the robot
        while True:
            x = random.uniform(*x_bounds)
            y = random.uniform(*y_bounds)
            z = random.uniform(*z_bounds)
            pos = [x,y,z]
            if x**2 + y**2 >= R**2: # Check the point is not too close to the robot
                return pos

Next things to fix:
* Self collision and cortorsionism -> add joint position information.
* Orientation must be controlled by the model.
* The random position generator should be cylindrical, not a squared region