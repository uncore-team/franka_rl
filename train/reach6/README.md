# reach6

The goal of this model is ...

Next things to fix:
* Self collision and cortorsionism -> add joint position information.
* Orientation must be controlled by the model.

Action has seven dimensions (joints)
The goal orientation is always the same, vertical (pi,0,0) -> 

1. Try joint controller

    self.vmax = 1 #rad/s

    return gym.spaces.Box(low=-self.vmax, high=self.vmax, shape=(7,), dtype=float)

    "rel_orient": gym.spaces.Box(-1, 1, shape=(4,), dtype=float)

    def PandaNullAct(self):
        return np.zeros((8,))

    def PandaCommToAction(self, actrec):
        act = self.PandaNullAct()
        act[0:7] = actrec
        act[7] = 0 
        return act

Looks like the representation of the quaternion was incorrect (w,x,y,z), while dm_robotics_panda uses (x,y,z,w)
    
2. Quaternion --> (x,y,z,w)
Maybe, a condition for "correct orientation" should be added to termination "ifs".

. Try Curriculum learning --> First learn position, then add slowly orientation reward.