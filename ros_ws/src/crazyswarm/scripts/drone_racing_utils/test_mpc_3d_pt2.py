# test_mpc_track_line.py
import numpy as np
from mpc_controller_3D import mpc, N, dt

# drive along +x at 0.5 m/s, flat in y,z
x0 = np.array([0,0,0,    # position
               0,0,0])  # velocity

# Build a straight‐line reference for N steps: step by v_ref*dt each time
v_ref = 0.5
traj = np.vstack([
    np.linspace(v_ref*dt, v_ref*dt*N, N),     # x positions ramp
    np.zeros(N),                              # y targets
    np.zeros(N),                              # z targets
])

a_x, a_y, a_z, x_pred = mpc(x0, traj)

print("First‐step accel:", a_x, a_y, a_z)
print("Predicted state at t=0:", x_pred)
