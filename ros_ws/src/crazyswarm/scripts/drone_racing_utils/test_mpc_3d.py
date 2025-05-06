# test_mpc_3d.py
import numpy as np
from mpc_controller_3D import mpc, N

def test_mpc_shapes_and_sanity():
    # 1) Zero‐state, zero trajectory → should return (0,0,0) accel
    x0 = np.zeros(6)                 # [x,y,z,vx,vy,vz] = 0
    traj = np.zeros((3, N))          # N steps of zero-position targets
    a_x, a_y, a_z, x_pred = mpc(x0, traj)

    # Shapes
    assert isinstance(a_x, float)
    assert isinstance(a_y, float)
    assert isinstance(a_z, float)
    assert isinstance(x_pred, np.ndarray) and x_pred.shape == (6,)

    # Sanity: with zero‐target and zero‐initial‐velocity,
    # the solver can satisfy the horizon with zero accel,
    # so expect very small accelerations.
    tol = 1e-6
    assert abs(a_x) < tol and abs(a_y) < tol and abs(a_z) < tol

    print("✓ Basic shape & zero‐traj sanity passed.")

if __name__ == "__main__":
    test_mpc_shapes_and_sanity()
