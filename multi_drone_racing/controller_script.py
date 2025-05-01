#!/usr/bin/env python

import numpy as np
from pycrazyswarm import Crazyswarm
import time

# 1) Black-boxed MPC callback stub returning controls and predicted next x/y/vx/vy
def mpc_callback(measured_states):
    """
    Inputs:
      measured_states: list of np.array([x, y, z, vx, vy, vz]) for each drone
    Returns:
      list of tuples (ax, ay, x_ref, y_ref, vx_ref, vy_ref)
    """
    outputs = []
    for state in measured_states:
        # TODO: replace with your MPC solver output
        ax, ay = 0.0, 0.0
        x, y, z, vx, vy, vz = state
        # Predicted next-step in x/y and velocities (stub: hold current)
        x_ref, y_ref = x, y
        vx_ref, vy_ref = vx, vy
        outputs.append((ax, ay, x_ref, y_ref, vx_ref, vy_ref))
    return outputs


def executeTrajectory(timeHelper, cfs, horizon, stopping_horizon, rate=100, z_setpoints=None):
    """
    Runs receding-horizon control:
      1. Measure actual state (Vicon)
      2. Call MPC → ax, ay and predicted next x/y/vx/vy
      3. Send predicted x/y, constant z, predicted vx/vy, zero vz, and accelerations via cmdFullState
      4. Log actual measured state
    """
    dt = 1.0 / rate
    total_steps = horizon + stopping_horizon
    num_drones = len(cfs)

    # altitude-hold gains (cascade PID)
    K2_alt = np.array([9.1704, 16.8205])

    # allocate log: 6 rows per drone × total_steps columns
    X_log = np.zeros((6 * num_drones, total_steps))

    # initialize history for finite-difference velocity
    prev_pos = [np.array(cf.position()) for cf in cfs]

    step = 0
    t0 = timeHelper.time()

    while not timeHelper.isShutdown() and step < total_steps:
        # 1) measure actual state
        measured_states = []
        for i, cf in enumerate(cfs):
            p = np.array(cf.position())
            v = (p - prev_pos[i]) / dt
            prev_pos[i] = p
            measured_states.append(np.hstack([p, v]))  # [x,y,z,vx,vy,vz]

        # 2) MPC: get control + predicted next x/y/vx/vy
        u_and_x = mpc_callback(measured_states)

        # 3) send commands and log
        for i, cf in enumerate(cfs):
            ax, ay, x_ref, y_ref, vx_ref, vy_ref = u_and_x[i]

            # constant altitude and zero vertical velocity
            z_ref = z_setpoints[i]
            vz_ref = 0.0

            # altitude-hold acceleration
            z_meas = measured_states[i][2]
            vz_meas = measured_states[i][5]
            az = K2_alt @ np.array([z_ref - z_meas,
                                     vz_ref - vz_meas])

            cf.cmdFullState(
                np.array([x_ref, y_ref, z_ref]),
                np.array([vx_ref, vy_ref, vz_ref]),
                np.array([ax, ay, az]),
                0.0,
                np.zeros(3)
            )

            # log the measured state
            X_log[i*6:(i+1)*6, step] = measured_states[i]

        step += 1
        timeHelper.sleepForRate(rate)

    return X_log


if __name__ == "__main__":
    horizon = 50
    stopping_horizon = 15
    rate = 10.0  # Hz

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies[:3]  # first three drones

    # desired constant take-off altitudes
    Zs = [0.5, 0.5, 0.5]

    # take off all three
    for cf, Z in zip(cfs, Zs):
        cf.takeoff(targetHeight=Z, duration=2.0)
    timeHelper.sleep(2.0)

    # run the MPC-driven trajectory
    X_log = executeTrajectory(
        timeHelper, cfs,
        horizon, stopping_horizon,
        rate=rate,
        z_setpoints=Zs
    )

    # stop setpoints and land
    for cf in cfs:
        cf.notifySetpointsStop()
    for cf in cfs:
        cf.land(targetHeight=0.05, duration=2.0)
    timeHelper.sleep(2.0)

    # save measured states
    np.savetxt(
        'three_drone_mpc_predicted_state.csv',
        X_log,
        delimiter=',',
        header='x,y,z,vx,vy,vz per drone'
    )
