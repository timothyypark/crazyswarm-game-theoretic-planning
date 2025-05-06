#!/usr/bin/env python
"""
This Code is only for following the center line for one drone, does not implement any racing behavior - proof of concept

"""
import numpy as np
from pycrazyswarm import Crazyswarm
import time
from drone_racing_utils.mpc_controller import mpc
from drone_racing_utils.drone_waypoints import WaypointGenerator

# try:
#     import keyboard
#     _KEYBOARD_AVAILABLE = True
# except ImportError:
#     _KEYBOARD_AVAILABLE = False

trajectory_type = "counter stadium"
LOOKAHEAD_HORIZON = 1.0   
LOOKAHEAD_DT      = 0.1    
SHIFT_THRESHOLD   = 0.1   
H = 8
DT = 0.1

waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 1.3)
tracks_pos = {"counter circle": [(4.4, 0.9), (4.1, 1.8), (4.5, 0)], 
                "counter oval": [(0, 0), (0, 0), (0, 0)], 
                "counter square": [(0, 0), (0, 0), (0, 0)], 
                "counter rectangle": [(1, 2.5), (-1.5, 2), (-1.5, -2)], 
                "counter stadium": [(-1.7, 0.4), (-1.4, 0.7), (-1.3, 1.1)]}


def vehicle_dynamics(xk, uk):
    x    = xk[0]
    y    = xk[1]
    x_dot = xk[2]
    y_dot = xk[3]
    a_x  = uk[0]
    a_y  = uk[1]

    x_next = np.array([
        x + x_dot * DT,
        y + y_dot * DT,
        x_dot + a_x * DT,
        y_dot + a_y * DT,
    ])
    return x_next


def callback(state_arr, last_i):
    """
    Inputs:
      state_arr: np.array([x, y, z, vx, vy, vz]) for drone in question
    Returns:
      list of tuples (ax, ay, x_ref, y_ref, vx_ref, vy_ref)
    """

    # global _emergency_land
    # Check for emergency land on spacebar
    # if _KEYBOARD_AVAILABLE and keyboard.is_pressed('space'):
    #     print("[Emergency] Spacebar pressed. Triggering emergency landing in callback...")
    #     _emergency_land = True
    #     return None, None, None, last_i
    

    v_factor = 1
    x, y, _, vx, vy, _ = state_arr
    xy_state = x,y,vx,vy
    if last_i == -1:
        dists = np.sqrt((waypoint_generator.raceline[:,0]-x)**2 + (waypoint_generator.raceline[:,1]-y)**2)
        closest_idx = np.argmin(dists)
    else:
        raceline_ext = np.concatenate((waypoint_generator.raceline[last_i:,:],waypoint_generator.raceline[:50,:]),axis=0)
        dists = np.sqrt((raceline_ext[:50,0]-x)**2 + (raceline_ext[:50,1]-y)**2)
        closest_idx = (np.argmin(dists) + last_i)%len(waypoint_generator.raceline)
    
    curr_idx = (closest_idx+1)%len(waypoint_generator.raceline)
    next_idx = (curr_idx+1)%len(waypoint_generator.raceline)
    next_dist = np.sqrt((waypoint_generator.raceline[next_idx,0]-waypoint_generator.raceline[curr_idx,0])**2 + (waypoint_generator.raceline[next_idx,1]-waypoint_generator.raceline[curr_idx,1])**2)
    dist_target = 0
    lookahead_factor = .3
    traj = []

    for t in np.arange(LOOKAHEAD_DT, LOOKAHEAD_HORIZON + LOOKAHEAD_DT/2,LOOKAHEAD_DT):
        dist_target += v_factor*waypoint_generator.raceline[curr_idx,2]*LOOKAHEAD_DT
        while dist_target - next_dist > 0. :
            dist_target -= next_dist
            curr_idx = next_idx
            next_idx = (next_idx+1)%len(waypoint_generator.raceline)
            next_dist = np.sqrt((waypoint_generator.raceline[next_idx,0]-waypoint_generator.raceline[curr_idx,0])**2 + (waypoint_generator.raceline[next_idx,1]-waypoint_generator.raceline[curr_idx,1])**2)

        ratio = dist_target/next_dist
        pt = (1.-ratio)*waypoint_generator.raceline[next_idx,:2] + ratio*waypoint_generator.raceline[curr_idx,:2]
        traj.append(pt) # just follow the raceline (centerline)
    traj = np.array(traj)
    print("traj", traj)
    print(xy_state, traj.shape)
    exit(0)
    ax, ay, state = mpc(np.array(xy_state),np.array(traj),lookahead_factor=lookahead_factor) #TODO: change mpc controller to return state
    return  ax, ay, state, closest_idx


def executeTrajectory(timeHelper, cfs, horizon, stopping_horizon, dt = 0.1, rate=10, z_setpoints=None):
    """
    Runs receding-horizon control:
      1. Measure actual state (Vicon)
      2. Call MPC → ax, ay and predicted next x/y/vx/vy
      3. Send predicted x/y, constant z, predicted vx/vy, zero vz, and accelerations via cmdFullState
      4. Log actual measured state
    """
    assert(len(cfs) == 1) # code went from 3 drones to 1, covering ourselves for when we fix it
    total_steps = horizon + stopping_horizon
    num_drones = len(cfs)

    # altitude-hold gains (cascade PID)
    K2_alt = np.array([9.1704, 16.8205])

    # allocate log: 6 rows per drone × total_steps columns
    X_log = np.zeros((6 * num_drones, total_steps))

    # initialize history for finite-difference velocity
    prev_pos = [np.array(cf.position()) for cf in cfs]
    prev_vel = np.array([0.13,0.22, 0.0])
    step = 0
    t0 = timeHelper.time()
    last_i = -1
    while not timeHelper.isShutdown() and step < total_steps:
        # 1) measure actual state
        measured_states = []
        for i, cf in enumerate(cfs):
            p = np.array(cf.position())
            print("p: ", p)
            v = prev_vel
            print("v: ", v)
            prev_pos[i] = p
            measured_states.append(np.hstack([p, v]))  # [x,y,z,vx,vy,vz]

        # 2) MPC: get control + predicted next x/y/vx/vy
        ax, ay, state, last_i = callback(measured_states[0], last_i)
        print("opt_x: ", state)
        x_dyn = vehicle_dynamics(state, [ax, ay])
        print("x_dyn: ", x_dyn)
        print("u", ax, ay)

        prev_vel = np.array([x_dyn[2],x_dyn[3], 0.0])
        # if _emergency_land:
        #     break

        # 3) send commands and log
        for i, cf in enumerate(cfs):
            x_new, y_new, vx_new, vy_new = x_dyn
            # constant altitude and zero vertical velocity
            z_new = z_setpoints[i]
            vz_new = 0.0

            # altitude-hold acceleration
            z_meas = measured_states[i][2]
            vz_meas = measured_states[i][5]
            az = K2_alt @ np.array([z_new - z_meas,
                                     vz_new - vz_meas])

            cf.cmdFullState(
                np.array([x_new, y_new, z_new]),
                np.array([vx_new, vy_new, vz_new]),
                np.array([ax, ay, az]),
                0.0,
                np.zeros(3,)
            )

            # log the measured state
            X_log[i*6:(i+1)*6, step] = measured_states[i]

        step += 1
        timeHelper.sleepForRate(rate)
    # stop and land
    print("Stopping and landing...")
    for cf in cfs:
        cf.notifySetpointsStop()
    for cf in cfs:
        cf.land(targetHeight=0.05, duration=1.0)
    timeHelper.sleep(2.0)

    return X_log


if __name__ == "__main__":
    horizon = 400
    stopping_horizon = 15
    rate = 10.0  # Hz

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies[:1]

    # desired constant take-off altitudes
    Zs = [0.5]

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
    # np.savetxt(
    #     'three_drone_mpc_predicted_state.csv',
    #     X_log,
    #     delimiter=',',
    #     header='x,y,z,vx,vy,vz per drone'
    # )
