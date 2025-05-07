#!/usr/bin/env python
"""
This Code is only for following the center line for one drone, does not implement any racing behavior - proof of concept

"""
import numpy as np
from pycrazyswarm import Crazyswarm
import time
from drone_racing_utils.mpc_controller_3D import mpc_3D
from drone_racing_utils.drone_waypoints_3D import WaypointGenerator

# try:
#     import keyboard
#     _KEYBOARD_AVAILABLE = True
# except ImportError:
#     _KEYBOARD_AVAILABLE = False
N = 10
trajectory_type = "counter stadium"
# Zs = [0.5]

LOOKAHEAD_DT      = 0.1    
SHIFT_THRESHOLD   = 0.1   
# H = 20
DT = 0.1
# LOOKAHEAD_HORIZON = 3.0
H = N
LOOKAHEAD_HORIZON = N * DT #1.0   


waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 1.3)
tracks_pos = {"counter circle": [(4.4, 0.9), (4.1, 1.8), (4.5, 0)], 
                "counter oval": [(0, 0), (0, 0), (0, 0)], 
                "counter square": [(0, 0), (0, 0), (0, 0)], 
                "counter rectangle": [(1, 2.5), (-1.5, 2), (-1.5, -2)], 
                "counter stadium": [(-1.7, 0.4), (-1.4, 0.7), (-1.3, 1.1)]}


def vehicle_dynamics_3d(xk, uk):
    x, y, z, vx, vy, vz = xk
    ax, ay, az = uk

    x_next = np.array([
        x + vx * DT,
        y + vy * DT,
        z + vz * DT,
        vx + ax * DT,
        vy + ay * DT,
        vz + az * DT
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
    print("state_arr is ", state_arr)
    x, y, z, vx, vy, vz = state_arr
    xyz_state = x,y,z,vx,vy,vz
    if last_i == -1:
        dists = np.sqrt((waypoint_generator.raceline[:,0]-x)**2 + (waypoint_generator.raceline[:,1]-y)**2 + (waypoint_generator.raceline[:,2]-z)**2)
        closest_idx = np.argmin(dists)
    else:
        raceline_ext = np.concatenate((waypoint_generator.raceline[last_i:,:],waypoint_generator.raceline[:50,:]),axis=0)
        dists = np.sqrt((raceline_ext[:50,0]-x)**2 + (raceline_ext[:50,1]-y)**2 + (raceline_ext[:50,2]-z)**2)
        closest_idx = (np.argmin(dists) + last_i)%len(waypoint_generator.raceline)
    
    curr_idx = (closest_idx+1)%len(waypoint_generator.raceline)
    next_idx = (curr_idx+1)%len(waypoint_generator.raceline)
    next_dist = np.sqrt((waypoint_generator.raceline[next_idx,0]-waypoint_generator.raceline[curr_idx,0])**2 + (waypoint_generator.raceline[next_idx,1]-waypoint_generator.raceline[curr_idx,1])**2 + (waypoint_generator.raceline[next_idx,2]-waypoint_generator.raceline[curr_idx,2])**2)
    dist_target = 0
    lookahead_factor = .3
    traj = []

    for t in np.arange(LOOKAHEAD_DT, LOOKAHEAD_HORIZON + LOOKAHEAD_DT/2,LOOKAHEAD_DT):
        dist_target += v_factor*waypoint_generator.raceline[curr_idx,2]*LOOKAHEAD_DT
        while dist_target - next_dist > 0. :
            dist_target -= next_dist
            curr_idx = next_idx
            next_idx = (next_idx+1)%len(waypoint_generator.raceline)
            next_dist = np.sqrt((waypoint_generator.raceline[next_idx,0]-waypoint_generator.raceline[curr_idx,0])**2 + (waypoint_generator.raceline[next_idx,1]-waypoint_generator.raceline[curr_idx,1])**2 + (waypoint_generator.raceline[next_idx,2]-waypoint_generator.raceline[curr_idx,2])**2)

        ratio = dist_target/next_dist
        pt = (1.-ratio)*waypoint_generator.raceline[next_idx,:3] + ratio*waypoint_generator.raceline[curr_idx,:3]
        traj.append(pt[:3]) # just follow the raceline (centerline)
    traj = np.array(traj)
    print("traj", traj)
    print(xyz_state,traj.shape)
    # exit(0)
    ax, ay, az, state = mpc_3D(np.array(xyz_state),np.array(traj),lookahead_factor=lookahead_factor) #TODO: change mpc controller to return state
    return  ax, ay, az, state, closest_idx



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
    prev_vel = np.array([0.13,0.22, 0.1]) #changed z component to nonzero
    step = 0
    t0 = timeHelper.time()
    last_i = -1
    while not timeHelper.isShutdown() and step < total_steps:
        # 1) measure actual state
        measured_states = []
        for i, cf in enumerate(cfs):
            p = np.array(cf.position())
            # print("p: ", p)
            v = prev_vel
            # print("v: ", v)
            prev_pos[i] = p
            measured_states.append(np.hstack([p, v]))  # [x,y,z,vx,vy,vz]
        print("ms:", measured_states)
        # 2) MPC: get control + predicted next x/y/z/vx/vy/vz
      
        ax, ay, az, state, last_i = callback(measured_states[0], last_i)
        print(ax,ay, az)
        # exit(0)
        # az = 0.
        print("predicted state:", state)
        state_pred6 = [state[0],state[1],state[2],state[3], state[4], state[5]]
        x_dyn = vehicle_dynamics_3d(state_pred6, [ax, ay, az])

        prev_vel = np.array([x_dyn[3],x_dyn[4], x_dyn[5]])
        # if _emergency_land:
        #     break

        # 3) send commands and log
        for i, cf in enumerate(cfs):
            x_new, y_new, z_new, vx_new, vy_new, vz_new = x_dyn
            print("ax", ax, "ay", ay, "az", az)
            # if step == 1 :
            #     exit(0)
            accel3d = np.array([ax, ay, az])
            cf.cmdFullState(
                np.array([x_new, y_new, z_new]),
                np.array([vx_new, vy_new, vz_new]),
                accel3d,
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
    stopping_horizon = 30
    rate = 10.0  # Hz

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies[:1]

    # desired constant take-off altitudes -- modify for constant z
    Zs = [0.5]

    # take off all three
    for cf, Z in zip(cfs, Zs):
        cf.takeoff(targetHeight=Z, duration=2.0) #change the takeoff height here!
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
