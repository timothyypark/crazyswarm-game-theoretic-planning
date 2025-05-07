#!/usr/bin/env python
"""
This Code is only for following the center line for one drone, does not implement any racing behavior - proof of concept

"""
import numpy as np
from pycrazyswarm import Crazyswarm
import time
from drone_racing_utils.mpc_controller import mpc
from drone_racing_utils.drone_waypoints import WaypointGenerator
import jax
import jax.numpy as jnp

# try:
#     import keyboard
#     _KEYBOARD_AVAILABLE = True
# except ImportError:
#     _KEYBOARD_AVAILABLE = False

TRACK_LEN = 12.78  # this is hardcoded for counter stadium
HALF_LEN  = TRACK_LEN / 2
trajectory_type = "counter stadium"
LOOKAHEAD_HORIZON = 1.0   
LOOKAHEAD_DT      = 0.1    
SHIFT_THRESHOLD   = 0.1   
H = 8
DT = 0.1
lat_err_thresh  = 0.6   # metres
lat_err_gain    = 2.0    # dimensionless

waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 1.3)
waypoint_generator_opp = WaypointGenerator(trajectory_type, DT, H, 1.3)
waypoint_generator_opp1 = WaypointGenerator(trajectory_type, DT, H, 1.3)

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
    
    x, y, z, x_dot, y_dot, z_dot = xk[0], xk[1], xk[2], xk[3], xk[4], xk[5]
    a_x, a_y, a_z = uk[0], uk[1], uk[2]

    x_next = np.array([
        x + x_dot * DT,
        y + y_dot * DT,
        z + z_dot * DT,
        x_dot + a_x * DT,
        y_dot + a_y * DT,
        z_dot + a_z * DT
    ])
    return x_next


def follow_line_callback(state_arr, last_i, waypoint_gen):
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
        dists = np.sqrt((waypoint_gen.raceline[:,0]-x)**2 + (waypoint_gen.raceline[:,1]-y)**2)
        closest_idx = np.argmin(dists)
    else:
        raceline_ext = np.concatenate((waypoint_gen.raceline[last_i:,:],waypoint_gen.raceline[:50,:]),axis=0)
        dists = np.sqrt((raceline_ext[:50,0]-x)**2 + (raceline_ext[:50,1]-y)**2)
        closest_idx = (np.argmin(dists) + last_i)%len(waypoint_gen.raceline)
    
    curr_idx = (closest_idx+1)%len(waypoint_gen.raceline)
    next_idx = (curr_idx+1)%len(waypoint_gen.raceline)
    next_dist = np.sqrt((waypoint_gen.raceline[next_idx,0]-waypoint_gen.raceline[curr_idx,0])**2 + (waypoint_gen.raceline[next_idx,1]-waypoint_gen.raceline[curr_idx,1])**2)
    dist_target = 0
    lookahead_factor = .3
    traj = []

    for t in np.arange(LOOKAHEAD_DT, LOOKAHEAD_HORIZON + LOOKAHEAD_DT/2,LOOKAHEAD_DT):
        dist_target += v_factor*waypoint_gen.raceline[curr_idx,2]*LOOKAHEAD_DT
        while dist_target - next_dist > 0. :
            dist_target -= next_dist
            curr_idx = next_idx
            next_idx = (next_idx+1)%len(waypoint_gen.raceline)
            next_dist = np.sqrt((waypoint_gen.raceline[next_idx,0]-waypoint_gen.raceline[curr_idx,0])**2 + (waypoint_gen.raceline[next_idx,1]-waypoint_gen.raceline[curr_idx,1])**2)

        ratio = dist_target/next_dist
        pt = (1.-ratio)*waypoint_gen.raceline[next_idx,:2] + ratio*waypoint_gen.raceline[curr_idx,:2]
        traj.append(pt) # just follow the raceline (centerline)
    traj = np.array(traj)
    print("traj", traj)
    ax, ay, state = mpc(np.array(xy_state),np.array(traj),lookahead_factor=lookahead_factor) #TODO: change mpc controller to return state
    return  ax, ay, state, closest_idx

def calc_shift(s,s_opp,vs,vs_opp,sf1=0.4,sf2=0.1,t=1.0) :
    if vs == vs_opp :
        return 0.
    ttc = (s_opp-s)+(vs_opp-vs)*t
    eff_s = ttc 
    factor = sf1*np.exp(-sf2*np.abs(eff_s)**2)
    return factor

def comp_callback(full_state, pose,pose_opp,pose_opp1,sf1,sf2,lookahead_factor,v_factor,blocking_factor,last_i, waypoint_gen):
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
    s,e,v = pose

    v_factor = 1
    x, y, _, vx, vy, _ = full_state
    xy_state = x,y,vx,vy

    s_opp,e_opp,v_opp = pose_opp
    s_opp1,e_opp1,v_opp1 = pose_opp1

    if last_i == -1:
        dists = np.sqrt((waypoint_gen.raceline[:,0]-x)**2 + (waypoint_gen.raceline[:,1]-y)**2)
        closest_idx = np.argmin(dists)
    else:
        raceline_ext = np.concatenate((waypoint_gen.raceline[last_i:,:],waypoint_gen.raceline[:50,:]),axis=0)
        dists = np.sqrt((raceline_ext[:50,0]-x)**2 + (raceline_ext[:50,1]-y)**2)
        closest_idx = (np.argmin(dists) + last_i)%len(waypoint_gen.raceline)
    
    curr_idx = (closest_idx+1)%len(waypoint_gen.raceline)
    next_idx = (curr_idx+1)%len(waypoint_gen.raceline)
    next_dist = np.sqrt((waypoint_gen.raceline[next_idx,0]-waypoint_gen.raceline[curr_idx,0])**2 + (waypoint_gen.raceline[next_idx,1]-waypoint_gen.raceline[curr_idx,1])**2)
    
    dist_target = 0
    traj = []
    for t in np.arange(LOOKAHEAD_DT,
                   LOOKAHEAD_HORIZON + LOOKAHEAD_DT/2,
                   LOOKAHEAD_DT):
        dist_target += v_factor*waypoint_gen.raceline[curr_idx,2]*LOOKAHEAD_DT
            
        shift2 = calc_shift(s,s_opp,v,v_opp,sf1,sf2,t)
        if e>e_opp :
            shift2 = np.abs(shift2)
        else :
            shift2 = -np.abs(shift2)
        shift1 = calc_shift(s,s_opp1,v,v_opp1,sf1,sf2,t)
        if e>e_opp1 :
            shift1 = np.abs(shift1)
        else :
            shift1 = -np.abs(shift1)
        shift = shift1 + shift2
        
        
        if abs(shift2) > abs(shift1) :
            if (shift+e_opp)*shift < 0. : 
                shift = 0.
            else :
                if abs(shift2) > SHIFT_THRESHOLD:
                    shift += e_opp
        else :
            if (shift+e_opp1)*shift < 0. :
                shift = 0.
            else :
                if abs(shift1) >SHIFT_THRESHOLD:
                    shift += e_opp1
    
        dist_from_opp = s-s_opp
        if dist_from_opp < -HALF_LEN:
            dist_from_opp += TRACK_LEN
        if dist_from_opp > HALF_LEN:  
            dist_from_opp -= TRACK_LEN
        
        dist_from_opp1 = s-s_opp1
        if dist_from_opp1 < -HALF_LEN:
            dist_from_opp1 += TRACK_LEN
        if dist_from_opp1 > HALF_LEN:
            dist_from_opp1 -= TRACK_LEN

        if dist_from_opp>0 and (dist_from_opp < dist_from_opp1 or dist_from_opp1 < 0) :
            bf = 1 - np.exp(-blocking_factor*max(v_opp-v,0.))
            shift = shift + (e_opp-shift)*bf*calc_shift(s,s_opp,v,v_opp,sf1,sf2,t)/sf1
        elif dist_from_opp1>0 and (dist_from_opp1 < dist_from_opp or dist_from_opp < 0) :
            bf = 1 - np.exp(-blocking_factor*max(v_opp1-v,0.))
            shift = shift + (e_opp1-shift)*bf*calc_shift(s,s_opp1,v,v_opp1,sf1,sf2,t)/sf1
        
        while dist_target - next_dist > 0. :
            dist_target -= next_dist
            curr_idx = next_idx
            next_idx = (next_idx+1)%len(waypoint_gen.raceline)
            next_dist = np.sqrt((waypoint_gen.raceline[next_idx,0]-waypoint_gen.raceline[curr_idx,0])**2 + (waypoint_gen.raceline[next_idx,1]-waypoint_gen.raceline[curr_idx,1])**2)

        ratio = dist_target/next_dist
        pt = (1.-ratio)*waypoint_gen.raceline[next_idx,:2] + ratio*waypoint_gen.raceline[curr_idx,:2]
        theta_traj = np.arctan2(waypoint_gen.raceline[next_idx,1]-waypoint_gen.raceline[curr_idx,1],waypoint_gen.raceline[next_idx,0]-waypoint_gen.raceline[curr_idx,0]) + np.pi/2.
        shifted_pt = pt + shift*np.array([np.cos(theta_traj),np.sin(theta_traj)])
        traj.append(shifted_pt)

    traj = np.array(traj)
    print("traj", traj)
    ax, ay, state = mpc(np.array(xy_state),np.array(traj),lookahead_factor=lookahead_factor)
    return  ax, ay, state, closest_idx

def executeTrajectory(timeHelper, cfs, horizon, stopping_horizon, dt = 0.1, rate=10, z_setpoints=None):
    """
    Runs receding-horizon control:
      1. Measure actual state (Vicon)
      2. Call MPC → ax, ay and predicted next x/y/vx/vy
      3. Send predicted x/y, constant z, predicted vx/vy, zero vz, and accelerations via cmdFullState
      4. Log actual measured state
    """
    total_steps = horizon + stopping_horizon
    num_drones = len(cfs)

    # altitude-hold gains (cascade PID)
    K2_alt = np.array([9.1704, 16.8205])

    # allocate log: 6 rows per drone × total_steps columns
    X_log = np.zeros((6 * num_drones, total_steps))

    # initialize history for finite-difference velocity
    prev_pos = [np.array(cf.position()) for cf in cfs]
    prev_vel = [np.array([0.13,0.22, 0.0])] * num_drones
    step = 0
    t0 = timeHelper.time()
   
    last_i = -1
    last_i_opp = -1
    last_i_opp1 = -1

    curr_speed_factor = 1.
    curr_lookahead_factor = 0.24
    curr_sf1 = 0.2
    curr_sf2 = 0.2
    blocking = 0.2
        
    curr_speed_factor_opp = 1.
    curr_lookahead_factor_opp = 0.15
    curr_sf1_opp = 0.1
    curr_sf2_opp = 0.5
    blocking_opp = 0.2
    
    curr_speed_factor_opp1 = 1.
    curr_lookahead_factor_opp1 = 0.15
    curr_sf1_opp1 = 0.1
    curr_sf2_opp1 = 0.5
    blocking_opp1 = 0.2

    while not timeHelper.isShutdown() and step < total_steps:
        # 1) measure actual state
        measured_states = []
        for i, cf in enumerate(cfs):
            p = np.array(cf.position())
            print("p: ", p)
            v = prev_vel[i]
            print("v: ", v)
            prev_pos[i] = p
            measured_states.append(np.hstack([p, v]))  # [x,y,z,vx,vy,vz]

        vx = prev_vel[0][0]
        vy = prev_vel[0][1]
        vx_opp = prev_vel[1][0]
        vy_opp = prev_vel[1][1]
        vx_opp1 = prev_vel[2][0]
        vy_opp1 = prev_vel[2][1]

        # 2) MPC: get control + predicted next x/y/vx/vy
            

        target_pos_tensor, _, s, e = waypoint_generator.generate(jnp.array(measured_states[0]),dt=DT)
        target_pos_tensor_opp, _, s_opp, e_opp = waypoint_generator_opp.generate(jnp.array(measured_states[1]),dt=DT)
        target_pos_tensor_opp1, _, s_opp1, e_opp1 = waypoint_generator_opp1.generate(jnp.array(measured_states[2]),dt=DT)
        target_pos_list = np.array(target_pos_tensor)


        v_ego = np.sqrt(vx**2 + vy**2)
        v_opp = np.sqrt(vx_opp**2 + vy_opp**2)
        v_opp1 = np.sqrt(vx_opp1**2 + vy_opp1**2)

        ax, ay, state, last_i = comp_callback(measured_states[0], (s,e,v_ego),(s_opp,e_opp,v_opp),(s_opp1,e_opp1,v_opp1), curr_sf1, curr_sf2, curr_lookahead_factor*2, curr_speed_factor**2, blocking, last_i, waypoint_generator)


        ax_opp, ay_opp, state_opp, last_i_opp = follow_line_callback(measured_states[1], last_i_opp, waypoint_generator_opp)
        ax_opp1, ay_opp1, state_opp1, last_i_opp1 = follow_line_callback(measured_states[2], last_i_opp1, waypoint_generator_opp1)

        print("opt_x: ", state)

        x_dyn = vehicle_dynamics(state, [ax, ay])
        x_dyn_opp = vehicle_dynamics(state_opp, [ax_opp, ay_opp])
        x_dyn_opp1 = vehicle_dynamics(state_opp1, [ax_opp1, ay_opp1])

        state_dynamics = [x_dyn, x_dyn_opp, x_dyn_opp1]
        accelerations = [(ax, ay), (ax_opp, ay_opp), (ax_opp1, ay_opp1)]

        if abs(e) > lat_err_thresh :
            x_dyn[2] *= np.exp(-lat_err_gain*(abs(e)-lat_err_thresh))
            x_dyn[3] *= np.exp(-lat_err_gain*(abs(e)-lat_err_thresh))
        if abs(e_opp) > lat_err_thresh:
            x_dyn_opp[2] *= np.exp(-lat_err_gain*(abs(e_opp)- lat_err_thresh))
            x_dyn_opp[3] *= np.exp(-lat_err_gain*(abs(e_opp)- lat_err_thresh))
        if abs(e_opp1) > lat_err_thresh :
            x_dyn_opp1[2] *= np.exp(-lat_err_gain*(abs(e_opp1)- lat_err_thresh))
            x_dyn_opp1[3] *= np.exp(-lat_err_gain*(abs(e_opp1)- lat_err_thresh))
       
       
        print("x_dyn: ", x_dyn)
        print("u", ax, ay)

        prev_vel = np.array([x_dyn[2],x_dyn[3], 0.0], [x_dyn_opp[2],x_dyn_opp[3], 0.0], [x_dyn_opp1[2],x_dyn_opp1[3], 0.0])
        # if _emergency_land:
        #     break
        # 3) send commands and log
        for i, cf in enumerate(cfs):
            x_new, y_new, vx_new, vy_new = state_dynamics[i]
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
                np.array([accelerations[i][0], accelerations[i][1], az]),
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
    num_cfs = 3

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies[:num_cfs+1]

    # desired constant take-off altitudes
    Zs = [0.5, .7, .9]

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
