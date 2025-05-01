#!/usr/bin/env python

import numpy as np

from pycrazyswarm import *
import uav_trajectory
import time
import os
import csv
import socket
import pickle
vel = None

real_experiment = False
real_position = True

def write_csv(data, file_path='traj_data.csv'):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.flush()  # Ensure data is written to disk


def executeTrajectory(timeHelper, cf1, cf2, horizon, stopping_horizon, rate=100, offset1=np.zeros(3), offset2=np.zeros(3)):
    K1 = np.array([3.1127])
    K2 = np.array([ 9.1704,   16.8205])
    total_horizon = horizon + stopping_horizon
    X_list[:,0] = np.array([
        cf1.initialPosition[0] + offset1[0],
        cf1.initialPosition[1] + offset1[1],
        cf1.initialPosition[2] + offset1[2],
        0.0,
        0.0,
        0.0,
        cf2.initialPosition[0] + offset2[0],
        cf2.initialPosition[1] + offset2[1],
        cf2.initialPosition[2] + offset2[2],
        0.0,
        0.0,
        0.0
    ])
    x_star = 0.0
    vx_star = 0.0
    vy_star1 = 0.5
    vy_star2 = 0.5
    z_star1 = offset1[2]
    z_star2 = offset2[2]
    vz_star = 0.0
    dt = 0.1
    
    start_time = timeHelper.time() # get current time
    next_X = np.zeros((12))
    velocity1 = [0.0, 0.0, 0.0]
    velocity2 = [0.0, 0.0, 0.0]
    state1 = [0.0, 0.0, 0.0]
    state2 = [0.0, 0.0, 0.0]
    onboard_velocity1 = [0.0, 0.0, 0.0]
    onboard_velocity2 = [0.0, 0.0, 0.0]
    while not timeHelper.isShutdown(): # while the timeHelper is not shutdown
        t = timeHelper.time() - start_time # get the time
        if t > (total_horizon)*dt: # if the time is greater than the duration of the trajectory
            break 
        if real_position:
            state1 = cf1.position()
            state2 = cf2.position()
        if real_experiment:
            # global vel
            # listen_to_ros_topic_vel()
            # onboard_velocity[0] = vel[0]
            # onboard_velocity[1] = vel[1]
            # onboard_velocity[2] = vel[2]
            # Kalman:
            # velocity = onboard_velocity # comment out if we use Kalman!
            pass
        else:
            state1 = cf1.position()
            state2 = cf2.position()
            onboard_velocity1 = velocity1
            onboard_velocity2 = velocity2
            pass
        # velocity = read_csv()
        # velocity = cf.velocity()
        
        X = np.array([state1[0], state1[1], state1[2], 
                    velocity1[0], velocity1[1], velocity1[2],
                    state2[0], state2[1], state2[2],
                    velocity2[0], velocity2[1], velocity2[2]
                    ])
        if t > horizon*dt:
            vy_star1 = 0.0
            vy_star2 = 0.0
        action1 = [
            K2@np.array([x_star-state1[0], vx_star-velocity1[0]]),
            K1@np.array([vy_star1-velocity1[1]]),
            K2@np.array([z_star1-state1[2], vz_star-velocity1[2]])
        ]
        action2 = [
            K2@np.array([x_star-state2[0], vx_star-velocity2[0]]),
            K1@np.array([vy_star2-velocity2[1]]),
            K2@np.array([z_star2-state2[2], vz_star-velocity2[2]])
        ]
        next_X = X + dt*np.array([
            velocity1[0],
            velocity1[1],
            velocity1[2],
            action1[0],
            action1[1],
            action1[2],
            velocity2[0],
            velocity2[1],
            velocity2[2],
            action2[0],
            action2[1],
            action2[2]
        ])
        velocity1 = [next_X[3], next_X[4], next_X[5]]
        velocity2 = [next_X[9], next_X[10], next_X[11]]
        state1 = [next_X[0], next_X[1], next_X[2]]
        state2 = [next_X[6], next_X[7], next_X[8]]
        print(X)
        # cf.cmdVelocityWorld(np.array([next_X[3], next_X[4], next_X[5]]), 0.0)
        # cf.cmdPosition(np.array([next_X[0], next_X[1], next_X[2]]), 0.0)
        cf1.cmdFullState(
            next_X[0:3],  # position
            next_X[3:6],  # velocity
            np.zeros((3,)),  # acceleration
            0.0,  # yaw
            np.zeros((3,)) # Omega
        ) 
        cf2.cmdFullState(
            next_X[6:9],  # position
            next_X[9:12],  # velocity
            np.zeros((3,)),  # acceleration
            0.0,  # yaw
            np.zeros((3,)) # Omega
        )
        # import pdb; pdb.set_trace()
        X_list[:,int(t/dt)] = np.array([
            state1[0],
            state1[1],
            state1[2],
            onboard_velocity1[0],
            onboard_velocity1[1],
            onboard_velocity1[2],
            state2[0],
            state2[1],
            state2[2],
            onboard_velocity2[0],
            onboard_velocity2[1],
            onboard_velocity2[2]
        ])
        timeHelper.sleepForRate(rate)


if __name__ == "__main__":
    horizon = 50 # 30 # 50
    stopping_horizon = 15 # 10 #15
    X_list = np.zeros((12,horizon+stopping_horizon))
    swarm = Crazyswarm() # create a Crazyswarm object
    timeHelper = swarm.timeHelper
    cf1 = swarm.allcfs.crazyflies[0]
    cf2 = swarm.allcfs.crazyflies[1]

    rate = 10.0#30.0
    Z1 = 0.5
    Z2 = 0.5
    cf1.takeoff(targetHeight=Z1, duration=2.0)
    cf2.takeoff(targetHeight=Z2, duration=2.0)
    timeHelper.sleep(2.0)
    
    executeTrajectory(timeHelper, cf1,cf2, horizon, stopping_horizon, rate, offset1=np.array([0, 0, Z1]), offset2=np.array([0, 0, Z2]))
    
    cf1.notifySetpointsStop()
    cf2.notifySetpointsStop()
    
    cf1.land(targetHeight=0.05, duration=2.0)
    cf2.land(targetHeight=0.05, duration=2.0)
    timeHelper.sleep(2.0)
    # import pdb; pdb.set_trace()
    np.savetxt('pid_pid_traj_data.csv', X_list, delimiter=',', header='')




# TODO: 
# 1. create a double integrator trajectory [x_t, v_t]
# 2. See whether we do some reasonable thing!

# answer the above questions: yeah, we have done some reasonable thing!

pid_pid.py
Displaying pid_pid.py.
timmypark@berkeley.edu. Press tab to insert.