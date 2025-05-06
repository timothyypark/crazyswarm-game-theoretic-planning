import numpy as np
import casadi as ca

# IBR params
N = 10

dt = 0.1

# Dynamics equations (double integrator model)
def vehicle_dynamics(xk, uk):
    # x    = xk[0]
    # y    = xk[1]
    # x_dot = xk[2]
    # y_dot = xk[3]
    # a_x  = uk[0]
    # a_y  = uk[1]
    
    x, y, z, x_dot, y_dot, z_dot = xk[0], xk[1], xk[2], xk[3], xk[4], xk[5]
    a_x, a_y, a_z = uk[0], uk[1], uk[2]

    x_next = ca.vertcat(
        x     + x_dot * dt,
        y     + y_dot * dt,
        z     + z_dot  * dt,
        x_dot + a_x    * dt,
        y_dot + a_y    * dt,
        z_dot + a_z    * dt,
    )
    return x_next

# CasADi symbols -- in 3d, now 6-state, 3-control, 3D targets
qr_ratio     = ca.MX.sym('qr_ratio', 1)
x            = ca.MX.sym('x', 6, N+1)
u            = ca.MX.sym('u', 3, N)
x0_         = ca.MX.sym('x0', 6)
target_state = ca.MX.sym('target_states', N*3)

# Weights for cost
a_max = 2.5      # [m/s^2] acceleration bound (high so constraint is inactive)
v_max = 0.7     # [m/s]   maximum speed bound (adjust as needed)
Q = np.diag([10000, 10000, 10000]) * qr_ratio  # state error weight
dQ = ca.MX(Q)
R = np.diag([100, 100, 100])              # control effort weight

# Build objective and constraints
cost = 0
g    = []
# initial state constraint
g.append(x[:,0] - x0_)

for k in range(N):
    # dynamics constraints
    x_next = vehicle_dynamics(x[:,k], u[:,k])
    g.append(x[:,k+1] - x_next)

    # tracking cost
    
    #prev in 2d
    # err = x[:2,k]
    # err = ca.vertcat(err[0] - target_state[k], err[1] - target_state[N+k])
    
    #change for 3d
    # err = x[:3, k] - target_state[k : k+3*N : N]  # picks [k, N+k, 2N+k]
    tx = target_state[k]           # x-target at index k
    ty = target_state[N + k]       # y-target at index N+k
    tz = target_state[2*N + k]     # z-target at index 2N+k

    err = ca.vertcat(
        x[0, k] - tx,
        x[1, k] - ty,
        x[2, k] - tz,
    )
    cost += ca.mtimes([err.T, dQ, err])

    # control effort cost
    cost += ca.mtimes([u[:,k].T, R, u[:,k]])

# acceleration bounds
g.extend([(u[0,k]**2 + u[1,k]**2 + u[2,k]**2 - a_max**2) for k in range(N)])

# speed bounds
# g.extend([(x[2,k]**2 + x[3,k]**2 - v_max**2) for k in range(N)])
g.extend([(x[3,k]**2 + x[4,k]**2 + x[5,k]**2 - v_max**2) for k in range(N)])

# concatenate variables and constraints
opt_variables   = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
opt_constraints = ca.vertcat(*g)

# NLP problem definition
nlp_prob = {
    'f': cost,
    'x': opt_variables,
    'g': opt_constraints,
    'p': ca.vertcat(x0_, target_state, qr_ratio)
}

opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 500}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# big for inequalities
big = 1e20
eq_cnt      = 6 * (N+1) #changed from 4 to 6
ineq_acc    = N
ineq_speed = N

# build bounds for g
lbg = np.concatenate([
    np.zeros(eq_cnt),             # equalities = 0
    -big * np.ones(ineq_acc),     # accel ≤ 0
    -big * np.ones(ineq_speed),   # speed ≤ 0
])
ubg = np.concatenate([
    np.zeros(eq_cnt),             # equalities = 0
    np.zeros(ineq_acc),           # accel ≤ 0
    np.zeros(ineq_speed),         # speed ≤ 0
])

# MPC wrapper function
def mpc(x0, traj, lookahead_factor=1.0):
    """
    Run MPC: x0    -> current state (6,)
             traj  -> reference trajectory shape (3, N)
             returns first control action (a_x, a_y, a_z)
    """
    x0 = np.array(x0).flatten()
    # build parameter vector
    # params = np.concatenate((x0, traj.T.flatten(), np.array([lookahead_factor])))

    # traj_flat = traj.flatten(order='F')
    # params = np.concatenate((x0, traj_flat, np.array([lookahead_factor])))
    
    tx = traj[0, :]     # shape (N,)
    ty = traj[1, :]
    tz = traj[2, :]
    packed = np.concatenate((tx, ty, tz))  # shape (3*N,)
    params = np.concatenate((x0, packed, np.array([lookahead_factor])))
    
    # initial guess for x and u
    x_init = np.tile(x0, (N+1, 1)).T
    u_init = np.zeros((3, N))
    initial_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))


    # solve
    sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg, p=params)
    opt = sol['x']

    # extract controls
    opt_x = ca.reshape(opt[:6*(N+1)], 6, N+1).full()[:,0]
    opt_u = ca.reshape(opt[6*(N+1):], 3, N).full()
    a_x, a_y, a_z = opt_u[:,0]
    return float(a_x), float(a_y), float(a_z), np.array(opt_x)
