import numpy as np
import casadi as ca

# IBR params
N = 10

dt = 0.1

# Dynamics equations (double integrator model)
def vehicle_dynamics(xk, uk):
    x    = xk[0]
    y    = xk[1]
    x_dot = xk[2]
    y_dot = xk[3]
    a_x  = uk[0]
    a_y  = uk[1]

    x_next = ca.vertcat(
        x     + x_dot * dt,
        y     + y_dot * dt,
        x_dot + a_x    * dt,
        y_dot + a_y    * dt
    )
    return x_next

# CasADi symbols
qr_ratio     = ca.MX.sym('qr_ratio', 1)
x            = ca.MX.sym('x', 4, N+1)
u            = ca.MX.sym('u', 2, N)
x0_         = ca.MX.sym('x0', 4)
target_state = ca.MX.sym('target_states', N*2)

# Weights for cost
a_max = 2.5      # [m/s^2] acceleration bound (high so constraint is inactive)
v_max = 0.9     # [m/s]   maximum speed bound (adjust as needed)
Q = np.diag([10000, 10000]) * qr_ratio  # state error weight
dQ = ca.MX(Q)
R = np.diag([100, 100])              # control effort weight

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
    err = x[:2,k]
    err = ca.vertcat(err[0] - target_state[k], err[1] - target_state[N+k])
    cost += ca.mtimes([err.T, dQ, err])

    # control effort cost
    cost += ca.mtimes([u[:,k].T, R, u[:,k]])

# acceleration bounds
g.extend([(u[0,k]**2 + u[1,k]**2 - a_max**2) for k in range(N)])

# speed bounds
g.extend([(x[2,k]**2 + x[3,k]**2 - v_max**2) for k in range(N)])

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
eq_cnt      = 4 * (N+1)
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
    Run MPC: x0    -> current state (4,)
             traj  -> reference trajectory shape (2, N)
             returns first control action (a_x, a_y)
    """
    x0 = np.array(x0).flatten()
    # build parameter vector
    params = np.concatenate((x0, traj.T.flatten(), np.array([lookahead_factor])))

    # initial guess for x and u
    x_init = np.tile(x0, (N+1, 1)).T
    u_init = np.zeros((2, N))
    initial_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))


    # solve
    sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg, p=params)
    opt = sol['x']

    # extract controls
    opt_x = ca.reshape(opt[:4*(N+1)], 4, N+1).full()[:,0]
    opt_u = ca.reshape(opt[4*(N+1):], 2, N).full()
    a_x, a_y = opt_u[:,0]
    return float(a_x), float(a_y), np.array(opt_x)
