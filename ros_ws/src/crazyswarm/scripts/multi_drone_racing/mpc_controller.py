import numpy as np
import time
import casadi as ca



# IBR params
N = 10
dt = 0.1
# Dynamics equations (Bicycle model)
def vehicle_dynamics(xk, uk):
    x = xk[0]
    y = xk[1]
    x_dot = xk[2]
    y_dot = xk[3]
    a_x = uk[0]
    a_y = uk[1]
    
    x_next = ca.vertcat(
        x + x_dot* dt,
        y + y_dot * dt,
        x_dot + a_x * dt,
        y_dot + a_y * dt)
    
    return x_next

qr_ratio = ca.MX.sym('qr_ratio', 1)
# Weight matrices for objective
Q = np.diag([1, 1]) / qr_ratio[0]   # Penalize lateral error (e) and heading error (psi)
R = np.diag([1, 1]) / 1e10  # Penalize control inputs (ax, ay)

# MPC definition
x = ca.MX.sym('x', 4, N+1) 
u = ca.MX.sym('u', 2, N)    
x0_ = ca.MX.sym('x0', 4)         # Initial state
target_state = ca.MX.sym('target states', N*2)  # Reference state: [x, y]


a_max = 5.0  # Maximum acceleration (m/s^2), adjust as needed
y_min, y_max = -.5, .5  # y bounds (meters)
cost = 0
g = [] # constraints list
g.append(x[:,0] - x0_[:4])
for k in range(N):
    x_next = vehicle_dynamics(x[:, k], u[:, k])
    g.append(x[:, k+1] - x_next)  # Dynamics constraints

    # Calculate cost function: state deviation and control effort
    state_error = x[:2, k] 
    state_error[0] -= target_state[k]
    state_error[1] -= target_state[N+k]
    cost += ca.mtimes([state_error.T, Q, state_error])  # State cost
    cost += ca.mtimes([u[:, k].T, R, u[:, k]])  # Control cost

# for k in range(N+1):  # Include all time steps, including k=0
#     g.append(x[1, k] - y_max)  # y ≤ y_max
#     g.append(y_min - x[1, k])  # y ≥ y_min
    

ref = ca.reshape(target_state, 2, N)
track_half_width = 0.2
for k in range(N):
    if k < N-1:
        tvec = ref[:,k+1] - ref[:,k]
    else:
        tvec = ref[:,k]   - ref[:,k-1]
    tvec = tvec / ca.norm_2(tvec)
    nvec = ca.vertcat(-tvec[1], tvec[0])

    delta = x[0:2, k] - ref[:,k]
    cte   = ca.dot(nvec, delta)
    g.append(cte - track_half_width)
    g.append(-cte - track_half_width)



for k in range(N):
    g.append(u[0,k]**2 + u[1,k]**2 - a_max**2)  # a_x^2 + a_y^2 <= a_max^2

opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
opt_constraints = ca.vertcat(*g)
nlp_prob = {'f': cost, 'x': opt_variables, 'g': opt_constraints, 'p': ca.vertcat(x0_, target_state, qr_ratio)}
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 500}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# target_state = np.array([0, 0, 0, 10])
def mpc(x0, traj, lookahead_factor=1.0):  # Remove mu, g, L
    a_max = 5.0  # Define here or as a parameter
    x0 = np.array(x0).flatten()
    params = np.concatenate((x0, traj.T.flatten(), np.array([lookahead_factor])))
    x_init = np.tile(x0, (N+1, 1)).T
    u_init = np.zeros((2, N))
    initial_guess = ca.vertcat(ca.reshape(x_init[:4], -1, 1), ca.reshape(u_init, -1, 1))
    big = 1e20
    # Update bounds #NOTE: from GPT :)
    eq_cnt    = 4*(N+1)   # initial + dynamics (4 per step, for steps 0…N)
    ineq1_cnt = 2*N        # cross-track: 2 constraints per step (cte ≤ +half, –cte ≤ +half)
    ineq3_cnt = N         # u_x^2 + u_y^2 - a_max^2 <= 0



    lbg = np.concatenate([
        np.zeros(eq_cnt),             # =0   for all equalities
        -big * np.ones(ineq1_cnt),    # <=0  for y ≤ y_max
        -big * np.ones(ineq3_cnt)     # <=0  for control bound
    ])

    ubg = np.concatenate([
        np.zeros(eq_cnt),             # =0
        np.zeros(ineq1_cnt),          # g <= 0
        np.zeros(ineq3_cnt)           # g <= 0
    ])
    solution = solver(x0=initial_guess, lbg=lbg, ubg=ubg, p=params)
    optimal_solution = solution['x']
    optimal_x = ca.reshape(optimal_solution[:4*(N+1)], 4, N+1).full()
    optimal_u = ca.reshape(optimal_solution[4*(N+1):], 2, N).full()
    u = optimal_u[:, 0]
    print("u", u)
    return float(u[0]), float(u[1])
