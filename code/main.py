import casadi as ca
import numpy as np


# Vehicle parameters
b = 1 # width
L = 1 # wheelbase
v_max = 1 # velocity limit
a_min = -1 # acceleration limits
a_max = 1 # acceleration limits
a_y_max = 1 # acceleration limits
delta_max = 1 # steering limit

# TODO: Load track data
# curvature
# w(s) = track width
S = 100  # total track length


# Create optimization problem
opti = ca.Opti()


# Define variables
T = opti.variable()  # final time
N = 100  # number of discretization intervals
dt = T / N  # time step

# State trajectory:
# z = [s, e_y, e_psi, v]
Z = opti.variable(4, N + 1)

# Control trajectory:
# u = [a, delta]
U = opti.variable(2, N)

# State variables
s = Z[0, :]
e_y = Z[1, :]
e_psi = Z[2, :]
v = Z[3, :]

# Control variables
a = U[0, :]
delta = U[1, :]


# Define objective
opti.minimize(T)


# Add constraints
opti.subject_to(T > 0)

# Boundary conditions
opti.subject_to(s[0] == 0)
opti.subject_to(s[-1] == S)
opti.subject_to(v[0] == 0)
opti.subject_to(e_y[0] == 0)
opti.subject_to(e_psi[0] == 0)

# Control and speed constraints
opti.subject_to(opti.bounded(a_min, a, a_max))
opti.subject_to(opti.bounded(-delta_max, delta, delta_max))
opti.subject_to(opti.bounded(0, v, v_max))

# Lateral acceleration constraint
opti.subject_to(v ** 2 / L * ca.tan(delta) <= a_y_max)

# TODO: Track boundary constraints
# ...


# TODO: Time discretization
# ...


# TODO: Set initial guess
# ...


# Configure solver
opti.solver("ipopt")


# Solve problem
solution = opti.solve()


# Extract results
# ...