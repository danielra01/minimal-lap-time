from pathlib import Path
from time import perf_counter

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


# Vehicle parameters
vehicle_width = 1.26       # b [m]
wheelbase = 1.20           # L [m]
v_max = 12.5               # maximum velocity [m/s]
a_drive_max = 3.0          # maximum acceleration [m/s^2]
a_brake_max = 6.0          # maximum deceleration magnitude [m/s^2]
a_y_max = 8.0              # maximum lateral acceleration [m/s^2]
delta_max = 0.60           # maximum steering angle [rad]
safety_margin = 0.20       # [m]


# Load track data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_FILE = PROJECT_ROOT / "data" / "track_curvature.csv"
track_data = np.genfromtxt(TRACK_FILE, delimiter=",", names=True, dtype=float)
s_data = np.asarray(track_data["s_m"], dtype=float)
kappa_data = np.asarray(track_data["kappa_1pm"], dtype=float)
S = float(s_data[-1])
track_width = 7.0           # w [m]
usable_half_width = (track_width - vehicle_width) / 2.0

# Static obstacle in Frenet coordinates
obstacle_s = np.array([0.20, 0.30, 0.47, 0.54, 0.65, 0.79, 0.87]) * S  # position along the centerline [m]
num_obstacles = len(obstacle_s)                                        # number of obstacles
obstacle_e_y = np.zeros(num_obstacles)                                 # lateral offset from the centerline [m]
obstacle_length = np.full(num_obstacles, 1.5)                          # obstacle length along the track [m]
obstacle_width = np.full(num_obstacles, 0.8)                           # obstacle width across the track [m]

# Effective obstacle radii including vehicle dimensions and safety margin
obstacle_radius_s = (0.5 * obstacle_length + 0.5 * wheelbase + safety_margin)
obstacle_radius_e_y = (0.5 * obstacle_width + 0.5 * vehicle_width + safety_margin)


# Create interpolant of the curvature
kappa_fun = ca.interpolant("kappa_fun", "linear", [s_data], kappa_data)


# Create optimization problem
opti = ca.Opti()


# Define variables
T = opti.variable()         # final time
N = 300                     # number of discretization intervals
dt = T / N                  # time step

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


# Dynamic function
def vehicle_dynamics(z: ca.MX, u: ca.MX) -> ca.MX:
    """Continuous-time kinematic bicycle model in Frenet coordinates."""
    s_k = z[0]
    e_y_k = z[1]
    e_psi_k = z[2]
    v_k = z[3]

    a_k = u[0]
    delta_k = u[1]

    # Keep interpolation evaluation inside the available track range.
    kappa_k = kappa_fun(s_k)
    denominator = 1.0 - kappa_k * e_y_k

    s_dot = v_k * ca.cos(e_psi_k) / denominator
    e_y_dot = v_k * ca.sin(e_psi_k)
    e_psi_dot = v_k / wheelbase * ca.tan(delta_k) - kappa_k * s_dot
    v_dot = a_k

    return ca.vertcat(s_dot, e_y_dot, e_psi_dot, v_dot)

def cumulative_trapezoid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration using only NumPy."""
    result = np.zeros_like(values, dtype=float)
    result[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(grid))
    
    return result

def reconstruct_centerline(s_samples: np.ndarray, curvature_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the centerline from curvature as a function of arc length.

    The initial position is set to (0, 0) and the initial heading to 0 rad.
    Therefore, the reconstructed track may be translated or rotated relative
    to its real geographical position.
    """
    heading = cumulative_trapezoid(curvature_samples, s_samples)
    x_center = cumulative_trapezoid(np.cos(heading), s_samples)
    y_center = cumulative_trapezoid(np.sin(heading), s_samples)

    return x_center, y_center, heading

def rk4_step(z: ca.MX, u: ca.MX, step_size: ca.MX) -> ca.MX:
    """One fourth-order Runge-Kutta integration step."""
    k1 = vehicle_dynamics(z, u)
    k2 = vehicle_dynamics(z + 0.5 * step_size * k1, u)
    k3 = vehicle_dynamics(z + 0.5 * step_size * k2, u)
    k4 = vehicle_dynamics(z + step_size * k3, u)

    return z + step_size / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

for k in range(N):
    z_next = rk4_step(Z[:, k], U[:, k], dt)
    opti.subject_to(Z[:, k + 1] == z_next)

    # State halfway through the interval
    z_mid = rk4_step(Z[:, k], U[:, k], 0.5 * dt)

    s_mid = z_mid[0]
    e_y_mid = z_mid[1]

    for i in range(num_obstacles):
        obstacle_metric_mid = (
            ((s_mid - obstacle_s[i]) / obstacle_radius_s[i]) ** 2
            + ((e_y_mid - obstacle_e_y[i]) / obstacle_radius_e_y[i]) ** 2
        )

        opti.subject_to(obstacle_metric_mid >= 1.0)


# Define objective
opti.minimize(T)


# Add constraints
T_min = 1.0
T_max = 300.0
opti.subject_to(opti.bounded(T_min, T, T_max))

# Boundary conditions
opti.subject_to(s[0] == 0.0)
opti.subject_to(e_y[0] == 0.0)
opti.subject_to(e_psi[0] == 0.0)
opti.subject_to(v[0] == 0.0)

opti.subject_to(s[-1] == S)
opti.subject_to(e_y[-1] == 0.0)
opti.subject_to(e_psi[-1] == 0.0)

# Track constraint
opti.subject_to(
    opti.bounded(-usable_half_width,e_y,usable_half_width))

# Control and speed constraints
opti.subject_to(opti.bounded(-a_brake_max, a, a_drive_max))
opti.subject_to(opti.bounded(-delta_max, delta, delta_max))
opti.subject_to(opti.bounded(0, v, v_max))

# Lateral acceleration constraint
v_control = v[:-1]
a_y = v_control**2 / wheelbase * ca.tan(delta)
opti.subject_to(a_y <= a_y_max)
opti.subject_to(a_y >= -a_y_max)

# Frenet-coordinate validity constraint
frenet_margin = 1e-3
kappa_nodes = kappa_fun(s)
opti.subject_to(1.0 - kappa_nodes * e_y >= frenet_margin)

# Obstacle avoidance constraints
for i in range(num_obstacles):
    obstacle_metric = (
        ((s - obstacle_s[i]) / obstacle_radius_s[i]) ** 2
        + ((e_y - obstacle_e_y[i]) / obstacle_radius_e_y[i]) ** 2
    )

    opti.subject_to(obstacle_metric >= 1.0)


# s only move forward
for k in range(N):
    opti.subject_to(s[k + 1] >= s[k])
opti.subject_to(opti.bounded(0.0, s, S))


# Set initial guess
v_cruise_guess = min(8.0, 0.7 * v_max)

t_acc = v_cruise_guess / a_drive_max
s_acc = 0.5 * a_drive_max * t_acc**2

if s_acc < S:
    T_guess = t_acc + (S - s_acc) / v_cruise_guess
else:
    T_guess = np.sqrt(2.0 * S / a_drive_max)

time_guess = np.linspace(0.0, T_guess, N + 1)

v_guess = np.minimum(a_drive_max * time_guess, v_cruise_guess)

s_guess = np.zeros(N + 1)

for k in range(N):
    dt_guess = time_guess[k + 1] - time_guess[k]
    s_guess[k + 1] = (s_guess[k] + 0.5 * (v_guess[k] + v_guess[k + 1]) * dt_guess)

s_guess *= S / s_guess[-1]

a_guess = np.zeros(N)

for k in range(N):
    if time_guess[k] < t_acc:
        a_guess[k] = a_drive_max
    else:
        a_guess[k] = 0.0

kappa_guess = np.interp(s_guess[:-1], s_data, kappa_data)

delta_guess = np.arctan(wheelbase * kappa_guess)

delta_guess = np.clip(delta_guess, -delta_max, delta_max)

opti.set_initial(T, T_guess)
opti.set_initial(s, s_guess)
opti.set_initial(v, v_guess)
opti.set_initial(a, a_guess)
opti.set_initial(delta, delta_guess)
# opti.set_initial(e_y, 0.5 * usable_half_width)
opti.set_initial(e_y, 0.0)
opti.set_initial(e_psi, 0.0)

delta_required = np.arctan(wheelbase * kappa_data)

print(
    "Required centerline steering range:",
    np.min(delta_required),
    np.max(delta_required),
)

print("Configured delta_max:", delta_max)
if np.max(np.abs(delta_required)) > delta_max:
    print(
        "Warning: the steering limit is too restrictive "
        "to follow parts of the centerline."
    )

# Configure solver
solver_options = {
    "expand": True,
}
ipopt_options = {
    "max_iter": 3000,
    "tol": 1e-7,
    "constr_viol_tol": 1e-7,
    "acceptable_tol": 1e-5,
    "acceptable_constr_viol_tol": 1e-5,
    "print_level": 5,
}
opti.solver(
    "ipopt",
    solver_options,
    ipopt_options,
)


# Solve problem
try:
    solve_start = perf_counter()
    solution = opti.solve()
    solve_time = perf_counter() - solve_start

except RuntimeError as error:
    print("The optimization did not converge.")
    print(error)
    print("Solver status:")
    print(opti.stats()["return_status"])
    T_debug = float(opti.debug.value(T))
    print(f"Last final-time iterate: {T_debug:.6f} s")
    raise


# Extract optimization results
T_opt = float(solution.value(T))
Z_opt = np.asarray(solution.value(Z))
U_opt = np.asarray(solution.value(U))

# State trajectories
s_opt = Z_opt[0, :]
e_y_opt = Z_opt[1, :]
e_psi_opt = Z_opt[2, :]
v_opt = Z_opt[3, :]

# Control trajectories
a_opt = U_opt[0, :]
delta_opt = U_opt[1, :]

# Time grids
time_nodes = np.linspace(0.0, T_opt, N + 1)
time_controls = time_nodes[:-1]

# Evaluate track curvature along the optimized trajectory
kappa_opt = np.interp(s_opt, s_data, kappa_data)

# Compute derived physical quantities
a_y_opt = v_opt[:-1]**2 / wheelbase * np.tan(delta_opt)
frenet_denominator_opt = 1.0 - kappa_opt * e_y_opt
mean_velocity = np.trapezoid(v_opt, time_nodes) / T_opt
maximum_velocity = np.max(v_opt)
mean_velocity_kmh = 3.6 * mean_velocity
maximum_velocity_kmh = 3.6 * maximum_velocity

# Reconstruct the Cartesian centerline from kappa(s).
x_center_data, y_center_data, heading_data = reconstruct_centerline(s_data, kappa_data)

# Interpolate the centerline and its heading at the optimized s nodes.
x_center_opt = np.interp(s_opt, s_data, x_center_data)
y_center_opt = np.interp(s_opt, s_data, y_center_data)
heading_opt = np.interp(s_opt, s_data, heading_data)

# Unit normal vector along the centerline.
normal_x_opt = -np.sin(heading_opt)
normal_y_opt = np.cos(heading_opt)

# Optimized kart reference-point trajectory.
x_trajectory_opt = x_center_opt + e_y_opt * normal_x_opt
y_trajectory_opt = y_center_opt + e_y_opt * normal_y_opt

track_half_width = track_width / 2.0

normal_x_data = -np.sin(heading_data)
normal_y_data = np.cos(heading_data)

# Physical track boundaries.
x_left_boundary = x_center_data + track_half_width * normal_x_data
y_left_boundary = y_center_data + track_half_width * normal_y_data

x_right_boundary = x_center_data - track_half_width * normal_x_data
y_right_boundary = y_center_data - track_half_width * normal_y_data

# Solver information

solver_stats = opti.stats()

solver_status = solver_stats.get(

    "return_status",

    "Unknown",

)

solver_iterations = solver_stats.get(

    "iter_count",

    np.nan,

)

# Print solution summary

print("\nSolution summary")
print("----------------")
print(f"Optimal lap time: {T_opt:.3f} s")
print(f"Mean velocity: {mean_velocity:.3f} m/s ({mean_velocity_kmh:.2f} km/h)")
print(f"Maximum velocity: {maximum_velocity:.3f} m/s ({maximum_velocity_kmh:.2f} km/h)")
print(f"Lateral-offset range: [{np.min(e_y_opt):.3f}, {np.max(e_y_opt):.3f}] m")
print(f"Maximum absolute lateral offset: {np.max(np.abs(e_y_opt)):.3f} m")
print(f"Acceleration range: [{np.min(a_opt):.3f}, {np.max(a_opt):.3f}] m/s^2")
print(f"Steering-angle range: [{np.min(delta_opt):.3f}, {np.max(delta_opt):.3f}] rad")
print(f"Lateral-acceleration range: [{np.min(a_y_opt):.3f}, {np.max(a_y_opt):.3f}] m/s^2")
print(f"Minimum Frenet denominator: {np.min(frenet_denominator_opt):.6f}")
print(f"Solver status: {solver_status}")
print(f"Solver iterations: {solver_iterations}")
print(f"Solver time: {solve_time:.3f} s")


# Plots
plt.figure()
plt.plot(s_opt, v_opt)
plt.xlabel("Centerline distance s [m]")
plt.ylabel("Velocity v [m/s]")
plt.title("Optimal velocity profile")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(s_opt, e_y_opt)
plt.axhline(usable_half_width, linestyle="--")
plt.axhline(-usable_half_width, linestyle="--")
plt.xlabel("Centerline distance s [m]")
plt.ylabel("Lateral offset e_y [m]")
plt.title("Optimal lateral position")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.step(time_controls, a_opt, where="post")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration a [m/s²]")
plt.title("Optimal longitudinal acceleration")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.step(time_controls, delta_opt, where="post")
plt.xlabel("Time [s]")
plt.ylabel("Steering angle delta [rad]")
plt.title("Optimal steering input")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))

# Fill the physical track area.
plt.fill(
    np.concatenate([
        x_left_boundary,
        x_right_boundary[::-1],
    ]),
    np.concatenate([
        y_left_boundary,
        y_right_boundary[::-1],
    ]),
    alpha=0.25,
    label="Track",
    zorder=1
)

# Physical boundaries.
plt.plot(
    x_left_boundary,
    y_left_boundary,
    linewidth=1.0,
    label="Track boundaries",
    zorder=2
)

plt.plot(
    x_right_boundary,
    y_right_boundary,
    linewidth=1.0,
    zorder=3
)

# Centerline.
plt.plot(
    x_center_data,
    y_center_data,
    linestyle="--",
    linewidth=1.5,
    label="Centerline",
    zorder=4
)

# Optimized trajectory.
plt.plot(
    x_trajectory_opt,
    y_trajectory_opt,
    linewidth=2.0,
    label="Optimized trajectory",
    zorder=7
)

# Starting point.
plt.scatter(
    x_center_data[0],
    y_center_data[0],
    marker="o",
    label="Start",
    zorder=5
)

# Obstacle safety regions.
theta = np.linspace(0.0, 2.0 * np.pi, 150)
for i in range(num_obstacles):
    # Ellipse boundary in Frenet coordinates
    s_obstacle_boundary = (obstacle_s[i] + obstacle_radius_s[i] * np.cos(theta))
    e_y_obstacle_boundary = (obstacle_e_y[i] + obstacle_radius_e_y[i] * np.sin(theta))

    # Centerline position and heading at every ellipse point
    x_center_obstacle = np.interp(s_obstacle_boundary, s_data, x_center_data)
    y_center_obstacle = np.interp(s_obstacle_boundary, s_data, y_center_data)
    heading_obstacle = np.interp(s_obstacle_boundary, s_data, heading_data)

    # Unit normal to the centerline
    normal_x_obstacle = -np.sin(heading_obstacle)
    normal_y_obstacle = np.cos(heading_obstacle)

    # Frenet-to-Cartesian transformation
    x_obstacle_boundary = (x_center_obstacle + e_y_obstacle_boundary * normal_x_obstacle)
    y_obstacle_boundary = (y_center_obstacle + e_y_obstacle_boundary * normal_y_obstacle)

    plt.fill(
        x_obstacle_boundary,
        y_obstacle_boundary,
        color="tab:green",
        label="Obstacle safety region" if i == 0 else None,
        zorder=6
    )

# plt.scatter(
#     x_trajectory_opt,
#     y_trajectory_opt,
#     s=12,
#     color="black",
#     zorder=8,
#     label="Optimization nodes",
# )


plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Track, centerline, and optimized trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

for i in range(num_obstacles):
    obstacle_metric_opt = (
        ((s_opt - obstacle_s[i]) / obstacle_radius_s[i]) ** 2
        + ((e_y_opt - obstacle_e_y[i]) / obstacle_radius_e_y[i]) ** 2
    )

    print(
        f"Obstacle {i + 1}:",
        obstacle_metric_opt.min()
    )

# Save result to compare
np.savez(
    "results_obstacles.npz",
    s=s_opt,
    e_y=e_y_opt,
    v=v_opt,
    total_time=T_opt
)