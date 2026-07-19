import numpy as np
import matplotlib.pyplot as plt


# Parameters
vehicle_width = 1.26       # b [m]
track_width = 7.0           # w [m]
usable_half_width = (track_width - vehicle_width) / 2.0

baseline = np.load("results_baseline.npz")
obstacles = np.load("results_obstacles.npz")

# Plot optimal lateral position
plt.figure()

plt.plot(
    baseline["s"],
    baseline["e_y"],
    label="Baseline"
)

plt.plot(
    obstacles["s"],
    obstacles["e_y"],
    label="With obstacles"
)

plt.axhline(usable_half_width, linestyle="--")
plt.axhline(-usable_half_width, linestyle="--")
plt.xlabel("Centerline distance s [m]")
plt.ylabel("Lateral offset e_y [m]")
plt.title("Optimal lateral position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot optimal velocity profile
plt.figure()

plt.plot(
    baseline["s"],
    baseline["v"],
    label="Baseline"
)

plt.plot(
    obstacles["s"],
    obstacles["v"],
    label="With obstacles"
)

plt.xlabel("Centerline distance s [m]")
plt.ylabel("Velocity v [m/s]")
plt.title("Optimal velocity profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()