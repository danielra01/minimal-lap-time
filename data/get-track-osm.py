"""
Minimal track model: arc length s  ->  curvature kappa(s).

Steps: read OSM json -> project to UTM 32N (meters) -> periodic smoothing
spline -> curvature on a uniform s-grid -> CSV (s, kappa) + one plot.
"""
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import splprep, splev
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")

# --- smoothing: allowed total squared deviation [m^2] between spline and
#     the raw OSM points. Rule of thumb: s ~= N * sigma^2, where sigma is the
#     assumed positional error per node. N=81, sigma~0.5 m  ->  s ~= 20.
SMOOTHING = 20.0

# 1. read + order nodes in way sequence
data = json.load(open("full.json"))
node = {e["id"]: (e["lon"], e["lat"])
        for e in data["elements"] if e["type"] == "node"}
way = next(e for e in data["elements"] if e["type"] == "way")
ll = np.array([node[i] for i in way["nodes"]])

# 2. project WGS84 -> UTM 32N (EPSG:25832), meters
E, N = Transformer.from_crs(
    "EPSG:4326", "EPSG:25832", always_xy=True).transform(ll[:, 0], ll[:, 1])
# drop duplicated closing vertex (loop)
pts = np.column_stack([E, N])[:-1]

# 3. periodic smoothing spline + curvature
tck, _ = splprep([pts[:, 0], pts[:, 1]], s=SMOOTHING, per=1, k=3)
u = np.linspace(0, 1, 2000, endpoint=False)
dx, dy = splev(u, tck, der=1)
ddx, ddy = splev(u, tck, der=2)
kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
ds = np.hypot(*splev(u, tck, der=1)) * (1.0 / len(u))     # arc length per step
s = np.concatenate([[0], np.cumsum(ds)[:-1]])
L = ds.sum()

# 4. resample onto a uniform s-grid
s_uni = np.linspace(0, L, 1000, endpoint=False)
k_uni = np.interp(s_uni, s, kappa, period=L)

# 5. save CSV: arc length -> curvature
np.savetxt("track_curvature.csv", np.column_stack([s_uni, k_uni]),
           delimiter=",", header="s_m,kappa_1pm", comments="")

print(f"length L      = {L:6.1f} m")
print(
    f"R_min         = {1/np.max(np.abs(k_uni)):5.2f} m   (max|kappa| = {np.max(np.abs(k_uni)):.3f} 1/m)")
print(f"rows in CSV   = {len(s_uni)}  (uniform ds = {L/len(s_uni):.2f} m)")

# 6. plot: s -> kappa
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(s_uni, k_uni, lw=1.5, color="#185FA5")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("Arc length s [m]")
ax.set_ylabel("Curvature kappa(s) [1/m]")
ax.set_title(
    f"Track curvature profile  (smoothing s = {SMOOTHING:.0f} m^2,  L = {L:.0f} m)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("curvature_profile.png", dpi=130)
print("written: track_curvature.csv , curvature_profile.png")
