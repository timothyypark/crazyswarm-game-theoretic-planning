"""waypoint_generator.py

Unified waypoint generator for holonomic **drone** simulations and optional
matplotlib visualisation.

Supported track specs
---------------------
* **Analytic**:   ``counter circle`` | ``counter oval`` | ``counter square``
* **YAML**   :    A file ending in ``.yaml`` (same format as original car code).
* **Custom CSV**: Any other string is treated as a basename and loaded from
  ``ref_trajs/<name>_with_speeds.csv`` containing columns ``x,y[,v_ref]``.

Returned format
---------------
``generate(obs) -> np.ndarray`` of shape ``(H+1, 3)`` with columns
``[x, y, v_ref]``.

GPU: heavy maths use **JAX** (`jax.numpy`) so everything runs on CUDA/TPU when
available.  Visualisation uses standard Matplotlib and therefore runs on CPU.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Analytic reference tracks
# ---------------------------------------------------------------------------

def _counter_circle(theta: float) -> jnp.ndarray:
    R = 4.5
    return jnp.array([R * jnp.cos(theta), R * jnp.sin(theta)])


def _counter_oval(theta: float) -> jnp.ndarray:
    Rx, Ry = 1.2, 1.4
    return jnp.array([Rx * jnp.cos(theta), Ry * jnp.sin(theta)])


def _counter_square(theta: float) -> jnp.ndarray:
    """Unit square centred at origin parameterised by wrapped angle ``theta``."""
    theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
    h = 2.0  # half‑width
    if -jnp.pi / 4 <= theta <= jnp.pi / 4:
        return jnp.array([h, jnp.tan(theta) * h])
    if -3 * jnp.pi / 4 <= theta <= -jnp.pi / 4:
        return jnp.array([-jnp.tan(theta + jnp.pi / 2) * h, -h])
    if theta >= 3 * jnp.pi / 4 or theta <= -3 * jnp.pi / 4:
        return jnp.array([-h, -jnp.tan(theta) * h])
    return jnp.array([jnp.tan(theta - jnp.pi / 2) * h, h])

def _counter_rounded_rectangle(theta: float) -> jnp.ndarray:
    """3×5 m rectangle with *straight* sides and 0.5 m quarter‑circle fillets.

    The parameter ``theta`` (any real number) is mapped linearly onto the
    **perimeter arc‑length** so that equal increments in ``theta`` correspond
    to equal travel distance, which is convenient for vmap sampling.
    """
    # Geometry -------------------------------------------------------------
    W, H, r = 3.0, 5.0, 0.5  # full width, full height, corner radius (m)
    sx, sy = W - 2 * r, H - 2 * r  # lengths of the straight edges

    perim = 2 * (sx + sy) + 2 * jnp.pi * r  # total perimeter
    s = (theta % (2 * jnp.pi)) / (2 * jnp.pi) * perim  # unwrap to arc‑length

    # Cumulative arc‑length boundaries of each segment (clock‑wise, start top‑mid)
    B0 = sx
    B1 = B0 + 0.5 * jnp.pi * r
    B2 = B1 + sy
    B3 = B2 + 0.5 * jnp.pi * r
    B4 = B3 + sx
    B5 = B4 + 0.5 * jnp.pi * r
    B6 = B5 + sy

    # Convenience lambdas for local coordinates within each segment --------
    top     = lambda s0: jnp.array([( W/2 - r) - s0,  H/2])
    tl_arc  = lambda s0: jnp.array([-(W/2) + r * (1 - jnp.sin(s0 / r)),  H/2 - r * (1 - jnp.cos(s0 / r))])
    left    = lambda s0: jnp.array([ -W/2,  (H/2 - r) - s0])
    bl_arc  = lambda s0: jnp.array([-(W/2 - r) + r * (-jnp.cos(s0 / r)), -(H/2 - r) + r * (-jnp.sin(s0 / r))])
    bottom  = lambda s0: jnp.array([-(W/2 - r) + s0, -H/2])
    br_arc  = lambda s0: jnp.array([( W/2 - r) + r * (jnp.sin(s0 / r)), -(H/2 - r) + r * (-jnp.cos(s0 / r))])
    right   = lambda s0: jnp.array([  W/2, -(H/2 - r) + s0])
    tr_arc  = lambda s0: jnp.array([( W/2 - r) + r * ( jnp.cos(s0 / r)),  (H/2 - r) + r * ( jnp.sin(s0 / r))])

    # Select piece‑wise position with jnp.select (works in JIT / vmap) ------
    conds = (
        s < B0,
        (s >= B0) & (s < B1),
        (s >= B1) & (s < B2),
        (s >= B2) & (s < B3),
        (s >= B3) & (s < B4),
        (s >= B4) & (s < B5),
        (s >= B5) & (s < B6),
    )

    choices_x = (
        top(s)[0],
        tl_arc(s - B0)[0],
        left(s - B1)[0],
        bl_arc(s - B2)[0],
        bottom(s - B3)[0],
        br_arc(s - B4)[0],
        right(s - B5)[0],
    )
    choices_y = (
        top(s)[1],
        tl_arc(s - B0)[1],
        left(s - B1)[1],
        bl_arc(s - B2)[1],
        bottom(s - B3)[1],
        br_arc(s - B4)[1],
        right(s - B5)[1],
    )

    x = jnp.select(conds, choices_x, default=tr_arc(s - B6)[0])
    y = jnp.select(conds, choices_y, default=tr_arc(s - B6)[1])

    return jnp.array([x, y])


_ANALYTIC_MAP: dict[str, Callable[[float], jnp.ndarray]] = {
    "counter circle": _counter_circle,
    "counter oval": _counter_oval,
    "counter square": _counter_square,
    "counter rectangle": _counter_rounded_rectangle,
}

# ---------------------------------------------------------------------------
# Helpers to load external trajectory files
# ---------------------------------------------------------------------------

def _load_from_yaml(yaml_file: Path, speed: float) -> np.ndarray:
    """Parse YAML from the original car repo and return (N,3) `[x,y,v]`."""
    yml = yaml.safe_load(open(yaml_file, "r"))
    ox, oy = yml["track_info"]["ox"], yml["track_info"]["oy"]
    scale = float(yml["track_info"]["scale"])
    cname = Path(yml["track_info"]["centerline_file"]).stem
    csv_path = yaml_file.parent / ".." / "ref_trajs" / f"{cname}_with_speeds.csv"
    df = pd.read_csv(csv_path)
    arr = df[["x", "y", "v_ref"]].values.astype(float)
    arr[:, :2] = arr[:, :2] * scale + np.array([ox, oy])
    arr[:, 2] = speed  # override speed uniformly
    return arr


def _load_from_csv(basename: str, speed: float) -> np.ndarray:
    """Load `ref_trajs/<basename>_with_speeds.csv`.
    If `v_ref` missing, the provided uniform `speed` is inserted.
    """
    csv_path = Path(__file__).resolve().parent / ".." / "ref_trajs" / f"{basename}_raceline_with_speeds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if {"x", "y"}.issubset(df.columns):
        arr_xy = df[["x", "y"]].values.astype(float)
    else:  # legacy order (s,x,y, ...)
        arr_xy = df.iloc[:, [1, 2]].values.astype(float)
    v = (
        df["v_ref"].values.astype(float)
        if "v_ref" in df.columns
        else np.full(len(arr_xy), speed)
    )
    return np.column_stack((arr_xy, v))

# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class WaypointGenerator:
    def __init__(self, waypoint_type: str, dt: float, horizon: int, speed: float):
        self.dt = float(dt)
        self.H = int(horizon)
        self.nom_speed = float(speed)

        # Resolve track spec --------------------------------------------------
        if waypoint_type in _ANALYTIC_MAP:
            self._mode = "analytic"
            self._fn: Callable[[float], jnp.ndarray] = _ANALYTIC_MAP[waypoint_type]
            # pre-sample for analytic tracks (helps closest-point search)
            ts = jnp.arange(0.0, 2 * jnp.pi, self.dt / self.nom_speed)
            path_xy = np.asarray(jax.vmap(self._fn)(ts))
            # raceline: [x, y, v_ref]
            self.raceline = np.column_stack((path_xy, np.full(len(path_xy), self.nom_speed)))
        elif waypoint_type.endswith(".yaml"):
            self._mode = "yaml"
            traj = _load_from_yaml(Path(waypoint_type), speed=self.nom_speed)
            # traj already is (N,3) = [x, y, v_ref]
            self.raceline = traj
        else:
            self._mode = "csv"
            traj = _load_from_csv(waypoint_type, speed=self.nom_speed)
            # traj is (N,3) = [x, y, v_ref]
            self.raceline = traj

        # Compute cumulative arc-length along raceline -----------------------
        # differences between successive (x,y)
        deltas = np.diff(self.raceline[:, :2], axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        print("segment_lengths", segment_lengths)
        # cum_s[0] = 0, then cumulative sum of segment_lengths
        self.cum_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.left_boundary, self.right_boundary = self.get_boundaries(0.5)
        self.waypoint_list_np = np.array(self.raceline, dtype=float)

    def generate(
        self,
        obs: np.ndarray,
        dt: float,
        kin_horizon: float = 1.2,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        pos = obs[:2]
        # find closest point index
        idx = int(np.linalg.norm(self.raceline[:, :2] - pos[None, :], axis=1).argmin())

        # lateral error
        def tangent(i):
            p0, p1 = self.raceline[i, :2], self.raceline[(i+1)%len(self.raceline), :2]
            t = p1 - p0
            return t / np.linalg.norm(t)

        t_hat = tangent(idx)
        n_hat = np.array([-t_hat[1], t_hat[0]])
        e_lateral = float(np.dot(pos - self.raceline[idx, :2], n_hat))

        # progress
        s_progress = float(self.cum_s[idx])

        # build (H+1) future waypoints [x, y, vx, vy]
        target = []
        for k in range(self.H + 1):
            j = (idx + k) % len(self.raceline)
            # unit-tangent times reference speed
            p0 = self.raceline[j, :2]
            t = tangent(j)
            v_ref = float(self.raceline[j, 2])
            vx, vy = t * v_ref
            target.append([p0[0], p0[1], float(vx), float(vy)])
        target_pos_list = np.asarray(target, dtype=float)

        # kinematic look-ahead point
        # distance per index
        # use the next segment length for ds
        ds = np.linalg.norm(self.raceline[(idx+1)%len(self.raceline), :2] - self.raceline[idx, :2])
        # use the local v_ref for kin‐step
        local_v = float(self.raceline[idx, 2])
        kin_step = int(round((kin_horizon * local_v) / max(ds, 1e-6)))
        kin_pos = self.raceline[(idx + kin_step) % len(self.raceline), :2]

        return target_pos_list, kin_pos, s_progress, e_lateral
    
    def calc_shifted_traj(self, traj, shift_dist) :
        # This function calculates the shifted trajectory given the original trajectory and the shift distance.
        traj_ = np.copy(traj)
        traj_[:-1] = traj[1:]
        traj_[-1] = traj[0]
        _traj = np.copy(traj)
        _traj[1:] = traj[:-1]
        _traj[0] = traj[-1]
        yaws = np.arctan2(traj_[:,1] - _traj[:,1], traj_[:,0] - _traj[:,0])
        traj_new = np.copy(traj)
        traj_new[:,0] = traj[:,0] + shift_dist * np.cos(yaws + np.pi/2)
        traj_new[:,1] = traj[:,1] + shift_dist * np.sin(yaws + np.pi/2)
        return traj_new

    def get_boundaries(self, half_width: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (left_boundary, right_boundary), each an (N,2) array of points,
        offset from the centerline by +half_width and −half_width respectively.
        """
        center_xy = self.raceline[:, :2]
        left  = self.calc_shifted_traj(center_xy, +half_width)
        right = self.calc_shifted_traj(center_xy, -half_width)
        return left, right

# ---------------------------------------------------------------------------
# Stand‑alone demo & visualisation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Choose a track and generator parameters --------------------------------
    gen = WaypointGenerator("counter rectangle", dt=0.1, horizon=8, speed=0.5)

    # Dense sampling of the full track for plotting --------------------------
    if gen._mode == "analytic":
        dense_xy = gen.raceline
    else:
        dense_xy = gen.raceline

    # Fake drone at some arbitrary starting point ---------------------------
    obs = np.array([dense_xy[0, 0], dense_xy[0, 1], 0.0, 0.0])
    waypoints, _, _, _ = gen.generate(obs, 0.1)

    # Plot track and look‑ahead waypoints -----------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(dense_xy[:, 0], dense_xy[:, 1], "k--", alpha=0.4, label="track")
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c=np.linspace(0, 1, len(waypoints)), cmap="viridis", label="Horizon")
    plt.scatter(obs[0], obs[1], c="red", s=60, label="Drone pos")
    plt.axis("equal")
    plt.title("Waypoint generator preview")
    plt.legend()
    plt.tight_layout()
    plt.show()
