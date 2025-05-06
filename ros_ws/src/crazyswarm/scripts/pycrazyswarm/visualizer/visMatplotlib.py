import warnings
from collections import deque

from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import numpy as np


class VisMatplotlib:
    def __init__(self, max_tail_length=200):
        # Set up 3D figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 3])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Scatter plot handle (for the drones)
        self.plot = None
        self.timeAnnotation = self.ax.annotate(
            "Time", xy=(0, 0), xycoords='axes fraction',
            fontsize=12, ha='right', va='bottom'
        )

        # Default line color for graph edges
        self.line_color = 0.3 * np.ones(3)

        # For connectivity graph
        self.graph_edges = None
        self.graph_lines = None
        self.graph = None

        # --- Trails: buffers and plot handles ---
        self.max_tail = max_tail_length
        self.trails = None          # will become list of deques
        self.trail_plots = None     # list of Line3D objects

    def setGraph(self, edges):
        """Set edges of graph visualization - sequence of (i,j) tuples."""
        n_edges = len(edges)
        if self.graph_edges is None or n_edges != len(self.graph_edges):
            self.graph_lines = np.zeros((n_edges, 2, 3))
        self.graph_edges = edges

        if self.graph is None:
            self.graph = Line3DCollection(self.graph_lines, edgecolor=self.line_color)
            self.ax.add_collection(self.graph)

    def showEllipsoids(self, radii):
        warnings.warn("showEllipsoids not implemented in Matplotlib visualizer.")

    def update(self, t, crazyflies):
        N = len(crazyflies)
        xs, ys, zs = [], [], []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # per-drone colors

        # --- Initialize trails and trail plots on first call ---
        if self.trails is None:
            self.trails = [deque(maxlen=self.max_tail) for _ in range(N)]
            self.trail_plots = []
            for i in range(N):
                line, = self.ax.plot([], [], [], lw=1, color=colors[i])
                self.trail_plots.append(line)

        # Collect current positions and append to each trail
        for i, cf in enumerate(crazyflies):
            x, y, z = cf.position()
            xs.append(x)
            ys.append(y)
            zs.append(z)
            self.trails[i].append((x, y, z))

        # Update main scatter plot of drones
        if self.plot is None:
            self.plot = self.ax.scatter(xs, ys, zs, c=colors[:N])
        else:
            self.plot._offsets3d = (xs, ys, zs)
            self.plot.set_facecolors(colors[:N])
            self.plot.set_edgecolors(colors[:N])
            self.plot._facecolor3d = self.plot.get_facecolor()
            self.plot._edgecolor3d = self.plot.get_edgecolor()

        # --- Update each drone's tail line ---
        for i, line in enumerate(self.trail_plots):
            trail = np.array(self.trails[i])
            if trail.size:
                line.set_data(trail[:, 0], trail[:, 1])
                line.set_3d_properties(trail[:, 2])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        # Update connectivity graph segments if any
        if self.graph is not None:
            for k, (i, j) in enumerate(self.graph_edges):
                self.graph_lines[k, 0, :] = xs[i], ys[i], zs[i]
                self.graph_lines[k, 1, :] = xs[j], ys[j], zs[j]
            self.graph.set_segments(self.graph_lines)

        # Update timestamp
        self.timeAnnotation.set_text(f"{t:.2f} s")

        # Pause briefly to refresh the plot
        plt.pause(1e-4)

    def render(self):
        warnings.warn("Rendering video not supported in VisMatplotlib yet.")
        return None