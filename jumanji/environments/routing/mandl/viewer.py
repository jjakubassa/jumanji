# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from importlib import resources
from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import rotate

from jumanji.environments.routing.mandl.types import State, VehicleBatch
from jumanji.viewer import Viewer


class MandlViewer(Viewer):
    FIGURE_SIZE = (10.0, 10.0)
    NODE_SIZE = 1000
    ICON_SIZE = 0.04
    OFFSET = 0.005
    INITIAL_OFFSET = 0.005  # Initial offset to ensure routes are away from the edge line
    ICON_PATH = "bus-transportation-public-svgrepo-com.png"

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for the Mandl environment.

        Args:
            name: the window name to be used when initializing the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._name = name

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

        self._display: Callable[[plt.Figure], Optional[np.ndarray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        self._precomputed_positions: jnp.ndarray
        self._precomputed_angles: jnp.ndarray

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the given state of the Mandl environment.

        Args:
            state: the environment state to render.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._precompute_positions_and_angles(state.network.nodes, state.network.links)
        self._add_network_state(ax, state)
        if save_path:
            print(f"Saving figure to {save_path}")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig = plt.figure(f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = fig.add_subplot(111)
        plt.close(fig)
        self._prepare_figure(ax)
        self._precompute_positions_and_angles(states[0].network.nodes, states[0].network.links)

        def make_frame(state_index: int) -> None:
            ax.clear()  # Clear the axes to remove previous positions
            self._prepare_figure(ax)  # Re-prepare the figure after clearing
            state = states[state_index]
            self._add_network_state(ax, state)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        plt.close(self._name)

    def _clear_display(self) -> None:
        import IPython.display

        IPython.display.clear_output(True)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    def _precompute_positions_and_angles(self, nodes: jnp.ndarray, links: jnp.ndarray) -> None:
        """Precompute positions and angles between all pairs of nodes."""
        n_nodes = links.shape[0]
        max_time = int(jnp.max(jnp.where(links == jnp.inf, -jnp.inf, links))) + 1

        # Initialize angles matrix (n_nodes x n_nodes)
        self._precomputed_angles = jnp.zeros((n_nodes, n_nodes))

        # Initialize positions array (n_nodes x n_nodes x max_time)
        self._precomputed_positions = jnp.zeros((n_nodes, n_nodes, max_time, 2))

        for from_node_idx in range(n_nodes):
            for to_node_idx in range(n_nodes):
                # Calculate angle between every pair of nodes
                from_node = nodes[from_node_idx]
                to_node = nodes[to_node_idx]
                if from_node_idx != to_node_idx:  # Skip self-loops
                    dx = to_node[0] - from_node[0]
                    dy = to_node[1] - from_node[1]
                    angle = jnp.degrees(np.arctan2(dy, dx)) + 180
                    self._precomputed_angles = self._precomputed_angles.at[
                        from_node_idx, to_node_idx
                    ].set(angle)

                # Store interpolated positions
                if links[from_node_idx, to_node_idx] < jnp.inf:
                    total_time = max(1, int(links[from_node_idx, to_node_idx]))
                    for time_on_edge in range(total_time + 1):
                        interpolation_factor = time_on_edge / total_time
                        pos = from_node + interpolation_factor * (to_node - from_node)
                        self._precomputed_positions = self._precomputed_positions.at[
                            from_node_idx, to_node_idx, time_on_edge
                        ].set(pos)
                else:
                    # For unconnected nodes, store from_node position at time 0
                    self._precomputed_positions = self._precomputed_positions.at[
                        from_node_idx, to_node_idx, 0
                    ].set(from_node)

    def _add_network_state(self, ax: plt.Axes, state: State) -> None:
        """Add the network state to the plot, including nodes, links, vehicles, and passengers."""

        # Get unique routes and assign colors
        unique_routes = {tuple(route[route != -1].tolist()) for route in state.routes.nodes}
        route_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_routes)))
        route_color_dict = dict(zip(unique_routes, route_colors, strict=False))

        # Assign unique offsets for each route, alternating positive and negative
        route_offsets = {
            route: ((-1) ** i) * (self.INITIAL_OFFSET + (i // 2) * self.OFFSET)
            for i, route in enumerate(unique_routes)
        }

        # Plot network links
        for from_node_idx in range(state.network.links.shape[0]):
            for to_node_idx in range(state.network.links.shape[1]):
                if state.network.links[from_node_idx, to_node_idx] < jnp.inf:
                    from_node = state.network.nodes[from_node_idx]
                    to_node = state.network.nodes[to_node_idx]
                    ax.plot(
                        [from_node[0], to_node[0]],
                        [from_node[1], to_node[1]],
                        "gray",
                        alpha=0.5,
                        zorder=1,
                    )

        # Plot route lines with offset
        for route, color in route_color_dict.items():
            route_nodes = list(route)
            routes_labels = "-".join(
                [str(node + 1) for node in route_nodes]
            )  # plot uses 1 based indexing
            route_offset = route_offsets[route]

            for i in range(len(route_nodes) - 1):
                from_node = state.network.nodes[route_nodes[i]]
                to_node = state.network.nodes[route_nodes[i + 1]]
                dx = to_node[0] - from_node[0]
                dy = to_node[1] - from_node[1]
                length = np.sqrt(dx**2 + dy**2)
                perpx = -dy * route_offset / length
                perpy = dx * route_offset / length
                ax.plot(
                    [from_node[0] + perpx, to_node[0] + perpx],
                    [from_node[1] + perpy, to_node[1] + perpy],
                    color=color,
                    linewidth=2,
                    zorder=2,
                    label=f"Route {routes_labels}" if i == 0 else "",
                )

        # Plot nodes
        ax.scatter(
            state.network.nodes[:, 0],
            state.network.nodes[:, 1],
            c="white",
            s=self.NODE_SIZE,
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )

        # Node labels
        for i in range(len(state.network.nodes)):
            ax.annotate(
                str(i + 1),
                (state.network.nodes[i, 0], state.network.nodes[i, 1]),
                ha="center",
                va="center",
                zorder=4,
            )

        # Plot buses on routes
        for vehicle_idx in range(len(state.vehicles.ids)):
            route = tuple(
                state.routes.nodes[state.vehicles.route_ids[vehicle_idx]][
                    state.routes.nodes[state.vehicles.route_ids[vehicle_idx]] != -1
                ].tolist()
            )
            if not route:  # Skip if route is empty
                continue

            pos = self._get_position(state.vehicles, vehicle_idx)
            route_color = route_color_dict[route]

            # Calculate offset
            start_node, end_node = state.vehicles.current_edges[vehicle_idx]
            current_pos = state.network.nodes[start_node]
            next_pos = state.network.nodes[end_node]
            dx, dy = next_pos - current_pos
            length = np.sqrt(dx**2 + dy**2)
            route_offset = route_offsets[route]
            perpx = -dy * route_offset / length
            perpy = dx * route_offset / length

            self._plot_bus(
                ax,
                pos[0] + perpx,
                pos[1] + perpy,
                state.vehicles,
                vehicle_idx,
                route_color,
                state,
            )

        # Plot waiting passengers
        if len(state.passengers.ids) > 0:
            wait_mask = (state.passengers.departure_times <= state.current_time) & (
                state.passengers.statuses == 0
            )
            waiting_origins = state.passengers.origins[wait_mask]
            wait_counts = collections.Counter(waiting_origins.tolist())

            for node, count in wait_counts.items():
                pos = state.network.nodes[node]
                for i in range(count):
                    ax.plot(
                        pos[0] - 0.02,
                        pos[1] + i * 0.0002,
                        marker="o",
                        color="black",
                        markersize=2,
                        zorder=5,
                    )
            ax.plot([], [], "o", color="black", markersize=2, label="Waiting Passengers")

        ax.legend(loc="upper right")
        ax.set_title(f"Transit Network State at t={state.current_time:.1f} min")

    def _get_position(self, vehicles: VehicleBatch, vehicle_idx: int) -> jnp.ndarray:
        start_node = int(vehicles.current_edges[vehicle_idx, 0])
        end_node = int(vehicles.current_edges[vehicle_idx, 1])
        time_on_edge = int(vehicles.times_on_edge[vehicle_idx])
        return self._precomputed_positions[start_node, end_node, time_on_edge]

    def _plot_bus(
        self,
        ax: plt.Axes,
        pos_x: float,
        pos_y: float,
        vehicles: VehicleBatch,
        vehicle_idx: int,
        color: np.ndarray,
        state: State,
        icon_size: float = 0.04,
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> None:
        """Plot a single bus using a PNG icon with offset for alignment on the route line"""
        adjusted_pos_x = pos_x + offset_x
        adjusted_pos_y = pos_y + offset_y

        start_node, end_node = vehicles.current_edges[vehicle_idx]
        angle = self._precomputed_angles[(int(start_node), int(end_node))]

        # Load the bus icon and rotate it
        with resources.path("jumanji.environments.routing.mandl", self.ICON_PATH) as path:
            bus_icon = mpimg.imread(str(path))
            bus_icon = rotate(bus_icon, angle, reshape=False, mode="nearest")
            bus_icon = np.clip(bus_icon, 0, 1)

        # Plot the rotated bus icon
        ax.imshow(
            bus_icon,
            extent=[
                adjusted_pos_x - icon_size / 2,
                adjusted_pos_x + icon_size / 2,
                adjusted_pos_y - icon_size / 2,
                adjusted_pos_y + icon_size / 2,
            ],
            zorder=4,
        )

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            import IPython.display

            IPython.display.display(fig)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray[Any]:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())