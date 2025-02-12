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
from collections.abc import Sequence
from importlib import resources
from typing import Any, Callable, Optional

import jax.numpy as jnp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from numpy.typing import NDArray
from scipy.ndimage import rotate

from jumanji.environments.routing.mandl.types import (
    Fleet,
    NetworkData,
    PassengerStatus,
    State,
    VehicleDirection,
)
from jumanji.viewer import Viewer


class MandlViewer(Viewer):
    FIGURE_SIZE = (10.0, 10.0)
    NODE_SIZE = 1000
    ICON_SIZE = 0.04
    OFFSET = 0.005
    INITIAL_OFFSET = 0.005
    ICON_PATH = "bus-transportation-public-svgrepo-com.png"

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Initialize the viewer.

        Args:
            name: Name of the window
            render_mode: Either "human" or "rgb_array"
        """
        self._name = name
        self._animation: Optional[animation.Animation] = None
        self._network_data: Optional[NetworkData] = None

        self._display: Callable[[plt.Figure], Optional[np.ndarray]]
        if render_mode == "rgb_array":
            self._display = self.display_rgb_array
        elif render_mode == "human":
            self._display = self.display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the current state."""
        self.clear_display()
        fig, ax = self.get_fig_ax()
        ax.clear()
        self.prepare_figure(ax)
        self.precompute_positions_and_angles(
            state.network.node_coordinates, state.network.travel_times
        )
        self.add_network_state(ax, state)
        if save_path:
            print(f"Saving figure to {save_path}")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.Animation:
        """Create an animation from a sequence of states."""
        fig = plt.figure(f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = fig.add_subplot(111)
        plt.close(fig)

        self.prepare_figure(ax)
        self.precompute_positions_and_angles(
            states[0].network.node_coordinates, states[0].network.travel_times
        )

        def make_frame(state_index: int) -> None:
            ax.clear()
            self.prepare_figure(ax)
            state = states[state_index]
            self.add_network_state(ax, state)

        self._animation = animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def clear_display(self) -> None:
        """Clear the current display."""
        try:
            import IPython.display

            IPython.display.clear_output(True)
        except ImportError:
            pass

    def get_fig_ax(self) -> tuple[plt.Figure, plt.Axes]:
        """Get the figure and axes objects."""
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

    def prepare_figure(self, ax: plt.Axes) -> None:
        """Prepare the figure for plotting."""
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    def precompute_positions_and_angles(
        self, node_coordinates: jnp.ndarray, travel_times: jnp.ndarray
    ) -> None:
        """Store network data for position calculations."""
        self._network_data = NetworkData(
            node_coordinates=node_coordinates,
            travel_times=travel_times,
            is_terminal=jnp.zeros(len(node_coordinates), dtype=bool),
        )

    def add_network_state(self, ax: plt.Axes, state: State) -> None:
        """Add the network state to the plot."""
        # Get unique routes and assign colors
        routes_list = []
        for i in range(state.routes.stops.shape[0]):
            route = state.routes.stops[i]
            valid_stops = route[route != -1]
            routes_list.append(tuple(valid_stops.tolist()))

        unique_routes = set(routes_list)
        colors = cm.Set3(np.linspace(0, 1, len(unique_routes)))
        route_color_dict = dict(zip(unique_routes, colors, strict=False))

        # Assign unique offsets for each route
        route_offsets = {
            route: ((-1) ** i) * (self.INITIAL_OFFSET + (i // 2) * self.OFFSET)
            for i, route in enumerate(unique_routes)
        }

        # Plot network links
        nodes = state.network.node_coordinates
        travel_times = state.network.travel_times
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if travel_times[i, j] < jnp.inf:
                    from_node = nodes[i]
                    to_node = nodes[j]
                    ax.plot(
                        [from_node[0], to_node[0]],
                        [from_node[1], to_node[1]],
                        "gray",
                        alpha=0.5,
                        zorder=1,
                    )

        # Plot route lines with offset
        for route in routes_list:
            color = route_color_dict[route]
            routes_labels = "-".join([str(node + 1) for node in route])
            route_offset = route_offsets[route]

            for i in range(len(route) - 1):
                from_node = nodes[route[i]]
                to_node = nodes[route[i + 1]]
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
            nodes[:, 0],
            nodes[:, 1],
            c="white",
            s=self.NODE_SIZE,
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )

        # Node labels
        for i in range(len(nodes)):
            ax.annotate(
                str(i + 1),  # 1-based indexing for display
                (nodes[i, 0], nodes[i, 1]),
                ha="center",
                va="center",
                zorder=4,
            )

        # Plot vehicles
        for vehicle_idx in range(state.fleet.num_vehicles):
            route_id = state.fleet.route_ids[vehicle_idx]
            route = routes_list[route_id]
            pos = self.get_position(state.fleet, vehicle_idx, nodes)
            route_color = route_color_dict[route]

            # Calculate offset
            start_node, end_node = state.fleet.current_edges[vehicle_idx]
            current_pos = nodes[start_node]
            next_pos = nodes[end_node]
            dx, dy = next_pos - current_pos
            length = np.sqrt(dx**2 + dy**2) if start_node != end_node else 1.0
            route_offset = route_offsets[route]
            perpx = -dy * route_offset / length if start_node != end_node else 0.0
            perpy = dx * route_offset / length if start_node != end_node else 0.0

            self.plot_bus(
                ax,
                pos[0] + perpx,
                pos[1] + perpy,
                state.fleet,
                vehicle_idx,
                route_color,
                state,
            )

        # Plot waiting passengers
        if state.passengers.num_passengers > 0:
            wait_mask = state.passengers.statuses == PassengerStatus.WAITING
            waiting_origins = state.passengers.origins[wait_mask]
            wait_counts = collections.Counter(waiting_origins.tolist())

            for node, count in wait_counts.items():
                pos = nodes[node]
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

    def get_position(self, fleet: Fleet, vehicle_idx: int, nodes: jnp.ndarray) -> jnp.ndarray:
        """Get interpolated position of a vehicle."""
        start_node = int(fleet.current_edges[vehicle_idx, 0])
        end_node = int(fleet.current_edges[vehicle_idx, 1])
        if start_node == end_node:
            return nodes[start_node]

        time_on_edge = fleet.times_on_edge[vehicle_idx]
        assert self._network_data is not None
        travel_time = self._network_data.travel_times[start_node, end_node]
        progress = time_on_edge / travel_time

        start_pos = nodes[start_node]
        end_pos = nodes[end_node]
        return start_pos + progress * (end_pos - start_pos)

    def plot_bus(
        self,
        ax: plt.Axes,
        pos_x: float,
        pos_y: float,
        fleet: Fleet,
        vehicle_idx: int,
        color: np.ndarray,
        state: State,
        icon_size: float = 0.04,
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> None:
        """Plot a single bus using a PNG icon with offset for alignment on the route line."""
        adjusted_pos_x = pos_x + offset_x
        adjusted_pos_y = pos_y + offset_y

        start_node, end_node = fleet.current_edges[vehicle_idx]

        # Calculate angle based on direction of travel
        if start_node == end_node:
            # Vehicle is at a node, use previous angle or default
            angle = 0
        else:
            from_pos = state.network.node_coordinates[start_node]
            to_pos = state.network.node_coordinates[end_node]
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            angle = jnp.degrees(np.arctan2(dy, dx)) + 180

            # Adjust angle for backwards direction
            if fleet.directions[vehicle_idx] == VehicleDirection.BACKWARDS:
                angle += 180

        # Load the bus icon and rotate it
        with resources.path("jumanji.environments.routing.mandl.assets", self.ICON_PATH) as path:
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

        # Add passenger count indicator
        passengers_onboard = (fleet.passengers[vehicle_idx] != -1).sum()
        if passengers_onboard > 0:
            ax.text(
                adjusted_pos_x,
                adjusted_pos_y + icon_size / 2,
                str(passengers_onboard),
                color="white",
                backgroundcolor="black",
                ha="center",
                va="bottom",
                zorder=5,
            )

    def display_human(self, fig: plt.Figure) -> None:
        """Display the figure in human-readable format."""
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            try:
                import IPython.display

                IPython.display.clear_output(wait=True)
                IPython.display.display(fig)
            except ImportError:
                plt.show()
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def display_rgb_array(self, fig: plt.Figure) -> NDArray[Any]:
        """Convert the figure to an RGB array."""
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    def close(self) -> None:
        """Close all open figures and clean up resources."""
        plt.close(self._name)
        if self._animation is not None:
            plt.close(self._animation._fig)
