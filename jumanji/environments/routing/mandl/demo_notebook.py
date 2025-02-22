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

import marimo

__generated_with = "0.11.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import tqdm

    from jumanji.environments.routing.mandl import Mandl
    from jumanji.environments.routing.mandl.types import PassengerStatus, State

    return Mandl, PassengerStatus, State, jax, mo, plt, tqdm


@app.cell
def _(jax):
    n_steps = 24 * 60 + 200
    key = jax.random.PRNGKey(42)
    return key, n_steps


@app.cell
def _(Mandl, PassengerStatus, jax, key, n_steps, tqdm):
    # Create environment
    env = Mandl(network_name="mandl1", runtime=200, vehicle_capacity=50)

    # Reset environment and get initial state
    state, timestep = env.reset(key)

    print("\nInitial State Info:")
    print(f"Number of vehicles: {state.fleet.num_vehicles}")
    print(f"Number of passengers: {state.passengers.num_passengers}")
    print(f"Number of routes: {state.routes.num_routes}")
    print(f"Network nodes: {state.network.num_nodes}")

    # Print route information
    print("\nRoute Information:")
    for i in range(state.routes.num_routes):
        route = state.routes.stops[i]
        valid_stops = route[route != -1]
        route_type = "Fixed" if state.routes.types[i] == 0 else "Flexible"
        print(f"Route {i + 1} ({route_type}): {' -> '.join(str(s + 1) for s in valid_stops)}")

    # Print vehicle positions
    print("\nVehicle Positions:")
    for i in range(state.fleet.num_vehicles):
        edge = state.fleet.current_edges[i]
        direction = "Forward" if state.fleet.directions[i] == 0 else "Backward"
        time_on_edge = state.fleet.times_on_edge[i]
        print(
            f"Vehicle {i + 1}: Edge {edge[0] + 1}->{edge[1] + 1}, Direction: {direction}, "
            + f"Time on edge: {time_on_edge:.1f}"
        )

    # Print node coordinates
    print("\nNode Coordinates:")
    for i in range(state.network.num_nodes):
        coords = state.network.node_coordinates[i]
        print(f"Node {i + 1}: ({coords[0]:.3f}, {coords[1]:.3f})")

    # Render initial state
    print("\nRendering initial state...")
    # env.render(state, save_path="initial_state.png")

    # Simulate a few steps
    print("\nSimulating steps...")
    states = [state]
    step = jax.jit(env.step)

    for i in tqdm.tqdm(range(n_steps)):
        # For now, just use dummy action (no-op for all flexible routes)
        action = jax.numpy.full(
            state.routes.num_routes,
            state.network.num_nodes,  # no-op action
            dtype=int,
        )

        state, timestep = step(state, action)
        states.append(state)

        print(f"\nStep {i + 1}:")
        print(f"Time: {state.current_time:.1f}")
        print(
            "Not in system passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM).sum()}"
        )
        print(
            "Waiting passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.WAITING).sum()}"
        )
        print(
            "In-vehicle passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.IN_VEHICLE).sum()}"
        )
        print(
            "Completed passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.COMPLETED).sum()}"
        )

        print(f"Sum of waiting times: {state.passengers.time_waiting.sum():.2f}")
        print(f"Sum of in-vehicle times: {state.passengers.time_in_vehicle.sum():.2f}")
        print(
            f"Total Travel time: {state.passengers.time_in_vehicle.sum() + state.passengers.time_waiting.sum():.2f}"
        )

        # Render every few steps
        # if (i + 1) % 3 == 0:
        #     env.render(state)

    # Create animation
    # print("\nCreating animation...")
    # animation = env.animate(states, interval=500, save_path="mandl_simulation.gif")
    return (
        action,
        coords,
        direction,
        edge,
        env,
        i,
        route,
        route_type,
        state,
        states,
        step,
        time_on_edge,
        timestep,
        valid_stops,
    )


@app.cell
def _(env, plt, state):
    env.render(state)
    plt.show()
    return


@app.cell(hide_code=True)
def _(state):
    # Print timing statistics
    print("\nPassenger Statistics:")
    print(f"Number of passengers: {state.passengers.num_passengers}")
    print(f"Average waiting time: {state.passengers.time_waiting.mean():.2f}")
    print(f"Average in-vehicle time: {state.passengers.time_in_vehicle.mean():.2f}")
    print(
        f"Average total time: {(state.passengers.time_waiting + state.passengers.time_in_vehicle).mean():.2f}"
    )
    print(f"Maximum waiting time: {state.passengers.time_waiting.max():.2f}")
    print(f"Maximum in-vehicle time: {state.passengers.time_in_vehicle.max():.2f}")
    print(f"Sum of waiting times: {state.passengers.time_waiting.sum():.2f}")
    print(f"Sum of in-vehicle times: {state.passengers.time_in_vehicle.sum():.2f}")
    print(
        f"Total Travel time: {state.passengers.time_in_vehicle.sum() + state.passengers.time_waiting.sum():.2f}"
    )
    return


@app.cell
def _(plt, state):
    # Create figure for histograms
    plt.figure(figsize=(12, 5))

    # Plot waiting time distribution
    plt.subplot(1, 2, 1)
    plt.hist(state.passengers.time_waiting, bins=30, alpha=0.75)
    plt.title("Distribution of Waiting Times")
    plt.xlabel("Waiting Time")
    plt.ylabel("Frequency")

    # Plot in-vehicle time distribution
    plt.subplot(1, 2, 2)
    plt.hist(state.passengers.time_in_vehicle, bins=30, alpha=0.75)
    plt.title("Distribution of In-Vehicle Times")
    plt.xlabel("In-Vehicle Time")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(env, plt, states):
    env.animate(states[:10], interval=800, save_path="mandl_simulation.mp4")
    plt.show()
    return


@app.cell
def _():
    # busse einfärben in linien farbe

    # Andere visualisierung:
    # farbe der busse zeigt auslastung
    # pfeile zwischen OD zeigt demand
    # pfeile zwischen OD zeigt total travel time
    # pfeile zwischen OD zeigt kürzeste route
    # pfeile zwischen OD zeigt vehältnis von actual total travel time zu optimaler route
    return


@app.cell(hide_code=True)
def _(State, plt, state):
    from typing import Optional

    import jax.numpy as jnp
    import matplotlib.colors as mcolors
    import numpy as np
    import svgwrite
    from svgwrite import rgb

    from jumanji.environments.routing.mandl.types import VehicleDirection

    def create_network_svg(
        state: State, width: int = 600, height: int = 600, margin: int = 50
    ) -> svgwrite.Drawing:
        """Create an SVG visualization of the network."""
        dwg = svgwrite.Drawing(size=(width, height))

        # Add white background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

        # Get node coordinates
        nodes = np.array(state.network.node_coordinates)
        x_min, y_min = nodes.min(axis=0)
        x_max, y_max = nodes.max(axis=0)

        def scale_for_svg(points: np.ndarray) -> np.ndarray:
            """Scale array of points to SVG coordinates"""
            scaled = np.zeros_like(points, dtype=float)
            scaled[:, 0] = margin + (points[:, 0] - x_min) / (x_max - x_min) * (width - 2 * margin)
            scaled[:, 1] = margin + (points[:, 1] - y_min) / (y_max - y_min) * (height - 2 * margin)
            return scaled

        scaled_nodes = scale_for_svg(nodes)

        # Create symbols definitions
        defs = dwg.defs

        # Define bus symbol
        bus_symbol = dwg.symbol(id="bus", viewBox="0 0 100 100")
        bus_symbol.add(dwg.rect((6, 3), (6, 3), rx=0, ry=0))
        defs.add(bus_symbol)

        # Draw base edges first
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if state.network.travel_times[i, j] < float("inf"):
                    dwg.add(
                        dwg.line(
                            start=tuple(scaled_nodes[i]),
                            end=tuple(scaled_nodes[j]),
                            stroke="black",
                            stroke_width=2,
                            opacity=1,
                        )
                    )

        # Create route segment symbols
        ROUTE_OFFSET = 3  # pixels
        routes_list = []
        for i in range(state.routes.stops.shape[0]):
            route = state.routes.stops[i]
            valid_stops = route[route != -1]
            routes_list.append(tuple(valid_stops.tolist()))

        unique_routes = list(set(routes_list))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_routes)))
        route_color_dict = dict(zip(unique_routes, colors, strict=False))

        # Assign offsets to unique routes
        route_offsets = {
            route: ((-1) ** i) * (1 + i // 2) * ROUTE_OFFSET
            for i, route in enumerate(unique_routes)
        }

        for route_idx, route in enumerate(routes_list):
            color = route_color_dict[route]
            svg_color = rgb(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            offset = route_offsets[route]

            # Create route symbol
            route_id = f"route_{route_idx}"
            route_symbol = dwg.symbol(id=route_id)

            # Add each segment
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]

                from_pos = scaled_nodes[from_node]
                to_pos = scaled_nodes[to_node]

                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                length = np.sqrt(dx**2 + dy**2)
                angle = np.degrees(np.arctan2(dy, dx))

                # Calculate perpendicular offset
                if length > 0:
                    perpx = -dy / length * offset
                    perpy = dx / length * offset

                    # Apply offset to positions
                    from_pos_offset = from_pos + np.array([perpx, perpy])
                    to_pos_offset = to_pos + np.array([perpx, perpy])

                    # Draw the route segment directly instead of using edge symbol
                    route_symbol.add(
                        dwg.line(
                            start=tuple(from_pos_offset),
                            end=tuple(to_pos_offset),
                            stroke=svg_color,
                            stroke_width=2,
                        )
                    )

            defs.add(route_symbol)

            # Add route instance to main drawing
            dwg.add(dwg.use(f"#route_{route_idx}"))

        # Draw nodes
        node_radius = 20
        for i, (x, y) in enumerate(scaled_nodes):
            dwg.add(
                dwg.circle(
                    center=(x, y),
                    r=node_radius,
                    fill="white",
                    stroke="black",
                    stroke_width=2,
                )
            )
            dwg.add(
                dwg.text(
                    str(i + 1),
                    insert=(x, y + 6),
                    text_anchor="middle",
                    font_size=14,
                )
            )

        return dwg

    create_network_svg(state)
    return (
        Optional,
        VehicleDirection,
        create_network_svg,
        jnp,
        mcolors,
        np,
        rgb,
        svgwrite,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
