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

# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licensxe is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import resources
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float

from jumanji.environments.routing.mandl.types import (
    Fleet,
    NetworkData,
    Passengers,
    PassengerStatus,
    RouteBatch,
    RouteType,
    VehicleDirection,
)


def load_demand_data(network_name: str) -> pd.DataFrame:
    """Load demand data from the demand file."""
    assets_package = f"jumanji.environments.routing.mandl.assets.{network_name}"

    with resources.files(assets_package).joinpath(f"{network_name}_demand.txt").open("r") as f:
        demand_df = pd.read_csv(f)

    return demand_df


def load_network_data(network_name: str) -> NetworkData:
    """Load network data from files in the assets directory.

    Args:
        network_name: Name of the network (e.g., 'ceder1', 'mandl1')

    Returns:
        NetworkData instance containing the loaded network information with normalized coordinates
    """
    # Define the package path for assets
    assets_package = f"jumanji.environments.routing.mandl.assets.{network_name}"

    # Define filenames
    nodes_file = f"{network_name}_nodes.txt"
    links_file = f"{network_name}_links.txt"

    # Load nodes data
    with resources.files(assets_package).joinpath(nodes_file).open("r") as f:
        nodes_df = pd.read_csv(f)
    num_nodes = len(nodes_df)

    # Create node coordinates array and normalize to [0,1]
    node_coordinates = jnp.array(nodes_df[["lat", "lon"]].values, dtype=float)

    # First shift each axis by its minimum to make all coordinates positive
    mins = node_coordinates.min(axis=0)
    node_coordinates = node_coordinates - mins

    # Then normalize by the maximum value across both dimensions
    max_coord = node_coordinates.max()
    node_coordinates = node_coordinates / max_coord

    # Create is_terminal array
    is_terminal = jnp.array(nodes_df["terminal"].values, dtype=bool)

    # Load links data
    with resources.files(assets_package).joinpath(links_file).open("r") as f:
        links_df = pd.read_csv(f)

    # Initialize travel times matrix with infinity
    travel_times = jnp.full((num_nodes, num_nodes), jnp.inf)

    # Set diagonal to 0 (travel time to same node is 0)
    travel_times = travel_times.at[jnp.arange(num_nodes), jnp.arange(num_nodes)].set(0.0)

    # Fill in travel times from links data
    # Subtract 1 from indices since the file uses 1-based indexing
    for _, row in links_df.iterrows():
        from_node = int(row["from"]) - 1
        to_node = int(row["to"]) - 1
        travel_time = float(row["travel_time"])
        travel_times = travel_times.at[from_node, to_node].set(travel_time)

    return NetworkData(
        node_coordinates=node_coordinates,
        travel_times=travel_times,
        is_terminal=is_terminal,
    )


def create_initial_passengers(
    demand_data: pd.DataFrame,
    key: chex.PRNGKey,
    runtime: float = 60.0,
    deterministic: bool = True,
    mode: Literal["evenly_spaced", "rush_hour", "uniform_random", "all_at_start"] = "evenly_spaced",
) -> Passengers:
    """
    Create initial passengers based on demand data with deterministic or random departure times.
    """
    origins = []
    destinations = []

    for _, row in demand_data.iterrows():
        num_passengers = int(row["demand"])
        origin = int(row["from"]) - 1
        destination = int(row["to"]) - 1

        origins.extend([origin] * num_passengers)
        destinations.extend([destination] * num_passengers)

    origins = jnp.array(origins, dtype=jnp.int32)
    destinations = jnp.array(destinations, dtype=jnp.int32)
    num_passengers = len(origins)

    # Split the key for different random operations
    key, shuffle_key = jax.random.split(key)

    # Shuffle the OD pairs together
    shuffle_indices = jax.random.permutation(shuffle_key, num_passengers)
    origins = origins[shuffle_indices]
    destinations = destinations[shuffle_indices]

    if mode == "evenly_spaced":
        # Evenly space departure times
        desired_departure_times = jnp.linspace(0.0, runtime * 0.8, num_passengers, dtype=jnp.int32)
    elif mode == "rush_hour":
        # Create a bimodal distribution with two Gaussian components
        # Key parameters for morning and evening rush hours

        # Morning rush: centered at 25% of runtime
        morning_center = runtime * 0.25
        morning_std = runtime * 0.07  # Spread of morning rush

        # Evening rush: centered at 70% of runtime
        evening_center = runtime * 0.70
        evening_std = runtime * 0.09  # Evening rush typically has a wider spread

        # Relative weights - morning rush is typically heavier
        morning_weight = 0.6  # 60% of passengers in morning rush

        # Determine which rush hour each passenger belongs to
        is_morning_commuter = (
            jax.random.uniform(jax.random.split(key)[0], (num_passengers,)) < morning_weight
        )

        # Generate normal distributions for each rush hour
        morning_noise = jax.random.normal(jax.random.split(key)[1], (num_passengers,)) * morning_std
        evening_noise = jax.random.normal(jax.random.split(key)[2], (num_passengers,)) * evening_std

        # Combine into final distribution
        morning_times = morning_center + morning_noise
        evening_times = evening_center + evening_noise
        desired_departure_times = jnp.where(is_morning_commuter, morning_times, evening_times)

        # Ensure times are within valid range
        desired_departure_times = jnp.clip(desired_departure_times, 0.0, runtime)
    elif mode == "uniform_random":
        # uniform random departure times
        desired_departure_times = jax.random.uniform(
            key, shape=(num_passengers,), minval=0.0, maxval=runtime
        )
    elif mode == "all_at_start":
        desired_departure_times = jnp.zeros(num_passengers, dtype=jnp.float32)
    else:
        raise NotImplementedError

    # Sort passengers by departure time
    sort_indices = jnp.argsort(desired_departure_times)
    origins = origins[sort_indices]
    destinations = destinations[sort_indices]
    desired_departure_times = desired_departure_times[sort_indices]

    time_waiting = jnp.zeros(num_passengers)
    time_in_vehicle = jnp.zeros(num_passengers)
    statuses = jnp.full(num_passengers, PassengerStatus.NOT_IN_SYSTEM)

    return Passengers(
        origins=origins,
        destinations=destinations,
        desired_departure_times=desired_departure_times,
        time_waiting=time_waiting,
        time_in_vehicle=time_in_vehicle,
        statuses=statuses,
        has_transferred=jnp.zeros(num_passengers, dtype=bool),
        transfer_nodes=jnp.full(num_passengers, -1, dtype=jnp.int32),
    )


def load_solution_data(network_name: str) -> tuple[list[list[int]], list[int]]:
    assets_package = f"jumanji.environments.routing.mandl.assets.{network_name}"
    solution_file = f"{network_name}_solution.txt"

    with resources.files(assets_package).joinpath(solution_file).open("r") as f:
        lines = [line.strip() for line in f.readlines()]

    print("\nDEBUG: Solution file contents:")
    for i, line in enumerate(lines):
        print(f"Line {i}: {line}")

    # First line is name/description
    # Second line is number of routes
    num_routes = int(lines[1])
    print(f"\nDEBUG: Number of routes specified: {num_routes}")

    # Next num_routes lines are the routes
    routes = []
    for i in range(num_routes):
        # Convert from 1-based to 0-based indexing
        line = lines[i + 2]
        route = [int(node) - 1 for node in line.split("-")]
        routes.append(route)
        print(f"DEBUG: Loaded route {i}: {route}")

    # Read vehicles per route from the subsequent lines
    vehicles_per_route = []
    for i in range(num_routes):
        vehicles = int(lines[2 + num_routes + i])
        vehicles_per_route.append(vehicles)
        print(f"DEBUG: Route {i} vehicles: {vehicles}")

    if not routes:
        raise ValueError("No routes found in solution file")

    print("\nDEBUG: Final loaded data:")
    print(f"Number of routes loaded: {len(routes)}")
    print(f"Number of vehicle assignments: {len(vehicles_per_route)}")
    for i, (route, vehicles) in enumerate(zip(routes, vehicles_per_route, strict=False)):
        print(f"Route {i}: {route} with {vehicles} vehicles")

    return routes, vehicles_per_route


def calculate_route_total_time(route: list[int], travel_times: jnp.ndarray) -> Float[Array, ""]:
    """Calculate total travel time for a route including return journey."""
    # Forward journey
    forward_time = sum(travel_times[route[i], route[i + 1]] for i in range(len(route) - 1))

    # Return journey
    backward_time = sum(travel_times[route[i], route[i - 1]] for i in range(len(route) - 1, 0, -1))

    return jnp.array(forward_time + backward_time)


def map_route_time_to_position(
    route: list[int], travel_times: jnp.ndarray, target_time: jnp.ndarray
) -> tuple[tuple[int, int], jnp.ndarray, int]:
    """Maps a time point to position on route.

    Args:
        route: List of node indices in the route
        travel_times: Matrix of travel times between nodes
        target_time: Time point to map to position

    Returns:
        tuple of (current_edge, time_on_edge, direction)
        where current_edge is (from_node, to_node)
    """
    route_length = len(route)
    accumulated_time = jnp.array(0.0)

    # Forward journey
    for i in range(route_length - 1):
        edge_time = travel_times[route[i], route[i + 1]]
        if accumulated_time + edge_time > target_time:
            return (
                (route[i], route[i + 1]),
                target_time - accumulated_time,
                VehicleDirection.FORWARD,
            )
        accumulated_time += edge_time

    # Return journey
    for i in range(route_length - 1, 0, -1):
        edge_time = travel_times[route[i], route[i - 1]]
        if accumulated_time + edge_time > target_time:
            return (
                (route[i], route[i - 1]),
                target_time - accumulated_time,
                VehicleDirection.BACKWARDS,
            )
        accumulated_time += edge_time

    # If we get here, target_time exceeds total route time
    # Wrap around to start of route
    return map_route_time_to_position(route, travel_times, target_time % accumulated_time)


def create_initial_fleet(
    routes: list[list[int]],
    vehicles_per_route: list[int],  # Now a list
    travel_times: jnp.ndarray,
    vehicle_capacity: int = 20,
) -> Fleet:
    """Create initial fleet with vehicles distributed along their routes."""
    # Initialize lists for fleet attributes
    route_ids = []
    current_edges = []
    times_on_edge = []
    directions = []

    # Distribute vehicles for each route
    for route_id, (route, num_vehicles) in enumerate(zip(routes, vehicles_per_route, strict=False)):
        # Calculate total route time (round trip)
        total_time = calculate_route_total_time(route, travel_times)

        # Calculate forward and backward segment times
        forward_segments = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        backward_segments = [(route[i], route[i - 1]) for i in range(len(route) - 1, 0, -1)]

        forward_times = [travel_times[src, dst] for src, dst in forward_segments]
        backward_times = [travel_times[src, dst] for src, dst in backward_segments]

        # Combine segments and their times
        all_segments = forward_segments + backward_segments
        all_times = forward_times + backward_times
        cumulative_times = jnp.cumsum(jnp.array(all_times))

        # Distribute vehicles evenly across total time
        for vehicle_idx in range(num_vehicles):  # Use the specific number for this route
            # Calculate target time for this vehicle
            target_time = jnp.round((vehicle_idx * total_time) / num_vehicles)

            # Find which segment this time falls into
            segment_idx = jnp.searchsorted(cumulative_times, target_time)
            if segment_idx == len(all_segments):
                segment_idx = 0
                target_time = 0

            # Calculate time within the segment
            prev_time = cumulative_times[segment_idx - 1] if segment_idx > 0 else 0
            time_on_edge = target_time - prev_time

            # Get segment and direction
            is_forward = segment_idx < len(forward_segments)
            edge = all_segments[segment_idx]

            route_ids.append(route_id)
            current_edges.append(edge)
            times_on_edge.append(float(time_on_edge))
            directions.append(
                VehicleDirection.FORWARD if is_forward else VehicleDirection.BACKWARDS
            )

    # Convert to JAX arrays
    fleet = Fleet(
        route_ids=jnp.array(route_ids, dtype=jnp.int32),
        current_edges=jnp.array(current_edges, dtype=jnp.int32),
        times_on_edge=jnp.array(times_on_edge, dtype=jnp.float32),
        passengers=jnp.full((len(route_ids), vehicle_capacity), -1, dtype=jnp.int32),
        directions=jnp.array(directions, dtype=jnp.int32),
    )

    return fleet


def create_initial_routes(
    routes: list[list[int]], num_flex_routes: int, max_stops: int
) -> RouteBatch:
    """Create initial routes based on solution routes."""
    print("\nDEBUG: Creating initial routes:")
    print(f"Number of input routes: {len(routes)}")
    for i, route in enumerate(routes):
        print(f"Route {i}: {route}")

    num_fix_routes = len(routes)
    total_routes = num_fix_routes + num_flex_routes

    # Find maximum route length (considering both solution and max_stops)
    max_solution_length = max(len(route) for route in routes)
    max_length = max(max_solution_length, max_stops)
    print(
        f"""
        Maximum route length: {max_length} (solution: {max_solution_length}, specified: {max_stops})
        """
    )

    # Create padded routes array for all routes (fixed + flexible)
    padded_routes = []

    # Add fixed routes
    for route in routes:
        padded = route + [-1] * (max_length - len(route))
        padded_routes.append(padded)
        print(f"Padded fixed route: {padded}")

    # Add empty flexible routes
    for _ in range(num_flex_routes):
        padded_routes.append([-1] * max_length)

    # Create route types array
    route_types = jnp.concatenate(
        [jnp.full(num_fix_routes, RouteType.FIXED), jnp.full(num_flex_routes, RouteType.FLEXIBLE)]
    )

    route_batch = RouteBatch(
        types=route_types.astype(jnp.int32),
        stops=jnp.array(padded_routes, dtype=jnp.int32),
        frequencies=jnp.ones(total_routes, dtype=jnp.float32),
        num_flex_routes=jnp.array(num_flex_routes),
        num_fix_routes=jnp.array(num_fix_routes),
    )

    print("\nDEBUG: Created RouteBatch:")
    print(f"Number of routes: {route_batch.num_routes}")
    print(f"Route types: {route_batch.types}")
    print(f"Stops:\n{route_batch.stops}")

    return route_batch
