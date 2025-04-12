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

from dataclasses import replace
from importlib import resources
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import pandas as pd
from beartype.typing import Optional
from jax._src.prng import PRNGKeyArray
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
    buffer_time_start: float,
    buffer_time_end: float,
    runtime: float = 60.0,
    mode: Literal["evenly_spaced", "rush_hour", "uniform_random", "all_at_start"] = "evenly_spaced",
) -> Passengers:
    """
    Create initial passengers with staggered system entry times.

    Args:
        demand_data: DataFrame containing passenger demand information
        key: Random key for stochastic operations
        runtime: Total runtime of the episode
        buffer_time: Minimum time between last passenger entry and episode end
        mode: Method for distributing passenger entry times
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

    # Calculate the available time window for passenger entries
    available_time = runtime - buffer_time_end - buffer_time_start
    start_time = buffer_time_start

    if mode == "evenly_spaced":
        # Spread entries between grace period and available time
        entry_times = jnp.linspace(start_time, start_time + available_time, num_passengers)

    elif mode == "rush_hour":
        # Adjust rush hour timing to respect grace period and buffer
        morning_center = start_time + (available_time * 0.3)
        morning_std = available_time * 0.1
        evening_center = start_time + (available_time * 0.7)
        evening_std = available_time * 0.1
        morning_weight = 0.6

        # Generate times for morning and evening rush
        key1, key2, key3 = jax.random.split(key, 3)
        is_morning = jax.random.uniform(key1, (num_passengers,)) < morning_weight

        morning_times = jax.random.normal(key2, (num_passengers,)) * morning_std + morning_center
        evening_times = jax.random.normal(key3, (num_passengers,)) * evening_std + evening_center

        # Combine and clip to ensure grace period and buffer
        entry_times = jnp.where(is_morning, morning_times, evening_times)
        entry_times = jnp.clip(entry_times, start_time, start_time + available_time)

    elif mode == "uniform_random":
        # Random times within available window after grace period
        entry_times = jax.random.uniform(
            key, shape=(num_passengers,), minval=start_time, maxval=start_time + available_time
        )

    elif mode == "all_at_start":
        # All passengers enter after grace period
        entry_times = jnp.full(num_passengers, start_time)

    else:
        raise ValueError(f"Unknown passenger initialization mode: {mode}")

    # Sort passengers by entry time
    sort_indices = jnp.argsort(entry_times)
    origins = origins[sort_indices]
    destinations = destinations[sort_indices]
    entry_times = entry_times[sort_indices]

    # Initialize other passenger attributes
    time_waiting = jnp.zeros(num_passengers)
    time_in_vehicle = jnp.zeros(num_passengers)
    statuses = jnp.full(num_passengers, PassengerStatus.NOT_IN_SYSTEM)

    return Passengers(
        origins=origins,
        destinations=destinations,
        desired_departure_times=entry_times,
        time_waiting=time_waiting,
        time_in_vehicle=time_in_vehicle,
        statuses=statuses,
        has_transferred=jnp.zeros(num_passengers, dtype=bool),
        transfer_nodes=jnp.full(num_passengers, -1, dtype=jnp.int32),
    )


def load_solution_data(
    network_name: str, solution_name: str
) -> tuple[tuple[tuple[int]], tuple[int]]:
    """
    Load solution data from file and return the specified solution.

    Args:
        network_name: Name of the network (e.g., 'mandl1')
        solution_name: Name of the solution to load

    Returns:
        Tuple of (routes, vehicles_per_route)
    """
    assets_package = f"jumanji.environments.routing.mandl.assets.{network_name}"
    solution_file = f"{network_name}_solution.txt"

    with resources.files(assets_package).joinpath(solution_file).open("r") as f:
        content = f.read()

    # Split file into solution sections
    solutions: dict[str, dict[str, list]] = {}
    current_solution = None
    current_section = None
    routes: list[tuple[int, ...]] = []
    vehicles: list[int] = []

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check for solution header
        if line.startswith("[") and line.endswith("]"):
            # Save previous solution if exists
            if current_solution:
                solutions[current_solution] = {"routes": routes.copy(), "vehicles": vehicles.copy()}
            # Start new solution
            current_solution = line[1:-1]
            routes = []
            vehicles = []
            current_section = None
            continue

        # Check for section headers
        if line.startswith("name:"):
            continue
        elif line.startswith("num_routes:"):
            continue
        elif line == "routes:":
            current_section = "routes"
            continue
        elif line == "vehicles:":
            current_section = "vehicles"
            continue

        # Process data based on current section
        if current_section == "routes":
            # Convert from 1-based to 0-based indexing
            route = [int(node) - 1 for node in line.split("-")]
            routes.append(tuple(route))
        elif current_section == "vehicles":
            vehicles.append(int(line))

    # Save last solution
    if current_solution:
        solutions[current_solution] = {"routes": routes, "vehicles": vehicles}

    # Check if requested solution exists
    if solution_name not in solutions:
        raise ValueError(
            f"Solution '{solution_name}' not found. Available solutions: {list(solutions.keys())}"
        )

    solution = solutions[solution_name]
    return tuple(solution["routes"]), tuple(solution["vehicles"])


def calculate_route_total_time(route: list[int], travel_times: jnp.ndarray) -> Float[Array, ""]:
    """Calculate total travel time for a route including return journey."""
    # Forward journey
    forward_time = sum(travel_times[route[i], route[i + 1]] for i in range(len(route) - 1))

    # Return journey
    backward_time = sum(travel_times[route[i], route[i - 1]] for i in range(len(route) - 1, 0, -1))

    return jnp.array(forward_time + backward_time)


def create_initial_fleet(
    num_routes: int,
    num_flex_routes: int,
    total_vehicles: int,
    vehicles_per_solution_route: tuple[int],
    vehicles_per_additional_fixed_route: Optional[tuple[int, ...]],
    vehicle_capacity: int,
) -> tuple[Fleet, jnp.ndarray]:
    """Create initial fleet and determine vehicle assignments."""
    # Create vehicles_per_route array
    vehicles_per_route = []
    remaining_vehicles = total_vehicles

    # First, allocate solution routes if any
    for vehicles in vehicles_per_solution_route:
        vehicles_per_route.append(vehicles)
        remaining_vehicles -= vehicles

    # Then, allocate additional fixed routes if specified
    if vehicles_per_additional_fixed_route is not None:
        for vehicles in vehicles_per_additional_fixed_route:
            vehicles_per_route.append(vehicles)
            remaining_vehicles -= vehicles
    else:
        # If no specific allocation for fixed routes, allocate evenly from remaining vehicles
        num_fixed_routes = num_routes - num_flex_routes - len(vehicles_per_solution_route)
        if num_fixed_routes > 0:
            vehicles_per_fixed = remaining_vehicles // (num_fixed_routes + num_flex_routes)
            for _ in range(num_fixed_routes):
                vehicles_per_route.append(vehicles_per_fixed)
                remaining_vehicles -= vehicles_per_fixed

    # Allocate remaining vehicles evenly among flex routes
    if num_flex_routes > 0:
        vehicles_per_flex = remaining_vehicles // num_flex_routes
        for _ in range(num_flex_routes):
            vehicles_per_route.append(vehicles_per_flex)
            remaining_vehicles -= vehicles_per_flex

        # Add any remaining vehicles to the last flex route
        if remaining_vehicles > 0:
            vehicles_per_route[-1] += remaining_vehicles

    # Create initial fleet with total vehicles
    initial_fleet = Fleet(
        route_ids=jnp.full((total_vehicles,), -1, dtype=jnp.int32),  # Initialize with -1
        current_edges=jnp.zeros((total_vehicles), dtype=jnp.int32),
        times_on_edge=jnp.zeros((total_vehicles,), dtype=jnp.float32),
        passengers=jnp.full((total_vehicles, vehicle_capacity), -1, dtype=jnp.int32),
        directions=jnp.zeros((total_vehicles,), dtype=jnp.int32),
    )

    return initial_fleet, jnp.array(vehicles_per_route)


def assign_routes_to_fleet(
    fleet: Fleet,
    route_batch: RouteBatch,
    network_data: NetworkData,
    vehicles_per_route: jax.Array,
    max_route_length: int,
) -> Fleet:
    """Assign routes to unassigned fleet using predefined vehicle allocations."""
    route_stops = route_batch.stops
    num_vehicles = fleet.num_vehicles

    # Calculate travel times for each edge in each route
    from_nodes = route_stops[:, :-1]
    to_nodes = route_stops[:, 1:]
    valid_edges = (from_nodes != -1) & (to_nodes != -1)
    edge_times = jnp.where(valid_edges, network_data.travel_times[from_nodes, to_nodes], 0.0)

    # Calculate cumulative times along each route
    cumsum_times = jnp.cumsum(edge_times, axis=1)
    route_total_times = jnp.sum(edge_times, axis=1)

    # Calculate cumulative vehicle counts for indexing
    cumsum_vehicles = jnp.cumsum(vehicles_per_route)

    def assign_single_vehicle(carry, vehicle_idx):
        """Modified function to work with scan properly"""
        route_ids, current_edges, times_on_edge, directions = carry

        # Find which route this vehicle belongs to
        route_id = jnp.searchsorted(cumsum_vehicles, vehicle_idx, side="right")

        # Calculate vehicle's position within its route
        vehicle_in_route_idx = vehicle_idx - jnp.where(
            route_id > 0, cumsum_vehicles[route_id - 1], 0
        )

        # Get route information
        route_time = route_total_times[route_id]
        route_cumsum = cumsum_times[route_id]
        is_flexible = route_batch.types[route_id] == RouteType.FLEXIBLE

        # Calculate vehicle position for fixed routes
        vehicles_in_route = vehicles_per_route[route_id]
        position_fraction = vehicle_in_route_idx / vehicles_in_route
        vehicle_time = position_fraction * (2 * route_time)
        is_backward = vehicle_time >= route_time
        vehicle_time = jnp.where(is_backward, vehicle_time - route_time, vehicle_time)

        # Calculate edge and time on edge
        current_edge = jnp.sum(vehicle_time > route_cumsum)
        prev_cumsum = jnp.where(current_edge > 0, route_cumsum[current_edge - 1], 0.0)
        time_on_edge = vehicle_time - prev_cumsum

        # Handle flexible routes
        current_edge = jnp.where(is_flexible, 0, current_edge)
        time_on_edge = jnp.where(is_flexible, 0.0, time_on_edge)
        direction = jnp.where(
            is_flexible,
            VehicleDirection.FORWARD,
            jnp.where(is_backward, VehicleDirection.BACKWARDS, VehicleDirection.FORWARD),
        )

        # Update arrays
        new_route_ids = route_ids.at[vehicle_idx].set(route_id)
        new_current_edges = current_edges.at[vehicle_idx].set(current_edge)
        new_times_on_edge = times_on_edge.at[vehicle_idx].set(time_on_edge)
        new_directions = directions.at[vehicle_idx].set(direction)

        return (new_route_ids, new_current_edges, new_times_on_edge, new_directions), None

    # Initialize arrays
    init_carry = (
        jnp.zeros(num_vehicles, dtype=jnp.int32),  # route_ids
        jnp.zeros(num_vehicles, dtype=jnp.int32),  # current_edges
        jnp.zeros(num_vehicles, dtype=jnp.float32),  # times_on_edge
        jnp.zeros(num_vehicles, dtype=jnp.int32),  # directions
    )

    # Process all vehicles
    (final_route_ids, final_current_edges, final_times_on_edge, final_directions), _ = jax.lax.scan(
        assign_single_vehicle,
        init_carry,
        jnp.arange(num_vehicles),
    )

    # Create updated fleet
    return replace(
        fleet,
        route_ids=final_route_ids,
        current_edges=final_current_edges,
        times_on_edge=final_times_on_edge,
        directions=final_directions,
    )


def create_initial_routes(
    solution_routes: tuple[tuple[int]],
    num_fix_routes: int,  # Additional fixed routes beyond solution routes
    num_flex_routes: int,
    network_data: NetworkData,
    max_stops: int,
    vehicles_per_route: jnp.ndarray,
    key: Optional[PRNGKeyArray] = None,
) -> RouteBatch:
    """Create initial routes combining solution, additional fixed, and flexible routes.

    Args:
        solution_routes: List of predefined route stop sequences from solution
        num_fix_routes: Number of additional fixed routes beyond solution routes
        num_flex_routes: Number of flexible routes to create
        network_data: Network structure data
        max_stops: Maximum number of stops per route
        key: Random key for initialization
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Calculate total routes
    total_fix_routes = len(solution_routes) + num_fix_routes
    total_routes = total_fix_routes + num_flex_routes

    # Find maximum route length
    max_solution_length = max([len(route) for route in solution_routes], default=0)
    max_length = max(max_solution_length, max_stops)

    # Create padded routes array
    padded_routes = []

    # Add solution routes
    for _, route in enumerate(solution_routes):
        padded = list(route) + [-1] * (max_length - len(route))
        padded_routes.append(padded)

    # Add additional fixed routes (empty initially)
    for _ in range(num_fix_routes):
        padded = [-1] * max_length
        padded_routes.append(padded)

    # Add flexible routes (empty initially)
    for _ in range(num_flex_routes):
        padded = [-1] * max_length
        padded_routes.append(padded)

    # Create route types array
    route_types = jnp.concatenate(
        [jnp.full(total_fix_routes, RouteType.FIXED), jnp.full(num_flex_routes, RouteType.FLEXIBLE)]
    )

    route_batch = RouteBatch(
        types=route_types.astype(jnp.int32),
        stops=jnp.array(padded_routes, dtype=jnp.int32),
        vehicles_per_route=vehicles_per_route,
        num_flex_routes=jnp.array(num_flex_routes),
        num_fix_routes=jnp.array(total_fix_routes),  # Total fixed routes including solution
    )
    return route_batch
