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
from functools import partial
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


def create_initial_fleet(
    num_routes: int,
    num_flex_routes: int,
    total_vehicles: int,
    vehicles_per_solution_route: list[int],
    vehicle_capacity: int,
) -> tuple[Fleet, tuple[int, ...]]:
    """Create initial fleet and determine vehicle assignments."""
    # Create vehicles_per_route array
    vehicles_per_route = []

    # Track remaining vehicles
    remaining_vehicles = total_vehicles

    # First, allocate solution routes
    num_fix_routes = num_routes - num_flex_routes
    for i in range(len(vehicles_per_solution_route)):
        vehicles = vehicles_per_solution_route[i]
        vehicles_per_route.append(vehicles)
        remaining_vehicles -= vehicles

    # Then, allocate flex routes
    # If we only have flex routes, distribute remaining vehicles among them
    if num_fix_routes == 0:
        vehicles_per_flex = remaining_vehicles // num_flex_routes
        leftover_vehicles = remaining_vehicles % num_flex_routes

        for i in range(num_flex_routes):
            vehicles = vehicles_per_flex
            if i < leftover_vehicles:
                vehicles += 1
            vehicles_per_route.append(vehicles)
    else:
        # Otherwise, give each flex route 1 vehicle
        vehicles_per_route.extend([1] * num_flex_routes)
        remaining_vehicles -= num_flex_routes

        # Distribute remaining vehicles among remaining fixed routes
        if num_fix_routes > 0:
            vehicles_per_remaining = remaining_vehicles // num_fix_routes
            leftover_vehicles = remaining_vehicles % num_fix_routes

            for i in range(num_fix_routes):
                vehicles = vehicles_per_remaining
                if i < leftover_vehicles:
                    vehicles += 1
                vehicles_per_route.append(vehicles)

    assert (
        jnp.sum(jnp.array(vehicles_per_route)) == total_vehicles
    ), f"Vehicle allocation mismatch: {jnp.sum(vehicles_per_route)} != {total_vehicles}"

    # Create initial fleet with total vehicles
    initial_fleet = Fleet(
        route_ids=jnp.full((total_vehicles,), -1, dtype=jnp.int32),
        current_edges=jnp.full((total_vehicles, 2), -1, dtype=jnp.int32),
        times_on_edge=jnp.zeros((total_vehicles,), dtype=jnp.float32),
        passengers=jnp.full((total_vehicles, vehicle_capacity), -1, dtype=jnp.int32),
        directions=jnp.zeros((total_vehicles,), dtype=jnp.int32),
    )

    return initial_fleet, tuple(vehicles_per_route)


@partial(jax.jit, static_argnums=(3,))
def assign_routes_to_fleet(
    fleet: Fleet,
    route_batch: RouteBatch,
    network_data: NetworkData,
    vehicles_per_route: tuple[int, ...],
    max_route_length: int,
) -> Fleet:
    """Assign routes to unassigned fleet using predefined vehicle allocations."""
    num_routes = len(route_batch.types)
    route_stops = route_batch.stops

    # Calculate travel times for each edge in each route
    from_nodes = route_stops[:, :-1]
    to_nodes = route_stops[:, 1:]
    valid_edges = (from_nodes != -1) & (to_nodes != -1)
    edge_times = jnp.where(valid_edges, network_data.travel_times[from_nodes, to_nodes], 0.0)

    # Calculate cumulative times along each route
    cumsum_times = jnp.cumsum(edge_times, axis=1)
    route_total_times = jnp.sum(edge_times, axis=1)

    def process_fixed_route(
        route_idx: int,
        n_vehicles: int,
        total_time: Float[Array, ""],
        route_cumsum: Array,
        route_stops_single: Array,
    ) -> tuple[Array, ...]:
        """Process a single route with known number of vehicles."""
        # Calculate vehicle positions
        vehicle_times = jnp.linspace(0.0, total_time * 2, n_vehicles)  # twice for back and forward
        is_backward = vehicle_times >= total_time
        vehicle_times = jnp.where(is_backward, vehicle_times - total_time, vehicle_times)

        edge_indices = jnp.sum(vehicle_times[:, None] > route_cumsum[None, :], axis=1)

        # Get edges for vehicles
        from_nodes = route_stops_single[edge_indices]
        to_nodes = route_stops_single[edge_indices + 1]
        current_edges = jnp.stack([from_nodes, to_nodes], axis=1)

        # Calculate times on edge
        prev_cumsum = jnp.where(edge_indices > 0, route_cumsum[edge_indices - 1], 0.0)
        times_on_edge = vehicle_times - prev_cumsum

        # Create route assignments and directions
        route_ids = jnp.full(n_vehicles, route_idx, dtype=jnp.int32)
        directions = jnp.full(n_vehicles, VehicleDirection.FORWARD, dtype=jnp.int32)

        return route_ids, current_edges, times_on_edge, directions

    def process_flex_route(
        route_idx: int,
        n_vehicles: int,  # Add n_vehicles parameter
        route_stops_single: Array,
    ) -> tuple[Array, ...]:
        """Process a flexible route, initializing all vehicles at the start."""
        # Get edge for vehicles
        _from_node = route_stops_single[0]
        _to_node = route_stops_single[1]

        # Create arrays with proper shapes for n_vehicles
        route_ids = jnp.full(n_vehicles, route_idx, dtype=jnp.int32)
        current_edges = jnp.tile(jnp.array([[_from_node, _to_node]]), (n_vehicles, 1))
        times_on_edge = jnp.zeros(n_vehicles, dtype=jnp.float32)
        directions = jnp.full(n_vehicles, VehicleDirection.FORWARD, dtype=jnp.int32)

        return route_ids, current_edges, times_on_edge, directions

    # Process each route separately and concatenate results
    all_route_ids = []
    all_current_edges = []
    all_times_on_edge = []
    all_directions = []

    for i in range(num_routes):
        # Bind values to avoid loop variable capture issues
        route_idx = i
        n_vehicles = vehicles_per_route[i]
        total_time = route_total_times[i]
        cumsum = cumsum_times[i]
        stops = route_stops[i]

        route_ids, current_edges, times_on_edge, directions = jax.lax.cond(
            route_batch.types[i] == RouteType.FIXED,
            lambda route_idx=route_idx,
            n_vehicles=n_vehicles,
            total_time=total_time,
            cumsum=cumsum,
            stops=stops: process_fixed_route(route_idx, n_vehicles, total_time, cumsum, stops),
            lambda route_idx=route_idx, n_vehicles=n_vehicles, stops=stops: process_flex_route(
                route_idx, n_vehicles, stops
            ),
        )

        all_route_ids.append(route_ids)
        all_current_edges.append(current_edges)
        all_times_on_edge.append(times_on_edge)
        all_directions.append(directions)

    # Concatenate all results
    return replace(
        fleet,
        route_ids=jnp.concatenate(all_route_ids),
        current_edges=jnp.concatenate(all_current_edges),
        times_on_edge=jnp.concatenate(all_times_on_edge),
        directions=jnp.concatenate(all_directions),
    )


def create_initial_routes(
    solution_routes: list[list[int]],
    num_fix_routes: int,  # Additional fixed routes beyond solution routes
    num_flex_routes: int,
    network_data: NetworkData,
    max_stops: int,
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
    print("\nDEBUG: Creating initial routes:")
    print(f"Number of solution routes: {len(solution_routes)}")
    print(f"Number of additional fixed routes: {num_fix_routes}")
    print(f"Number of flexible routes: {num_flex_routes}")

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
    for i, route in enumerate(solution_routes):
        padded = route + [-1] * (max_length - len(route))
        print(f"Solution route {i}: {route}")
        padded_routes.append(padded)

    # Add additional fixed routes (empty initially)
    for i in range(num_fix_routes):
        padded = [-1] * max_length
        print(f"Additional fixed route {i}: empty")
        padded_routes.append(padded)

    # Add flexible routes (empty initially)
    for i in range(num_flex_routes):
        padded = [-1] * max_length
        print(f"Flexible route {i}: empty")
        padded_routes.append(padded)

    # Create route types array
    route_types = jnp.concatenate(
        [jnp.full(total_fix_routes, RouteType.FIXED), jnp.full(num_flex_routes, RouteType.FLEXIBLE)]
    )

    route_batch = RouteBatch(
        types=route_types.astype(jnp.int32),
        stops=jnp.array(padded_routes, dtype=jnp.int32),
        frequencies=jnp.ones(total_routes, dtype=jnp.float32),
        num_flex_routes=jnp.array(num_flex_routes),
        num_fix_routes=jnp.array(total_fix_routes),  # Total fixed routes including solution
    )

    print("\nDEBUG: Created RouteBatch:")
    print(f"Number of routes: {route_batch.num_routes}")
    print(f"Number of fixed routes (including solution): {route_batch.num_fix_routes}")
    print(f"Number of flexible routes: {route_batch.num_flex_routes}")
    print(f"Route types: {route_batch.types}")
    print(f"Stops:\n{route_batch.stops}")

    return route_batch
