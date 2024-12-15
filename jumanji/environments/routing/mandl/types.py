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

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp


class NetworkData(NamedTuple):
    """Represents the physical network structure and properties"""

    nodes: jnp.ndarray
    """Coordinates of each node in the network. Shape: (n_nodes, 2)"""

    links: jnp.ndarray
    """Travel times between nodes in the network. Shape: (n_nodes, n_nodes)"""

    terminals: jnp.ndarray
    """Boolean array indicating whether each node is a terminal. Shape: (n_nodes,)"""


class RouteBatch(NamedTuple):
    """Batch of routes for vectorized operations"""

    ids: jnp.ndarray
    """Unique identifiers for each route. Shape: (n_routes,)"""

    nodes: jnp.ndarray
    """Sequence of node indices defining each route. Shape: (n_routes, max_route_length)"""

    frequencies: jnp.ndarray
    """Number of vehicles assigned to each route. Shape: (n_routes,)"""

    on_demand: jnp.ndarray
    """Boolean array indicating if each route is on-demand. Shape: (n_routes,)"""


class PassengerBatch(NamedTuple):
    """Batch of passengers for vectorized operations"""

    ids: jnp.ndarray
    """Unique identifiers for each passenger"""

    origins: jnp.ndarray
    """Origin nodes for each passenger"""

    destinations: jnp.ndarray
    """Destination nodes for each passenger"""

    departure_times: jnp.ndarray
    """Desired departure times for each passenger"""

    time_waiting: jnp.ndarray
    """Accumulated waiting time for each passenger"""

    time_in_vehicle: jnp.ndarray
    """Accumulated in-vehicle time for each passenger"""

    statuses: jnp.ndarray
    """Current status of each passenger (e.g., not in system, waiting, in-vehicle, completed)"""


class VehicleBatch(NamedTuple):
    """Batch of vehicles for vectorized operations"""

    ids: jnp.ndarray
    """Unique identifiers for each vehicle. Shape: (n_vehicles,)"""

    route_ids: jnp.ndarray
    """Route line numbers assigned to each vehicle. Shape: (n_vehicles,)"""

    current_edges: jnp.ndarray
    """Current edge (from_node, to_node) for each vehicle. Shape: (n_vehicles, 2)"""

    times_on_edge: jnp.ndarray
    """Time spent by each vehicle on their current edge. Shape: (n_vehicles,)"""

    passengers: jnp.ndarray
    """Passenger IDs currently in each vehicle. Shape: (n_vehicles, max_capacity)"""

    capacities: jnp.ndarray
    """Maximum passenger capacity of each vehicle. Shape: (n_vehicles,)"""

    directions: jnp.ndarray
    """Direction of travel for each vehicle (1 forward, -1 backward). Shape: (n_vehicles,)"""


@dataclass
class State:
    """Represents the complete state of the routing environment"""

    network: NetworkData
    """Network structure and properties containing nodes, links, and terminals"""

    vehicles: VehicleBatch
    """Current state of all vehicles in the system"""

    routes: RouteBatch
    """Current route definitions and assignments"""

    passengers: PassengerBatch
    """Current state of all passengers in the system"""

    current_time: int
    """Current simulation timestep"""

    key: chex.PRNGKey  # (2,)
    """Random number generator key for stochastic operations"""


class Observation(NamedTuple):
    """Represents the observable state of the environment that agents can use for decision making"""

    network: jnp.ndarray
    """Adjacency matrix representing the network connections. Shape: (n_nodes, n_nodes)"""

    travel_times: jnp.ndarray
    """Matrix of shortest path travel times between nodes via direct or transfer connections.
    Shape: (n_nodes, n_nodes)"""

    routes: jnp.ndarray
    """Sequence of node indices defining each route. Shape: (n_routes, max_route_length)"""

    origins: jnp.ndarray
    """Origin nodes for each passenger"""

    destinations: jnp.ndarray
    """Destination nodes for each passenger"""

    departure_times: jnp.ndarray
    """Departure times for each passenger"""

    time_waiting: jnp.ndarray
    """Waiting time accumulated by each passenger"""

    time_in_vehicle: jnp.ndarray
    """In-vehicle time accumulated by each passenger"""

    statuses: jnp.ndarray
    """Current status of each passenger"""

    route_ids: jnp.ndarray
    """Route line numbers assigned to each vehicle. Shape: (n_vehicles,)"""

    current_edges: jnp.ndarray
    """Current edge (from_node, to_node) for each vehicle. Shape: (n_vehicles, 2)"""

    times_on_edge: jnp.ndarray
    """Time spent by each vehicle on their current edge. Shape: (n_vehicles,)"""

    capacities: jnp.ndarray
    """Capacity of each vehicle. Shape: (n_vehicles,)"""

    directions: jnp.ndarray
    """Direction of travel for each vehicle. Shape: (n_vehicles,)"""

    frequencies: jnp.ndarray
    """Number of vehicles assigned to each route. Shape: (n_routes,)"""

    on_demand: jnp.ndarray
    """Boolean array indicating if each route is an on-demand route. Shape: (n_routes,)"""


class DirectPath(NamedTuple):
    """Represents a direct path using a single route"""

    route: int
    """ID of the route used for this path"""

    cost: float
    """Total travel cost/time for this path"""

    start: int
    """Starting node ID"""

    end: int
    """Ending node ID"""


class TransferPath(NamedTuple):
    """Represents a path that requires transferring between two routes"""

    first_route: int
    """ID of the first route in the transfer path"""

    second_route: int
    """ID of the second route in the transfer path"""

    total_cost: float
    """Total travel cost/time including transfer"""

    transfer_stop: int
    """Node ID where the transfer occurs"""

    start: int
    """Starting node ID"""

    end: int
    """Ending node ID"""
