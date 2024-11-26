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

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp


class NetworkData(NamedTuple):
    nodes: jnp.ndarray  # shape: (n_nodes, 2) for coordinates
    links: jnp.ndarray  # shape: (n_nodes, n_nodes) for travel times between nodes
    demand: jnp.ndarray  # shape: (n_nodes, n_nodes) demand matrix
    terminals: jnp.ndarray  # shape: (n_nodes,) boolean array
    icon_path: str  # path to the bus icon image


class Route(NamedTuple):
    nodes: jnp.ndarray  # sequence of node indices
    frequency: int  # trips per hour
    capacity: int  # passengers per vehicle


class Passenger(NamedTuple):
    id: int
    origin: int  # node index
    destination: int  # node index
    departure_time: float
    status: int  # 0: waiting, 1: in_vehicle, 2: delivered


class PassengerBatch(NamedTuple):
    """Batch of passengers for vectorized operations"""

    ids: jnp.ndarray
    origins: jnp.ndarray
    destinations: jnp.ndarray
    departure_times: jnp.ndarray
    statuses: jnp.ndarray


class Vehicle(NamedTuple):
    id: int
    route: Route
    current_edge: tuple[int, int]  # (start_node, end_node)
    time_on_edge: float  # time spent on the current edge in minutes
    passengers: jnp.ndarray  # passenger IDs
    capacity: int
    next_departure: float


@dataclass
class State:
    network: NetworkData
    vehicles: list[Vehicle]
    passengers: list[Passenger]
    current_time: float
    key: chex.PRNGKey  # (2,)
    save_path: Optional[str] = None


class Observation(NamedTuple):
    something: jnp.ndarray


class DirectPath(NamedTuple):
    route: int
    cost: float
    start: int
    end: int


class TransferPath(NamedTuple):
    first_route: int
    second_route: int
    total_cost: float
    transfer_stop: int
    start: int
    end: int
