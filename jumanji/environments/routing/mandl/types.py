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

from enum import IntEnum
from typing import TYPE_CHECKING

from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jumanji.types import TimeStep

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

# ruff: noqa: F722


class PassengerStatus(IntEnum):
    """Enum for passenger statuses."""

    NOT_IN_SYSTEM = 0
    WAITING = 1
    IN_VEHICLE = 2
    COMPLETED = 3


class RouteType(IntEnum):
    """Enum for route types."""

    FIXED = 0
    FLEXIBLE = 1


@dataclass
class NetworkData:
    """
    Represents the physical network structure and provides methods for network operations.
    """

    node_coordinates: Float[Array, " num_nodes 2"]
    travel_times: Int[Array, " num_nodes num_nodes"]
    is_terminal: Bool[Array, " num_nodes"]

    def is_connected(self, from_node: int, to_node: int) -> bool:
        """
        Check if two nodes are directly connected.

        Args:
            from_node: Index of the origin node.
            to_node: Index of the destination node.

        Returns:
            True if nodes are connected, False otherwise.
        """
        raise NotImplementedError

    def get_travel_time(self, from_node: int, to_node: int) -> int:
        """
        Get the travel time between two nodes.

        Args:
            from_node: Index of the origin node.
            to_node: Index of the destination node.

        Returns:
            Travel time between the nodes.
        """
        raise NotImplementedError


@dataclass
class RouteBatch:
    """
    Represents a collection of routes.
    """

    ids: Int[Array, " num_routes"]  # unique
    types: Int[Array, " num_routes"]  # dtype: RouteType
    stops: Int[Array, " num_routes max_route_length"]
    frequencies: Float[Array, " num_routes"]
    on_demand: Bool[Array, " num_routes"]

    @property
    def num_on_demand_routes(self) -> Int[Array, ""]:
        raise NotImplementedError

    @property
    def num_fixed_routes(self) -> Int[Array, ""]:
        raise NotImplementedError

    def get_valid_stops(self) -> Bool[Array, " num_routes max_route_length"]:
        """
        Get valid stops for each route.

        Returns:
            Boolean array of shape (num_routes, max_route_length) indicating valid stops.
        """
        raise NotImplementedError

    def update_routes(self, actions: Int[Array, " num_on_demand_routes"]) -> "RouteBatch":
        """
        Update flexible routes based on agent actions.

        Args:
            actions: Array of actions for flexible routes.

        Returns:
            Updated RouteBatch instance.
        """
        raise NotImplementedError


@dataclass
class Fleet:
    """
    Represents the collection of all vehicles in the simulation.
    """

    ids: Int[Array, " num_vehicles"]
    route_ids: Int[Array, " num_vehicles"]
    current_edges: Int[Array, " num_vehicles 2"]
    times_on_edge: Int[Array, " num_vehicles"]
    passengers: Int[Array, " num_vehicles max_capacity"]
    directions: Int[Array, " num_vehicles"]

    def move_vehicles(self, network: "NetworkData", routes: "RouteBatch") -> "Fleet":
        """
        Move all vehicles according to their routes and update positions.

        Args:
            network: NetworkData instance.
            routes: RouteBatch instance.

        Returns:
            Updated Fleet instance with new positions.
        """
        raise NotImplementedError

    def assign_passengers(
        self, passenger_state: "PassengerState", network: "NetworkData", routes: "RouteBatch"
    ) -> tuple["Fleet", "PassengerState"]:
        """
        Assign passengers to vehicles and update passenger statuses.

        Args:
            passenger_state: PassengerState instance.
            network: NetworkData instance.
            routes: RouteBatch instance.

        Returns:
            Tuple containing updated Fleet and PassengerState instances.
        """
        raise NotImplementedError


@dataclass
class PassengerState:
    """
    Represents the state of all passengers.
    """

    ids: Int[Array, " num_passengers"]
    origins: Int[Array, " num_passengers"]
    destinations: Int[Array, " num_passengers"]
    departure_times: Int[Array, " num_passengers"]
    time_waiting: Int[Array, " num_passengers"]
    time_in_vehicle: Int[Array, " num_passengers"]
    statuses: Int[Array, " num_passengers"]  # dtype: PassengerStatus

    def update_passengers(self, current_time: int) -> "PassengerState":
        """
        Update passenger statuses and times based on the current time.

        Args:
            current_time: The current simulation timestep.

        Returns:
            Updated PassengerState instance.
        """
        raise NotImplementedError

    def increment_wait_times(self) -> "PassengerState":
        """
        Increment waiting times for passengers who are waiting.

        Returns:
            Updated PassengerState instance with incremented waiting times.
        """
        raise NotImplementedError

    def increment_in_vehicle_times(self) -> "PassengerState":
        """
        Increment in-vehicle times for passengers who are in vehicles.

        Returns:
            Updated PassengerState instance with incremented in-vehicle times.
        """
        raise NotImplementedError


@dataclass
class State:
    """
    Represents the complete state of the environment.
    """

    network: "NetworkData"
    fleet: "Fleet"
    passenger_state: "PassengerState"
    routes: "RouteBatch"
    current_time: int
    key: PRNGKeyArray

    def step(self, actions: Int[Array, " ..."]) -> tuple["State", "TimeStep"]:
        """
        Advance the simulation by one timestep using the provided actions.

        Args:
            actions: Array of actions for flexible routes.

        Returns:
            Tuple containing the new State instance and a TimeStep instance.
        """
        raise NotImplementedError

    def reset(self, key: PRNGKeyArray) -> "State":
        """
        Reset the environment to the initial state.

        Args:
            key: Random number generator key.

        Returns:
            Initial State instance.
        """
        raise NotImplementedError


@dataclass
class Observation:
    """
    Represents the observable state provided to the agent.
    """

    network: "NetworkData"
    routes: "RouteBatch"
    fleet_positions: Float[Array, " num_vehicles 2"]
    passenger_statuses: Int[Array, " num_passengers"]
    current_time: int
    action_mask: Bool[Array, " num_flexible_routes num_nodes+1"]

    def get_action_mask(
        self, state: "State"
    ) -> Bool[Array, " num_flexible_routes num_nodes_plus_1"]:
        """
        Get a mask indicating valid actions for flexible routes.

        Args:
            state: State instance.

        Returns:
            Boolean array indicating valid actions.
        """
        raise NotImplementedError


@dataclass
class Metrics:
    """
    Represents performance metrics of the simulation.
    """

    completed_passengers: int
    total_waiting_time: float
    total_in_vehicle_time: float
    average_waiting_time: float
    average_in_vehicle_time: float
    vehicle_utilization: float

    @staticmethod
    def compute(state: "State") -> "Metrics":
        """
        Compute metrics based on the current state.

        Args:
            state: State instance.

        Returns:
            Metrics instance containing computed metrics.
        """
        raise NotImplementedError
