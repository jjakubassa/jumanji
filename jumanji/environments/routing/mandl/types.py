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

from dataclasses import replace
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING

import jax.numpy as jnp
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
    travel_times: Float[Array, " num_nodes num_nodes"]
    is_terminal: Bool[Array, " num_nodes"]

    def __post_init__(self) -> None:
        # Data validation
        if not jnp.all(self.travel_times >= 0):
            raise ValueError("travel_times must contain non-negative values")
        if not jnp.all(jnp.diag(self.travel_times) == 0):
            raise ValueError("Diagonal elements of travel_times must be zero (self-connections)")
        if not jnp.all(jnp.isfinite(self.node_coordinates)):
            raise ValueError("node_coordinates must contain finite values")

    def is_connected(self, from_node: Int[Array, ""], to_node: Int[Array, ""]) -> Bool[Array, ""]:
        """
        Check if two nodes are directly connected.

        Args:
            from_node: Index of the origin node.
            to_node: Index of the destination node.

        Returns:
            True if nodes are connected, False otherwise.
        """
        travel_time = self.travel_times[from_node, to_node]
        return jnp.isfinite(travel_time)

    def get_travel_time(
        self, from_node: Int[Array, ""], to_node: Int[Array, ""]
    ) -> Float[Array, ""]:
        """
        Get the travel time between two nodes.

        Args:
            from_node: Index of the origin node.
            to_node: Index of the destination node.

        Returns:
            Travel time between the nodes.
        """
        return self.travel_times[from_node, to_node]


@dataclass
class RouteBatch:
    """
    Represents a collection of routes.
    """

    types: Int[Array, " {self.num_flex_routes}+{self.num_fix_routes}"]  # dtype: RouteType
    stops: Int[Array, "{self.num_flex_routes}+{self.num_fix_routes} max_route_length"]
    frequencies: Float[Array, " {self.num_flex_routes}+{self.num_fix_routes}"]
    on_demand: Bool[Array, " {self.num_flex_routes}+{self.num_fix_routes}"]
    num_flex_routes: int
    num_fix_routes: int

    @cached_property
    def num_routes(self) -> Int[Array, ""]:
        return jnp.array(len(self.types))

    def get_valid_stops(self) -> Bool[Array, " num_routes max_route_length"]:
        """
        Get valid stops for each route.

        Returns:
            Boolean array of shape (num_routes, max_route_length) indicating valid stops.
        """
        return self.stops != -1

    def update_routes(self, actions: Int[Array, " num_flex_routes"]) -> "RouteBatch":
        """
        Update flexible routes based on agent actions.

        Args:
            actions: Array of actions for flexible routes (shape: (num_flexible_routes,))
        """
        flex_indices = jnp.where(self.types == RouteType.FLEXIBLE, size=self.num_flex_routes)
        next_free_stop = (self.stops[flex_indices] != -1).argmax(axis=1)
        new_stops = self.stops.at[flex_indices, next_free_stop].set(actions)
        return replace(self, stops=new_stops)


@dataclass
class Fleet:
    """
    Represents the collection of all vehicles in the simulation.
    """

    route_ids: Int[Array, " num_vehicles"]
    current_edges: Int[Array, " num_vehicles 2"]
    current_edges_indices: Int[Array, " num_vehicles"]
    times_on_edge: Float[Array, " num_vehicles"]
    passengers: Int[Array, " num_vehicles max_capacity"]
    directions: Int[Array, " num_vehicles"]

    def __post_init__(self) -> None:
        if self.route_ids.size == 0:
            raise ValueError("Fleet cannot be empty")
        if self.passengers.shape[1] == 0:
            raise ValueError("Fleet must have at max_capacity > 0")

    @cached_property
    def num_vehicles(self) -> int:
        return len(self.route_ids)

    @property
    def capacities_left(self) -> Int[Array, " num_vehicles"]:
        return (self.passengers == -1).sum(axis=1)

    @property
    def seat_is_available(self) -> Bool[Array, " num_vehicles"]:
        return self.capacities_left > 0

    @property
    def is_at_node(self) -> Bool[Array, " num_vehicles"]:
        return self.times_on_edge == 0

    def add_passenger(self, vehicle_id: Int[Array, ""], passenger_id: Int[Array, ""]) -> "Fleet":
        """
        Assign a passenger to the first available seat in the specified vehicle.
        Assumes that there is at leat on free seat in the vehicle specified.
        """
        idx_first_free_seat = (self.passengers[vehicle_id, :] == -1).argmax()
        new_passengers = self.passengers.at[vehicle_id, idx_first_free_seat].set(passenger_id)
        return replace(self, passengers=new_passengers)

    def remove_passenger(self, vehicle_id: Int[Array, ""], passenger_id: Int[Array, ""]) -> "Fleet":
        """
        Remove a passenger from the specified vehicle.
        """
        passenger_seat_idx = jnp.where(self.passengers[vehicle_id] == passenger_id, size=1)
        new_passengers = self.passengers.at[vehicle_id, passenger_seat_idx].set(-1)
        return replace(self, passengers=new_passengers)


@dataclass
class Passengers:
    """
    Represents the state of all passengers.
    """

    origins: Int[Array, " num_passengers"]
    destinations: Int[Array, " num_passengers"]
    desired_departure_times: Float[Array, " num_passengers"]
    time_waiting: Float[Array, " num_passengers"]
    time_in_vehicle: Float[Array, " num_passengers"]
    statuses: Int[Array, " num_passengers"]  # dtype: PassengerStatus

    @cached_property
    def num_passengers(self) -> int:
        return len(self.origins)

    def update_passengers(
        self,
        current_time: Float[Array, ""],
        to_in_vehicle: Bool[Array, " num_passengers"],
        to_completed: Bool[Array, " num_passengers"],
    ) -> "Passengers":
        """
        Update passenger statuses based on current time and indices of passengers:
        - NOT_IN_SYSTEM to WAITING based on departure times
        - WAITING to IN_VEHICLE based on provided indices
        - IN_VEHICLE to COMPLETED based on provided indices

        Args:
            current_time: Current simulation time.
            to_in_vehicle_indices: Indices of passengers transitioning to IN_VEHICLE status.
            to_completed_indices: Indices of passengers transitioning to COMPLETED status.

        Returns:
            Updated Passengers.
        """
        new_statuses = jnp.where(
            (self.statuses == PassengerStatus.NOT_IN_SYSTEM)
            & (self.desired_departure_times == current_time),
            PassengerStatus.WAITING,
            self.statuses,
        )
        new_statuses = jnp.where(to_in_vehicle, PassengerStatus.IN_VEHICLE, new_statuses)
        new_statuses = jnp.where(to_completed, PassengerStatus.COMPLETED, new_statuses)
        return replace(self, statuses=new_statuses)

    def increment_wait_times(self) -> "Passengers":
        """
        Increment waiting times for passengers who are waiting.

        Returns:
            Updated Passengers instance with incremented waiting times.
        """
        new_wait_times = jnp.where(
            self.statuses == PassengerStatus.WAITING, self.time_waiting + 1.0, self.time_waiting
        )
        return replace(self, time_waiting=new_wait_times)

    def increment_in_vehicle_times(self) -> "Passengers":
        """
        Increment in-vehicle times for passengers who are in vehicles.

        Returns:
            Updated Passengers instance with incremented in-vehicle times.
        """
        new_in_vehicle_times = jnp.where(
            self.statuses == PassengerStatus.IN_VEHICLE,
            self.time_in_vehicle + 1.0,
            self.time_in_vehicle,
        )
        return replace(self, time_in_vehicle=new_in_vehicle_times)


@dataclass
class State:
    """
    Represents the complete state of the environment.
    """

    network: "NetworkData"
    fleet: "Fleet"
    passenger_state: "Passengers"
    routes: "RouteBatch"
    current_time: Float
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
    current_time: float
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
