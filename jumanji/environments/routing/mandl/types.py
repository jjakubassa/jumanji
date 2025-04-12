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
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jumanji.types import TimeStep

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass
# ruff: noqa: F722
from typing import Annotated

from typing_extensions import TypeAlias

# satisfy mypy and jaxtyping
NumRoutes: TypeAlias = Annotated[int, "num_routes"]
NumVehicles: TypeAlias = Annotated[int, "num_vehicles"]
NumFlexRoutes: TypeAlias = Annotated[int, "num_flex_routes"]
NumPassengers: TypeAlias = Annotated[int, "num_passengers"]


class PassengerStatus(IntEnum):
    """Enum for passenger statuses."""

    NOT_IN_SYSTEM = 0
    WAITING = 1
    IN_VEHICLE = 2
    TRANSFERRING = 3
    COMPLETED = 4


class RouteType(IntEnum):
    """Enum for route types."""

    FIXED = 0
    FLEXIBLE = 1


class VehicleDirection(IntEnum):
    FORWARD = 0
    BACKWARDS = 1


@dataclass
class NetworkData:
    """
    Represents the physical network structure and provides methods for network operations.
    """

    node_coordinates: Float[Array, " num_nodes 2"]
    travel_times: Float[Array, " num_nodes num_nodes"]
    """Rows indicate source and columns indicate destination"""
    is_terminal: Bool[Array, " num_nodes"]

    @property
    def num_nodes(self) -> Int[Array, ""]:
        return jnp.array(len(self.is_terminal))


@dataclass
class RouteBatch:
    """
    Represents a collection of routes.
    """

    types: Int[Array, " num_routes"]  # dtype: RouteType
    stops: Int[Array, "num_routes max_route_length"]
    vehicles_per_route: Float[Array, " num_routes"]
    num_flex_routes: Int[Array, ""]
    num_fix_routes: Int[Array, ""]

    @property
    def num_routes(self) -> Int[Array, ""]:
        return self.num_flex_routes + self.num_fix_routes

    @property
    def max_route_length(self) -> int:
        result = self.stops.shape[1]
        assert isinstance(result, int)
        return result


@dataclass
class Fleet:
    """
    Represents the collection of all vehicles in the simulation.
    """

    route_ids: Int[Array, " num_vehicles"]
    current_edges: Int[Array, " num_vehicles"]
    times_on_edge: Float[Array, " num_vehicles"]
    passengers: Int[Array, " num_vehicles max_capacity"]
    directions: Int[Array, " num_vehicles"]

    # def __post_init__(self) -> None:
    # if self.route_ids.shape[0] == 0:
    #     raise ValueError("Fleet cannot be empty")
    # if self.passengers.shape[1] == 0:
    #     raise ValueError("Fleet must have at max_capacity > 0")

    @property
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
        return jnp.isclose(self.times_on_edge, 0.0, rtol=1e-5, atol=1e-8)

    @property
    def num_passengers(self) -> Int[Array, " num_vehicles"]:
        return (self.passengers != -1).sum(axis=1)


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
    has_transferred: Bool[Array, " num_passengers"]
    transfer_nodes: Int[Array, " num_passengers"]  # -1 if no transfer

    @property
    def num_passengers(self) -> int:
        return len(self.origins)


@dataclass
class State:
    """
    Represents the complete state of the environment.
    """

    network: NetworkData
    fleet: Fleet
    passengers: Passengers
    routes: RouteBatch
    current_time: Float[Array, ""]
    key: PRNGKeyArray


@dataclass
class Observation:
    """Represents the observable state provided to the agent."""

    # Network data (flattened from NetworkData
    num_nodes: Int[Array, ""]
    travel_times: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821
    is_terminal: Bool[Array, "{self.num_nodes}"]  # noqa: F821

    # Routes data (flattened from RouteBatch)
    num_routes: Int[Array, ""]
    num_flex_routes: Int[Array, ""]
    num_fix_routes: Int[Array, ""]
    max_route_length: Int[Array, ""]
    route_types: Int[Array, "{self.num_routes}"]  # noqa: F821
    route_stops: Int[Array, "{self.num_routes} {self.max_route_length}"]
    route_frequencies: Float[Array, "{self.num_routes}"]  # noqa: F821
    network_shortest_times: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821
    direct_travel_times: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821
    transfer_travel_times: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821

    # Fleet data
    num_vehicles: Int[Array, ""]
    fleet_positions: Int[Array, "{self.num_vehicles} 2"]

    # Passenger data
    future_demand: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821
    waiting_demand: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821
    transferring_demand: Float[Array, "{self.num_nodes}*{self.num_nodes}"]  # noqa: F821

    # Environment state
    current_time: Float[Array, ""]
    action_mask: Bool[Array, "{self.num_routes} {self.num_nodes}+1"]


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


### NetworkData related functions ###
def is_connected(
    network: NetworkData,
    from_node: Int[Array, "..."],
    to_node: Int[Array, "..."],
) -> Bool[Array, "..."]:
    travel_time = network.travel_times[from_node, to_node]
    return jnp.isfinite(travel_time)


def get_travel_time(
    network: NetworkData, from_node: Int[Array, "..."], to_node: Int[Array, "..."]
) -> Float[Array, "..."]:
    """Get the travel time between two nodes."""
    return network.travel_times[from_node, to_node]


### RouteBatch related functions ###
def get_valid_stops(routes: RouteBatch) -> Bool[Array, "num_routes max_route_length"]:
    """Get valid stops for each route."""
    return routes.stops != -1


def update_routes(
    routes: RouteBatch, num_nodes: Int[Array, ""], action: Int[Array, " NumVehicles"]
) -> RouteBatch:
    # find indices of first free stop (-1) in route
    stop_planned: Bool[Array, " max_route_length"] = routes.stops != -1
    next_free_stop: Int[Array, " num_routes"] = (stop_planned).argmin(axis=1)

    # handle do nothing actions
    no_op_mask = action == num_nodes

    # handle case of already full routes by not modifying anything
    routes_are_full = stop_planned.all(axis=1)
    route_idxs = jnp.arange(routes.stops.shape[0])
    original_stops = routes.stops[route_idxs, next_free_stop]
    masked_actions = jnp.where(routes_are_full | no_op_mask, original_stops, action)

    new_stops = routes.stops.at[route_idxs, next_free_stop].set(masked_actions)
    return replace(routes, stops=new_stops)


def get_last_stops(routes: RouteBatch) -> Int[Array, " NumRoutes"]:
    """Retrieve the last valid stop for each route."""
    valid_stop_counts = jnp.sum(routes.stops != -1, axis=1)
    last_stop_indices = valid_stop_counts - 1
    num_routes = routes.stops.shape[0]
    route_indices = jnp.arange(num_routes)
    last_stops = routes.stops[route_indices, last_stop_indices]
    return last_stops


### Fleet related functions ###
def add_passenger(
    fleet: Fleet, vehicle_id: Int[Array, ""], passenger_id: Int[Array, ""]
) -> "Fleet":
    """
    Assign a passenger to the first available seat in the specified vehicle.
    Assumes that there is at leat on free seat in the vehicle specified.
    """
    idx_first_free_seat = (fleet.passengers[vehicle_id, :] == -1).argmax()
    new_passengers = fleet.passengers.at[vehicle_id, idx_first_free_seat].set(passenger_id)
    return replace(fleet, passengers=new_passengers)


def remove_passenger(
    fleet: Fleet, vehicle_id: Int[Array, ""], passenger_id: Int[Array, ""]
) -> "Fleet":
    """
    Remove a passenger from the specified vehicle.
    """
    passenger_seat_idx = jnp.where(fleet.passengers[vehicle_id] == passenger_id, size=1)
    new_passengers = fleet.passengers.at[vehicle_id, passenger_seat_idx].set(-1)
    return replace(fleet, passengers=new_passengers)


### Fleet related functions ###
def update_passengers_to_waiting(
    passengers: Passengers,
    current_time: Float[Array, ""],
) -> "Passengers":
    """
    Update passenger statuses based on current time and indices of passengers:
    - NOT_IN_SYSTEM to WAITING based on departure times

    Args:
        passengers: Passengers,
        current_time: Current simulation time.

    Returns:
        Updated Passengers.
    """

    new_statuses = jnp.where(
        (passengers.statuses == PassengerStatus.NOT_IN_SYSTEM)
        & (passengers.desired_departure_times <= current_time),
        PassengerStatus.WAITING,
        passengers.statuses,
    )
    return replace(passengers, statuses=new_statuses)


def increment_wait_times(passengers: Passengers) -> Passengers:
    """Increment waiting times for passengers who are waiting."""
    new_wait_times = jnp.where(
        (passengers.statuses == PassengerStatus.WAITING)
        | (passengers.statuses == PassengerStatus.TRANSFERRING),
        passengers.time_waiting + 1.0,
        passengers.time_waiting,
    )
    return replace(passengers, time_waiting=new_wait_times)


def increment_in_vehicle_times(passengers: Passengers) -> Passengers:
    """Increment in-vehicle times for passengers who are in vehicles."""
    new_in_vehicle_times = jnp.where(
        passengers.statuses == PassengerStatus.IN_VEHICLE,
        passengers.time_in_vehicle + 1.0,
        passengers.time_in_vehicle,
    )
    return replace(passengers, time_in_vehicle=new_in_vehicle_times)


### State related functions ###
def step(state: State, actions: Int[Array, " ..."]) -> tuple[State, TimeStep]:
    """
    Advance the simulation by one timestep using the provided actions.

    Args:
        actions: Array of actions for flexible routes.

    Returns:
        Tuple containing the new State instance and a TimeStep instance.
    """
    raise NotImplementedError


def get_vehicles_position_and_dest_node(
    state: State,
) -> tuple[Int[Array, "NumVehicles"], Int[Array, "NumVehicles"]]:
    routes = state.fleet.route_ids
    is_forward = state.fleet.directions == VehicleDirection.FORWARD

    # Get both potential nodes
    node_1 = state.routes.stops[routes, state.fleet.current_edges]
    node_2 = state.routes.stops[routes, state.fleet.current_edges + 1]

    # Swap nodes based on direction
    current_from = jnp.where(is_forward, node_1, node_2)
    current_to = jnp.where(is_forward, node_2, node_1)

    return current_from, current_to


def get_vehicles_position_and_dest_node_with_turnaround(
    state: State,
) -> tuple[Int[Array, "NumVehicles"], Int[Array, "NumVehicles"]]:
    routes = state.fleet.route_ids
    is_forward = state.fleet.directions == VehicleDirection.FORWARD
    is_backward = state.fleet.directions == VehicleDirection.BACKWARDS

    # Get route types and find which routes are fixed
    route_types = state.routes.types[routes]
    is_fixed_route = route_types == RouteType.FIXED

    # Count valid stops in each route to find last valid stop index
    valid_stops_mask = state.routes.stops[routes] != -1
    num_valid_stops = jnp.sum(valid_stops_mask, axis=1)
    last_valid_stop_idx = num_valid_stops - 1

    # Check if vehicle is at the end of a fixed route
    is_at_last_stop = (
        (state.fleet.current_edges == last_valid_stop_idx - 1) & is_forward & is_fixed_route
    )
    is_at_first_stop = (state.fleet.current_edges == 0) & is_backward & is_fixed_route
    is_about_to_turn = is_at_last_stop | is_at_first_stop

    # Get both potential nodes based on current edge
    node_1 = state.routes.stops[routes, state.fleet.current_edges]
    node_2 = state.routes.stops[routes, state.fleet.current_edges + 1]

    # For normal cases, swap nodes based on direction
    current_from = jnp.where(is_forward, node_1, node_2)
    current_to = jnp.where(is_forward, node_2, node_1)

    # Handle turnaround cases:
    # If at last stop going forward, next move is the same node but going backward
    # If at first stop going backward, next move is the same node but going forward
    current_from = jnp.where(is_about_to_turn, current_from, current_from)
    current_to = jnp.where(is_about_to_turn, current_from, current_to)

    return current_from, current_to


def move_vehicles(state: State) -> State:
    """Move all vehicles according to their routes and update positions."""
    # First increment times_on_edge for all vehicles
    new_times = state.fleet.times_on_edge + 1.0

    # Get travel times for current edges
    current_from, current_to = get_vehicles_position_and_dest_node(state)
    travel_times = get_travel_time(state.network, current_from, current_to)

    # Check which vehicles have completed their current edge
    completed_edge = new_times >= travel_times

    # Compute updates for all vehicles
    new_edges, new_directions = _update_completed_vehicles(
        state,
        state.fleet.route_ids,
        state.fleet.current_edges,
        state.fleet.directions,
        completed_edge,
    )

    # Reset times_on_edge for vehicles that completed their edge
    new_times = jnp.where(completed_edge, 0.0, new_times)

    # Create updated fleet
    new_fleet = replace(
        state.fleet,
        current_edges=new_edges,
        times_on_edge=new_times,
        directions=new_directions,
    )

    return replace(state, fleet=new_fleet)


def _update_completed_vehicles(
    state: State,
    route_ids: Int[Array, " NumVehicles"],
    current_edges: Int[Array, "NumVehicles"],
    directions: Int[Array, " NumVehicles"],
    completed: Bool[Array, " NumVehicles"],
) -> tuple[Int[Array, "NumVehicles"], Int[Array, " NumVehicles"]]:
    # Get route information
    route_types = state.routes.types[route_ids]
    routes = state.routes.stops[route_ids]
    max_num_stops = state.routes.stops.shape[1]
    max_num_edges = max_num_stops - 1
    current_edges = state.fleet.current_edges

    # Find last valid edge for each route
    last_valid_edge_idx = max_num_edges - (routes == -1).sum(axis=1) - 1

    # For completed edges, determine if we need to reverse direction
    current_from, current_to = get_vehicles_position_and_dest_node(state)
    is_forward_old = directions == VehicleDirection.FORWARD
    is_backward_old = directions == VehicleDirection.BACKWARDS
    is_at_end = (current_edges == last_valid_edge_idx) & completed & is_forward_old
    is_at_start = (current_edges == 0) & completed & is_backward_old
    should_reverse = (route_types == RouteType.FIXED) & (is_at_end | is_at_start)

    # Update directions
    new_directions = jnp.where(
        should_reverse,
        1 - directions,  # Toggle direction
        directions,
    )

    # Calculate what the vehicle will do BEFORE direction change
    will_go_forward = ~should_reverse & is_forward_old  # Continue forward
    will_go_backward = ~should_reverse & is_backward_old  # Continue backward
    will_switch_to_forward = should_reverse & is_backward_old  # Was backward, now forward
    will_switch_to_backward = should_reverse & is_forward_old  # Was forward, now backward

    # Calculate next stop indices based on these consistent vehicle behaviors
    new_edges = jnp.select(
        [will_switch_to_forward, will_switch_to_backward, will_go_forward, will_go_backward],
        [0, last_valid_edge_idx, current_edges + 1, current_edges - 1],
        current_edges,  # Default case - shouldn't happen
    )

    # Keep old edges for vehicles that haven't completed their edge
    new_edges = jnp.where(completed, new_edges, current_edges)

    return new_edges, new_directions


def calculate_single_route_times(
    route_stops: jnp.ndarray,
    direction: jnp.ndarray,
    current_edge: jnp.ndarray,
    remaining_time: jnp.ndarray,
    is_at_stop: jnp.ndarray,
    current_from: jnp.ndarray,
    current_to: jnp.ndarray,
    travel_times: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate direct travel times between nodes for a single route."""
    num_nodes = travel_times.shape[0]

    # Initialize times matrix
    times = jnp.full((num_nodes, num_nodes), jnp.inf)
    times = times.at[jnp.diag_indices_from(times)].set(0)

    # Get stops in correct order based on direction
    stops = jnp.where(direction == VehicleDirection.FORWARD, route_stops, jnp.flip(route_stops))

    # Calculate next stop based on direction
    next_or_current_stop = jnp.where(is_at_stop, current_from, current_to)

    # Initialize arrival times
    max_num_stops = route_stops.shape[0]
    arrival_times = jnp.full(max_num_stops, jnp.inf)
    arrival_times = jnp.where(
        is_at_stop,
        arrival_times.at[next_or_current_stop - 1].set(0),
        arrival_times.at[next_or_current_stop - 1].set(remaining_time),
    )

    def arrival_time_at_stop_i(
        i: Int[Array, ""], start_times: Float[Array, " max_num_stops"]
    ) -> Float[Array, " max_num_stops"]:
        current_pos = stops[i + 1]
        prev_pos = stops[i]
        edge_time = jnp.where(
            (current_pos == -1) | (prev_pos == -1), jnp.inf, travel_times[prev_pos, current_pos]
        )
        new_time = start_times[i - 1] + edge_time
        return start_times.at[i].set(new_time)

    arrival_times = jax.lax.fori_loop(
        next_or_current_stop, max_num_stops - 1, arrival_time_at_stop_i, arrival_times
    )

    def process_outer_i(
        i: Int[Array, ""], times: Float[Array, " num_nodes num_nodes"]
    ) -> Float[Array, " num_nodes num_nodes"]:
        from_stop = stops[i]
        time_from_i = arrival_times[i - 1]

        def process_inner_j(
            j: Int[Array, ""],
            carry: tuple[Float[Array, " num_nodes num_nodes"], Int[Array, ""], Float[Array, ""]],
        ) -> tuple[Float[Array, " num_nodes num_nodes"], Int[Array, ""], Float[Array, ""]]:
            times, prev_pos, time_from_i = carry
            to_stop = stops[j]
            current_edge_time = travel_times[prev_pos, to_stop]
            total_time = time_from_i + current_edge_time

            times = jnp.where(
                (to_stop != -1) & jnp.isinf(times[from_stop, to_stop]),
                times.at[from_stop, to_stop].set(total_time),
                times,
            )

            return times, to_stop, total_time

        init_carry = (times, from_stop, time_from_i)
        final_times, _, _ = jax.lax.fori_loop(i + 1, max_num_stops, process_inner_j, init_carry)

        return final_times

    times = jax.lax.fori_loop(next_or_current_stop, max_num_stops - 1, process_outer_i, times)

    return times


def calculate_invehicle_times(state: State) -> jnp.ndarray:
    """Calculate direct travel times between nodes for all routes."""
    # Vectorize over routes
    stops = state.routes.stops[state.fleet.route_ids]

    current_from, current_to = get_vehicles_position_and_dest_node_with_turnaround(state)
    batched_calc = jax.vmap(calculate_single_route_times, in_axes=(0, 0, 0, 0, 0, 0, 0, None))

    return batched_calc(
        stops,
        state.fleet.directions,
        state.fleet.current_edges,
        state.fleet.times_on_edge,
        state.fleet.is_at_node,
        current_from,
        current_to,
        state.network.travel_times,
    )


def calculate_shortest_route_times(
    state: State,
) -> tuple[
    Float[Array, "num_routes num_nodes num_nodes"],
    Int[Array, "num_routes num_nodes num_nodes"],
]:
    """
    Calculate travel times between all pairs of stops for each route and track direction.
    Returns both times and direction indicators (-1: no connection, 0: forward, 1: backward).
    """
    num_nodes = state.network.travel_times.shape[0]
    max_route_length = state.routes.stops.shape[1]

    def calculate_single_route_times(
        route: Int[Array, " max_route_length"], route_type: Int[Array, ""]
    ) -> tuple[Float[Array, " num_nodes num_nodes"], Int[Array, " num_nodes num_nodes"]]:
        # Initialize result matrices
        result_times = jnp.full((num_nodes, num_nodes), jnp.inf)
        result_times = result_times.at[jnp.arange(num_nodes), jnp.arange(num_nodes)].set(0.0)
        # -1 for no connection, 0 for forward, 1 for backward
        result_directions = jnp.full((num_nodes, num_nodes), -1, dtype=jnp.int32)

        def accumulate_fn(
            i: Int[Array, ""], accumulated_times: Float[Array, " max_route_length"]
        ) -> Float[Array, " max_route_length"]:
            src = route[i]
            dest = route[i + 1]
            is_valid = (src >= 0) & (dest >= 0)
            segment_time = state.network.travel_times[src, dest]
            new_time = accumulated_times[i] + jnp.where(is_valid, segment_time, 0.0)
            return accumulated_times.at[i + 1].set(new_time)

        # Initialize times (0 for first stop)
        accumulated_times = jnp.zeros(max_route_length)
        accumulated_times = jax.lax.fori_loop(
            0, max_route_length - 1, accumulate_fn, accumulated_times
        )

        # Update result matrices for each source stop
        def update_from_stop(
            i: Int[Array, ""],
            result: tuple[Float[Array, " num_nodes num_nodes"], Int[Array, " num_nodes num_nodes"]],
        ) -> tuple[Float[Array, " num_nodes num_nodes"], Int[Array, " num_nodes num_nodes"]]:
            times, directions = result
            src = route[i]
            is_valid_src = src >= 0

            def update_to_stop(
                j: Int[Array, ""],
                res: tuple[
                    Float[Array, " num_nodes num_nodes"], Int[Array, " num_nodes num_nodes"]
                ],
            ) -> tuple[Float[Array, " num_nodes num_nodes"], Int[Array, " num_nodes num_nodes"]]:
                curr_times, curr_directions = res
                dest = route[j]
                is_valid_dest = dest >= 0
                is_valid = is_valid_src & is_valid_dest & (j > i)

                time = accumulated_times[j] - accumulated_times[i]

                # Forward pass updates
                curr_times = curr_times.at[src, dest].set(
                    jnp.where(is_valid, time, curr_times[src, dest])
                )
                curr_directions = curr_directions.at[src, dest].set(
                    jnp.where(is_valid, VehicleDirection.FORWARD, curr_directions[src, dest])
                )

                # For fixed routes, mirror the times with backward direction
                is_fixed = route_type == RouteType.FIXED
                curr_times = jnp.where(
                    is_valid & is_fixed, curr_times.at[dest, src].set(time), curr_times
                )
                curr_directions = jnp.where(
                    is_valid & is_fixed,
                    curr_directions.at[dest, src].set(VehicleDirection.BACKWARDS),
                    curr_directions,
                )

                return (curr_times, curr_directions)

            result = jax.lax.fori_loop(0, max_route_length, update_to_stop, (times, directions))
            assert isinstance(result, tuple)
            return result

        final_times, final_directions = jax.lax.fori_loop(
            0, max_route_length, update_from_stop, (result_times, result_directions)
        )
        return final_times, final_directions

    # Vectorize over routes and route types
    batch_calculate = jax.vmap(calculate_single_route_times)
    result = batch_calculate(state.routes.stops, state.routes.types)
    assert isinstance(result, tuple)
    return result


def find_best_transfer_route(
    state: State,
    origin: Int[Array, ""],
    destination: Int[Array, ""],
    route_times: Float[Array, "num_routes num_nodes num_nodes"],
) -> tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""]]:
    # Mask out flex routes by setting their times to infinity
    fixed_route_mask = state.routes.types == RouteType.FIXED
    masked_route_times = jnp.where(fixed_route_mask[:, None, None], route_times, jnp.inf)

    def update_best_transfer(
        carry: tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""]],
        transfer_node: Int[Array, ""],
    ) -> tuple[tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""]], None]:
        curr_best_time, curr_transfer_node, curr_first_leg_route, curr_second_leg_route = carry

        # Time to transfer point on first leg
        first_leg_times = masked_route_times[:, origin, transfer_node]
        # Time from transfer point to destination on second leg
        second_leg_times = masked_route_times[:, transfer_node, destination]

        # Calculate all possible combinations
        transfer_times = first_leg_times[:, None] + second_leg_times[None, :]

        # Find best combination for this transfer node
        best_time = jnp.min(transfer_times)
        best_idx = jnp.unravel_index(jnp.argmin(transfer_times), transfer_times.shape)
        first_leg_route, second_leg_route = best_idx

        # Update if better than current best
        is_better = best_time < curr_best_time
        new_best_time = jnp.where(is_better, best_time, curr_best_time)
        new_transfer_node = jnp.where(is_better, transfer_node, curr_transfer_node)
        new_first_leg_route = jnp.where(is_better, first_leg_route, curr_first_leg_route)
        new_second_leg_route = jnp.where(is_better, second_leg_route, curr_second_leg_route)

        return (new_best_time, new_transfer_node, new_first_leg_route, new_second_leg_route), None

    # Initialize with infinity/invalid values
    init_carry = (jnp.array(jnp.inf), jnp.array(-1), jnp.array(-1), jnp.array(-1))

    # Scan through all possible transfer nodes
    num_nodes = route_times.shape[1]
    (best_time, transfer_node, first_leg_route, second_leg_route), _ = jax.lax.scan(
        update_best_transfer, init_carry, jnp.arange(num_nodes)
    )

    return best_time, transfer_node, first_leg_route, second_leg_route


def floyd_warshall(
    travel_times: Float[Array, "num_nodes num_nodes"],
) -> Float[Array, "num_nodes num_nodes"]:
    """Calculate all-pairs shortest paths using Floyd-Warshall algorithm."""
    n = travel_times.shape[0]
    dist = travel_times.copy()

    def body_fun(
        k: Int[Array, ""], dist: Float[Array, "num_nodes num_nodes"]
    ) -> Float[Array, "num_nodes num_nodes"]:
        """Update distances using node k as intermediate."""
        # Get the k-th row and column
        row_k = dist[k, :]  # Shape: (n,)
        col_k = dist[:, k]  # Shape: (n,)

        # Calculate new distances through k
        new_dists = jnp.expand_dims(col_k, 1) + row_k  # Shape: (n, n)

        # Update distances where new path is shorter
        return jnp.minimum(dist, new_dists)

    dist = jax.lax.fori_loop(0, n, body_fun, dist)

    return dist


def assign_passengers(
    state: State,
    max_travel_time_ratio: float = 1.3,
) -> State:
    """Optimistic assignment considering both direct and transfer journeys."""
    # Calculate route times once at the start
    shortest_travel_time_per_route, direction_for_shortest_path_per_route = (
        calculate_shortest_route_times(state)
    )
    shortest_travel_time_overall = jnp.min(shortest_travel_time_per_route, axis=0)
    direct_route_exits = jnp.isfinite(shortest_travel_time_overall)
    current_from, _ = get_vehicles_position_and_dest_node(state)

    # decide acceptable routes
    route_time_good_enough = (
        shortest_travel_time_per_route <= shortest_travel_time_overall * max_travel_time_ratio
    ) & jnp.isfinite(shortest_travel_time_per_route)

    # change to vehicle view
    vehicle_time_good_enough = route_time_good_enough[state.fleet.route_ids]
    direction_for_shortest_path_per_vehicle = direction_for_shortest_path_per_route[
        state.fleet.route_ids
    ]
    shortest_travel_time_per_vehicle = shortest_travel_time_per_route[state.fleet.route_ids]

    def assign_single_passenger(
        carry: tuple[
            Int[Array, "NumVehicles max_capacity"],  # new_fleet_passengers
            Int[Array, "NumPassengers"],  # new_passenger_statuses
            Int[Array, "NumPassengers"],  # new_transfer_nodes
            Int[Array, "NumVehicles"],  # new_capacities_left
        ],
        passenger_idx: Int[Array, ""],
    ) -> tuple[tuple, None]:
        new_fleet_passengers, new_passenger_statuses, new_transfer_nodes, new_capacities_left = (
            carry
        )
        is_waiting = new_passenger_statuses[passenger_idx] == PassengerStatus.WAITING
        is_transferring = new_passenger_statuses[passenger_idx] == PassengerStatus.TRANSFERRING
        is_waiting_or_transferring = is_waiting | is_transferring

        def process_passenger() -> (
            tuple[
                Int[Array, "NumVehicles max_capacity"],
                Int[Array, "NumPassengers"],
                Int[Array, "NumPassengers"],
                Int[Array, "NumVehicles"],
            ]
        ):
            effective_origin = jnp.where(
                is_transferring,
                state.passengers.transfer_nodes[passenger_idx],
                state.passengers.origins[passenger_idx],
            )
            dest = state.passengers.destinations[passenger_idx]

            # Always prefer direct trips
            direct_possible = direct_route_exits[effective_origin, dest]
            only_flex_routes = jnp.all(state.routes.types == RouteType.FLEXIBLE)
            _, transfer_node, _, _ = jax.lax.cond(
                direct_possible | only_flex_routes,
                lambda: (jnp.inf, jnp.array(-1), jnp.array(-1), jnp.array(-1)),
                lambda: find_best_transfer_route(
                    state, effective_origin, dest, shortest_travel_time_per_route
                ),
            )
            effective_dest = jnp.where(
                direct_possible,
                dest,
                transfer_node,
            )

            vehicle_is_on_good_route = vehicle_time_good_enough[:, effective_origin, effective_dest]
            is_at_correct_stop = state.fleet.is_at_node & (current_from == effective_origin)
            moves_in_right_direction = (
                direction_for_shortest_path_per_vehicle[:, effective_origin, effective_dest]
                == state.fleet.directions
            )
            should_board = vehicle_is_on_good_route & is_at_correct_stop & moves_in_right_direction
            valid_times = jnp.where(
                should_board,
                shortest_travel_time_per_vehicle[:, effective_origin, effective_origin],
                jnp.inf,
            )

            # Among valid options, pick one with shortest travel time
            best_time = jnp.min(valid_times)
            has_best_time = valid_times == best_time

            # Among those with best time, pick one with highest capacity
            best_vehicle = jnp.argmax(has_best_time * new_capacities_left)

            def update_arrays(
                carry: tuple[
                    Int[Array, "NumVehicles max_capacity"],
                    Int[Array, "NumPassengers"],
                    Int[Array, "NumPassengers"],
                    Int[Array, "NumVehicles"],
                ],
            ) -> tuple[
                Int[Array, "NumVehicles max_capacity"],
                Int[Array, "NumPassengers"],
                Int[Array, "NumPassengers"],
                Int[Array, "NumVehicles"],
            ]:
                fleet_passengers, passenger_statuses, transfer_nodes, capacities_left = carry
                # Find first available seat
                idx_first_free_seat = (fleet_passengers[best_vehicle, :] == -1).argmax()

                # Update arrays
                new_fleet_passengers = fleet_passengers.at[best_vehicle, idx_first_free_seat].set(
                    passenger_idx
                )
                new_passenger_statuses = passenger_statuses.at[passenger_idx].set(
                    PassengerStatus.IN_VEHICLE
                )
                new_transfer_nodes = transfer_nodes.at[passenger_idx].set(transfer_node)
                new_capacities_left = capacities_left.at[best_vehicle].add(-1)

                return (
                    new_fleet_passengers,
                    new_passenger_statuses,
                    new_transfer_nodes,
                    new_capacities_left,
                )

            new_carry = jax.lax.cond(
                should_board[best_vehicle],
                update_arrays,
                lambda c: c,
                carry,
            )
            # assert isinstance(new_state, State)
            return new_carry  # type: ignore

        new_carry = jax.lax.cond(
            is_waiting_or_transferring,
            process_passenger,
            lambda: (
                new_fleet_passengers,
                new_passenger_statuses,
                new_transfer_nodes,
                new_capacities_left,
            ),
        )

        return new_carry, None

    init_carry = (
        state.fleet.passengers,
        state.passengers.statuses,
        state.passengers.transfer_nodes,
        state.fleet.capacities_left,
    )

    # Process all waiting passengers
    (final_fleet_passengers, final_statuses, final_transfer_nodes, final_capacities_left), _ = (
        jax.lax.scan(
            assign_single_passenger,
            init_carry,
            jnp.arange(state.passengers.num_passengers),
        )
    )

    new_fleet = replace(state.fleet, passengers=final_fleet_passengers)
    new_passengers = replace(
        state.passengers,
        statuses=final_statuses,
        transfer_nodes=final_transfer_nodes,
    )

    return replace(state, fleet=new_fleet, passengers=new_passengers)


def handle_completed_and_transferring_passengers(state: State) -> State:
    """
    For all vehicles that are at a node (i.e. times_on_edge==0), check each seat.
    For seats containing passengers:
    - If passenger's destination equals current node, they are marked COMPLETED and removed
    - If passenger's transfer node equals current node and they haven't transferred yet,
      they are marked TRANSFERRING and removed to wait for another vehicle.
    """
    passengers = state.fleet.passengers  # shape: (num_vehicles, max_capacity)
    is_at_stop = state.fleet.is_at_node

    # Find passengers in vehicles that are currently at any stop (not inbetween nodes)
    passenger_ids_masked = jnp.where(
        is_at_stop[:, None], passengers, -1
    )  # shape: (num_vehicles, max_capacity)

    # Find destinations, transfer nodes, and has_transferred flags for those passengers
    destinations = state.passengers.destinations[passengers]
    transfer_nodes = state.passengers.transfer_nodes[passengers]
    has_transferred = state.passengers.has_transferred[passengers]

    # Mask out invalid passengers (not in vehicle)
    dest_node_passenger_in_vehicle = jnp.where(passenger_ids_masked != -1, destinations, -1)
    transfer_node_passenger_in_vehicle = jnp.where(passenger_ids_masked != -1, transfer_nodes, -1)

    # Check if at destination or transfer node
    current_node, _ = get_vehicles_position_and_dest_node(state)
    passenger_is_at_dest = dest_node_passenger_in_vehicle == current_node[:, None]
    passenger_is_at_transfer_and_no_prev_transfer = (
        transfer_node_passenger_in_vehicle == current_node[:, None]
    ) & ~has_transferred

    def update_passenger_status(
        i: Int[Array, ""],
        carry: tuple[Int[Array, " NumPassengers"], Bool[Array, " NumPassengers"]],
    ) -> tuple[Int[Array, " NumPassengers"], Bool[Array, " NumPassengers"]]:
        statuses, has_transferred = carry
        vehicle_idx = i // passengers.shape[1]  # get vehicle index
        seat_idx = i % passengers.shape[1]  # get seat index

        passenger_id = passenger_ids_masked[vehicle_idx, seat_idx]
        is_valid_p_id = passenger_id != -1
        is_at_dest = passenger_is_at_dest[vehicle_idx, seat_idx] & is_valid_p_id
        is_at_transfer = (
            passenger_is_at_transfer_and_no_prev_transfer[vehicle_idx, seat_idx] & is_valid_p_id
        )

        status = statuses[passenger_id]

        # Update passenger status
        new_status = jnp.select(
            [is_at_dest, is_at_transfer],
            [PassengerStatus.COMPLETED, PassengerStatus.TRANSFERRING],
            default=status,
        )

        statuses = statuses.at[passenger_id].set(new_status)

        # Update has_transferred flag
        has_transferred = has_transferred.at[passenger_id].set(
            has_transferred[passenger_id] | is_at_transfer
        )

        return (statuses, has_transferred)

    # Initialize carry values
    init_statuses = state.passengers.statuses
    init_has_transferred = state.passengers.has_transferred

    # Loop over all passengers in vehicles
    total_passengers = passengers.shape[0] * passengers.shape[1]
    new_statuses, new_has_transferred = jax.lax.fori_loop(
        jnp.array(0),
        jnp.array(total_passengers),
        update_passenger_status,
        (init_statuses, init_has_transferred),
    )

    # Update fleet: remove passengers who should get off
    new_fleet = replace(
        state.fleet,
        passengers=jnp.where(
            passenger_is_at_dest | passenger_is_at_transfer_and_no_prev_transfer,
            -1,
            state.fleet.passengers,
        ),
    )

    # Update passenger state
    new_passengers = replace(
        state.passengers, statuses=new_statuses, has_transferred=new_has_transferred
    )

    return replace(state, fleet=new_fleet, passengers=new_passengers)


def get_position_in_route(state: State) -> Int[Array, "num_routes num_nodes"]:
    """Get the position of each node in the route. -1 if node not in route."""
    routes = state.routes.stops
    num_routes = routes.shape[0]
    num_nodes = len(state.network.is_terminal)
    max_num_stops = routes.shape[1]
    position_in_route = jnp.full((num_routes, num_nodes), -1)

    def update_position(
        i: Int[Array, ""], positions: Int[Array, "num_routes num_nodes"]
    ) -> Int[Array, "num_routes num_nodes"]:
        node = routes[:, i]
        return jnp.where(
            (node >= 0)[:, None],
            positions.at[jnp.arange(num_routes), node].set(i),
            positions,
        )

    return jax.lax.fori_loop(0, max_num_stops, update_position, position_in_route)


def get_direction_if_connected(
    state: State, origin: Int[Array, ""], dest: Int[Array, ""]
) -> Int[Array, " NumRoutes"]:
    """Determine if forward movement is needed to get from origin to destination in route."""
    positions = get_position_in_route(state)
    origin_pos = positions[:, origin]
    dest_pos = positions[:, dest]
    return jnp.select(
        [
            (origin_pos == -1) | (dest_pos == -1),
            dest_pos > origin_pos,
            dest_pos < origin_pos,
            dest_pos == origin_pos,
        ],
        [-1, VehicleDirection.FORWARD, VehicleDirection.BACKWARDS, 2],
    )
