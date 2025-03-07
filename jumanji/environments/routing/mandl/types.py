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
    frequencies: Float[Array, " num_routes"]
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
    current_edges: Int[Array, " num_vehicles 2"]
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
        return jnp.isclose(self.times_on_edge, 0.0)

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
    """
    Represents the observable state provided to the agent.
    """

    network: "NetworkData"
    routes: "RouteBatch"
    fleet_positions: Int[Array, " num_vehicles 2"]
    origins: Int[Array, " num_passengers"]
    destinations: Int[Array, " num_passengers"]
    desired_departure_times: Float[Array, " num_passengers"]
    passenger_statuses: Int[Array, " num_passengers"]
    current_time: Float[Array, ""]
    action_mask: Bool[Array, "num_routes {self.netwowrk.num_nodes+1}"]


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


def move_vehicles(state: State) -> State:
    """Move all vehicles according to their routes and update positions."""
    # First increment times_on_edge for all vehicles
    new_times = state.fleet.times_on_edge + 1.0

    # Get travel times for current edges
    current_from = state.fleet.current_edges[:, 0]
    current_to = state.fleet.current_edges[:, 1]
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
    current_edges: Int[Array, "NumVehicles 2"],
    directions: Int[Array, " NumVehicles"],
    completed: Bool[Array, " NumVehicles"],
) -> tuple[Int[Array, "NumVehicles 2"], Int[Array, " NumVehicles"]]:
    # Get route information
    route_types = state.routes.types[route_ids]
    routes = state.routes.stops[route_ids]

    # Find last valid stop for each route
    route_lengths = (routes != -1).sum(axis=1)
    last_valid_indices = route_lengths - 1

    # For completed edges, determine if we need to reverse direction
    current_stops = current_edges[:, 1]  # Use destination of current edge
    is_fixed = route_types == RouteType.FIXED

    # Check against actual last valid stop
    last_stops = jnp.take_along_axis(routes, last_valid_indices[:, None], axis=1)[:, 0]
    is_at_end = current_stops == last_stops
    is_at_start = current_stops == routes[:, 0]

    # Update directions for completed edges
    should_reverse = (
        completed
        & is_fixed
        & (
            (directions == VehicleDirection.FORWARD) & is_at_end
            | (directions == VehicleDirection.BACKWARDS) & is_at_start
        )
    )
    new_directions = jnp.where(
        should_reverse,
        jnp.where(
            directions == VehicleDirection.FORWARD,
            VehicleDirection.BACKWARDS,
            VehicleDirection.FORWARD,
        ),
        directions,
    )

    # Create new edges for completed vehicles
    # Always start from current destination
    new_from = current_edges[:, 1]

    # Find indices of current stops in route
    route_length = routes.shape[1]
    curr_stop_indices = jnp.array(
        [jnp.where(routes[i] == new_from[i], size=1)[0][0] for i in range(len(route_ids))]
    )

    # Calculate next stop indices based on direction
    next_indices = jnp.where(
        new_directions == VehicleDirection.FORWARD,
        jnp.minimum(curr_stop_indices + 1, route_length - 1),
        jnp.maximum(curr_stop_indices - 1, 0),
    )

    # Get next stops from route
    batch_indices = jnp.arange(len(route_ids))
    new_to = routes[batch_indices, next_indices]

    # Stack the arrays to create edges
    new_edges = jnp.column_stack([new_from, new_to])

    # Keep old edges for vehicles that haven't completed their edge
    new_edges = jnp.where(completed[:, None], new_edges, current_edges)

    return new_edges, new_directions


def calculate_route_times(
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


def assign_passengers(
    state: State,
    discount_factor_future: float = 0.95,
) -> State:
    """Optimistic assignment considering both direct and transfer journeys."""

    # Get all waiting passengers
    waiting_mask = (state.passengers.statuses == PassengerStatus.WAITING) | (
        state.passengers.statuses == PassengerStatus.TRANSFERRING
    )

    # Calculate route times once at the start
    route_times, route_directions = calculate_route_times(state)
    all_waiting_times = calculate_waiting_times(state, route_times, route_directions)

    def assign_single_passenger(
        state: State,
        passenger_idx: Int[Array, ""],
        route_times: Float[Array, "num_routes num_nodes num_nodes"],
        route_directions: Int[Array, "num_routes num_nodes num_nodes"],
    ) -> State:
        # Update origin in case passenger is transferring
        is_transferring = state.passengers.statuses[passenger_idx] == PassengerStatus.TRANSFERRING
        effective_origin = jnp.where(
            is_transferring,
            state.passengers.transfer_nodes[passenger_idx],
            state.passengers.origins[passenger_idx],
        )
        dest = state.passengers.destinations[passenger_idx]

        # Try direct route first
        in_vehicle_times = route_times[:, effective_origin, dest]
        has_direct_route = jnp.any(jnp.isfinite(in_vehicle_times))

        # If no direct route, find best transfer
        _, transfer_node, _, _ = jax.lax.cond(
            has_direct_route,
            lambda: (jnp.inf, jnp.array(-1), jnp.array(-1), jnp.array(-1)),
            lambda: find_best_transfer_route(state, effective_origin, dest, route_times),
        )
        in_vehicle_times = jax.lax.cond(
            has_direct_route,
            lambda: in_vehicle_times,
            lambda: route_times[:, effective_origin, transfer_node],
        )

        # Calculate best direction for each vehicle
        effective_dest = jnp.where(
            has_direct_route,
            dest,
            transfer_node,
        )
        required_directions = get_direction_if_connected(state, effective_origin, effective_dest)
        best_direction = required_directions[state.fleet.route_ids]  # routes -> vehicles

        # Calculate overall time
        route_ids = state.fleet.route_ids
        in_vehicle_times = in_vehicle_times[route_ids]  # num of routes -> num vehicles
        # wait_forward = all_waiting_times[:, effective_origin, VehicleDirection.FORWARD]
        # wait_backward = all_waiting_times[:, effective_origin, VehicleDirection.BACKWARDS]
        wait_times = jnp.select(
            [
                best_direction == VehicleDirection.FORWARD,
                best_direction == VehicleDirection.BACKWARDS,
            ],
            [
                all_waiting_times[:, effective_origin, VehicleDirection.FORWARD],
                all_waiting_times[:, effective_origin, VehicleDirection.BACKWARDS],
            ],
            default=jnp.inf,
        )
        all_waiting_times[:, effective_origin, best_direction]
        journey_times = wait_times + in_vehicle_times

        # Only board vehicles under some conditions
        moves_in_best_direction = (best_direction != -1) & (
            state.fleet.directions == best_direction
        )
        is_at_correct_stop = state.fleet.is_at_node & (
            state.fleet.current_edges[:, 0] == effective_origin
        )
        immediate_boarding_possible = (
            is_at_correct_stop & state.fleet.seat_is_available & moves_in_best_direction
        )

        # Check route validity and calculate options
        connects_target = jnp.isfinite(journey_times)
        immediate_times = jnp.where(
            immediate_boarding_possible & connects_target, journey_times, jnp.inf
        )
        future_times = jnp.where(
            connects_target & state.fleet.seat_is_available, journey_times, jnp.inf
        )

        # Board if immediate option exists and no significantly better future option exists
        assert isinstance(future_times, jnp.ndarray)
        assert isinstance(immediate_times, jnp.ndarray)
        best_future_time = jnp.min(future_times)
        should_board = jnp.isfinite(immediate_times) & (
            best_future_time >= immediate_times * discount_factor_future
        )

        # Among valid options, pick one with shortest travel time
        valid_times = jnp.where(should_board, immediate_times, jnp.inf)
        best_time = jnp.min(valid_times)
        has_best_time = valid_times == best_time

        # Among those with best time, pick one with highest capacity
        best_vehicle = jnp.argmax(has_best_time * state.fleet.capacities_left)

        def update_state(state: State) -> State:
            new_fleet = add_passenger(state.fleet, best_vehicle, passenger_idx)
            new_statuses = state.passengers.statuses.at[passenger_idx].set(
                PassengerStatus.IN_VEHICLE
            )
            # Set transfer node if this is a transfer journey
            new_transfer_nodes = state.passengers.transfer_nodes.at[passenger_idx].set(
                transfer_node
            )
            new_passengers = replace(
                state.passengers,
                statuses=new_statuses,
                transfer_nodes=new_transfer_nodes,
            )
            return replace(state, fleet=new_fleet, passengers=new_passengers)

        new_state = jax.lax.cond(
            should_board[best_vehicle],
            update_state,
            lambda s: s,
            state,
        )
        assert isinstance(new_state, State)
        return new_state

    # Process all waiting passengers
    final_state = jax.lax.fori_loop(
        0,
        state.passengers.num_passengers,
        lambda i, s: jax.lax.cond(
            waiting_mask[i],
            lambda s: assign_single_passenger(s, i, route_times, route_directions),
            lambda s: s,
            s,
        ),
        state,
    )

    assert isinstance(final_state, State)
    return final_state


def calculate_waiting_times(
    state: State,
    route_times: Float[Array, "num_routes num_nodes num_nodes"],
    route_directions: Int[Array, "num_routes num_nodes num_nodes"],
) -> Float[Array, "num_vehicles num_nodes 2"]:
    """Calculate waiting times for each vehicle-node pair in both directions.

    Returns:
        times[v, n, 0]: waiting time if going forward from node n for vehicle v
        times[v, n, 1]: waiting time if going backward from node n for vehicle v
    """
    # Get vehicle data
    route_ids = state.fleet.route_ids
    current_positions = state.fleet.current_edges[:, 0]
    current_destinations = state.fleet.current_edges[:, 1]
    current_directions = state.fleet.directions
    num_nodes = len(state.network.is_terminal)

    # Find last valid stop for each route
    route_lengths = (state.routes.stops[route_ids] != -1).sum(axis=1)
    last_valid_indices = route_lengths - 1
    first_valid_indices = jnp.zeros_like(route_lengths)

    # Get end stops based on direction
    end_stops = jnp.where(
        current_directions == VehicleDirection.FORWARD,
        state.routes.stops[route_ids, last_valid_indices],  # last stop for forward
        state.routes.stops[route_ids, first_valid_indices],  # first stop for backward
    )
    opposite_end_stops = jnp.where(
        current_directions != VehicleDirection.FORWARD,
        state.routes.stops[route_ids, last_valid_indices],  # last stop for forward
        state.routes.stops[route_ids, first_valid_indices],  # first stop for backward
    )

    # TODO: handle case of no valid stop

    # prepare case seperation
    is_moving_forward = current_directions == VehicleDirection.FORWARD
    node_is_on_route = jnp.any(jnp.isfinite(route_times) & (route_times > 0), axis=1)
    node_is_on_route = node_is_on_route[route_ids, :]  # routes -> vehicles
    is_ahead_of_node = ~vehicle_is_ahead_of_node(state)
    is_moving_backwards = ~is_moving_forward
    is_after_node = ~is_ahead_of_node
    remaining_edge_time = (
        state.network.travel_times[current_positions, current_destinations]
        - state.fleet.times_on_edge
    )

    # for handling current node of vehicle if vehicle is at node
    nodes = jnp.arange(num_nodes)  # shape: (num_nodes,)
    at_current_node = nodes[None, :] == current_positions[:, None]
    is_at_node = state.fleet.is_at_node[:, None]  # Add dimension for broadcasting

    # We should distinguish three cases based on the number of turns needed to be at node i with
    # the vehicle moving forward (in route direction). The analogus we are interested in
    # three cases with the vehicle moving in the opposite direction of the route.
    #
    # For example we could have this route with the vehicle midway between edge 3 and 2:
    # 6 --- 3 --- 2 --- 5
    #          |
    #       vehicle ->
    #
    # Now we would like to know how long it takes until we would need to wait until the
    # vehicle arrives at our stop (could be any stop). Every edge has a travel time of 2 minutes
    # in this example.
    #
    # Then we need zero turns of the vehicles to arrive at nodes 2 and 5 in the correct direction
    # andit would take 1 and 3 minutes to reach the nodes.
    # For node 6 and 3 we need two turn to reach the nodes in forward direction, totaling a
    # waiting time of 9 and 11 minutes.
    # If we would like to travel in the backwards direction we need one turn no matter
    # which node we are at.

    # case0f: needs zero turn - forward
    is_case_0f = node_is_on_route & is_moving_forward[:, None] & is_ahead_of_node
    direct_times = jnp.where(
        at_current_node & is_at_node,
        0,
        route_times[route_ids, current_destinations, :] + remaining_edge_time[:, None],
    )
    wait_time_0f = jnp.where(is_case_0f, direct_times, jnp.inf)

    # case0b: needs zero turn - backward
    is_case_0b = node_is_on_route & is_moving_backwards[:, None] & is_ahead_of_node
    wait_time_0b = jnp.where(is_case_0b, direct_times, jnp.inf)

    # case1f: needs one turn  - forward
    is_case_1f = node_is_on_route & is_moving_backwards[:, None]
    forward = route_times[route_ids, current_destinations, end_stops] + remaining_edge_time
    backward = route_times[route_ids, end_stops, :]
    wait_time_1f = jnp.where(is_case_1f, forward[:, None] + backward, jnp.inf)

    # case1b: needs one turn  - backward
    is_case_1b = node_is_on_route & is_moving_forward[:, None]
    wait_time_1b = jnp.where(is_case_1b, forward[:, None] + backward, jnp.inf)

    # case2f: needs two turns - forward
    is_case_2f = node_is_on_route & is_moving_forward[:, None] & is_after_node
    forward_1 = route_times[route_ids, current_destinations, end_stops] + remaining_edge_time
    backward = route_times[route_ids, end_stops, opposite_end_stops]
    forward_2 = route_times[route_ids, opposite_end_stops, :]
    wait_time_2f = jnp.where(
        is_case_2f,
        forward_2 + forward_1[:, None] + backward[:, None],
        jnp.inf,
    )

    # case2b: needs two turns - backward
    is_case_2b = node_is_on_route & is_moving_backwards[:, None] & is_after_node
    wait_time_2b = jnp.where(
        is_case_2b,
        forward_2 + forward_1[:, None] + backward[:, None],
        jnp.inf,
    )

    # Save time into results matrix
    num_nodes = len(state.network.is_terminal)
    num_vehicles = len(route_ids)
    times = jnp.full((num_vehicles, num_nodes, 2), jnp.inf)

    times_forward = jnp.select(
        [is_case_0f, is_case_1f, is_case_2f],
        [wait_time_0f, wait_time_1f, wait_time_2f],
        jnp.inf,
    )

    times_backwards = jnp.select(
        [is_case_0b, is_case_1b, is_case_2b],
        [wait_time_0b, wait_time_1b, wait_time_2b],
        jnp.inf,
    )

    times = times.at[:, :, VehicleDirection.FORWARD].set(times_forward)
    times = times.at[:, :, VehicleDirection.BACKWARDS].set(times_backwards)

    return times


def vehicle_is_ahead_of_node(state: State) -> Bool[Array, "num_routes num_nodes"]:
    # Get vehicle data
    route_ids = state.fleet.route_ids
    next_node = state.fleet.current_edges[:, 1]
    is_moving_forward = state.fleet.directions == VehicleDirection.FORWARD

    # Get routes each vehicle is following
    routes = state.routes.stops[route_ids]  # shape: (num_vehicles, max_route_length)

    # Find indices of next positions in routes
    next_pos_matches = routes == next_node[:, None]  # shape: (num_vehicles, max_route_length)
    curr_pos_found = next_pos_matches.any(axis=1)
    next_pos_indices = jnp.argmax(next_pos_matches, axis=1)
    next_pos_indices = jnp.where(curr_pos_found, next_pos_indices, -1)  # -1 indicates not found

    # Find indices of all nodes in routes for each vehicle
    num_nodes = len(state.network.is_terminal)
    nodes = jnp.arange(num_nodes)  # shape: (num_nodes,)
    routes_expanded = routes[:, None, :]  # shape: (num_vehicles, 1, max_route_length)
    nodes_expanded = nodes[None, :, None]  # shape: (1, num_nodes, 1)
    node_matches = (
        routes_expanded == nodes_expanded
    )  # shape: (num_vehicles, num_nodes, max_route_length)
    node_found = node_matches.any(axis=2)
    node_indices = jnp.argmax(node_matches, axis=2)
    node_indices = jnp.where(node_found, node_indices, -1)  # -1 indicates not found

    # Determine if vehicle is ahead of passenger's node
    valid_mask = (next_pos_indices[:, None] >= 0) & (node_indices >= 0)
    is_ahead = jnp.where(
        valid_mask,
        jnp.where(
            is_moving_forward[:, None],
            next_pos_indices[:, None]
            > node_indices,  # Moving forward: ahead if current index > node index
            next_pos_indices[:, None]
            < node_indices,  # Moving backward: ahead if current index < node index
        ),
        False,  # If indices are invalid, not ahead
    )

    # Set False for the node the vehicle is currently at
    current_node = state.fleet.current_edges[:, 0]
    is_at_node = state.fleet.is_at_node[:, None]  # Add dimension for broadcasting
    is_at_current_node = nodes[None, :] == current_node[:, None]
    is_ahead = jnp.where(is_at_node & is_at_current_node, False, is_ahead)

    return is_ahead


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
    has_transfered = state.passengers.has_transferred[passengers]

    # Mask out invalid passengers (not in vehicle)
    dest_node_passenger_in_vehicle = jnp.where(passenger_ids_masked != -1, destinations, -1)
    transfer_node_passenger_in_vehicle = jnp.where(passenger_ids_masked != -1, transfer_nodes, -1)

    # Check if at destination or transfer node
    current_node = state.fleet.current_edges[:, 0][:, None]  # shape: (num_vehicles, max_capacity)
    passenger_is_at_dest = dest_node_passenger_in_vehicle == current_node
    passenger_is_at_transfer_and_no_prev_transfer = (
        transfer_node_passenger_in_vehicle == current_node
    ) & ~has_transfered

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
