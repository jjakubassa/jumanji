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
    def num_nodes(self) -> int:
        return len(self.is_terminal)

    def is_connected(
        self,
        from_node: Int[Array, "..."],
        to_node: Int[Array, "..."],
    ) -> Bool[Array, "..."]:
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
    num_flex_routes: int
    num_fix_routes: int

    @property
    def num_routes(self) -> int:
        return self.num_flex_routes + self.num_fix_routes

    @property
    def max_route_length(self) -> int:
        result = self.stops.shape[1]
        assert isinstance(result, int)
        return result

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

    @property
    def last_stops(self) -> Int[Array, " num_routes"]:
        """
        Retrieve the last valid stop for each route.

        Returns:
            An integer array of shape (num_routes,) containing the last valid
            stop node for each route.
        """
        valid_stop_counts = jnp.sum(self.stops != -1, axis=1)
        last_stop_indices = valid_stop_counts - 1
        route_indices = jnp.arange(self.num_routes)
        last_stops = self.stops[route_indices, last_stop_indices]
        return last_stops

    @property
    def last_stops_flex_routes(self) -> Int[Array, " num_flex_routes"]:
        """
        Retrieve the last valid stop for each flexible route.

        Returns:
            An integer array of shape (num_flex_routes,) containing the last
            valid stop node for each flexible route.
        """
        flex_indices = jnp.where(self.types == RouteType.FLEXIBLE, size=self.num_flex_routes)[0]
        last_stops_all = self.last_stops  # Shape: (num_routes,)
        last_stops_flex = last_stops_all[flex_indices]  # Shape: (num_flex_routes,)
        return last_stops_flex


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

    def __post_init__(self) -> None:
        if self.route_ids.shape[0] == 0:
            raise ValueError("Fleet cannot be empty")
        if self.passengers.shape[1] == 0:
            raise ValueError("Fleet must have at max_capacity > 0")

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

    @property
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

    network: NetworkData
    fleet: Fleet
    passengers: Passengers
    routes: RouteBatch
    current_time: Float[Array, ""]
    key: PRNGKeyArray

    def step(self, actions: Int[Array, " ..."]) -> tuple["State", TimeStep]:
        """
        Advance the simulation by one timestep using the provided actions.

        Args:
            actions: Array of actions for flexible routes.

        Returns:
            Tuple containing the new State instance and a TimeStep instance.
        """
        raise NotImplementedError

    def move_vehicles(self) -> "State":
        """Move all vehicles according to their routes and update positions."""
        # First increment times_on_edge for all vehicles
        new_times = self.fleet.times_on_edge + 1.0

        # Get travel times for current edges
        current_from = self.fleet.current_edges[:, 0]
        current_to = self.fleet.current_edges[:, 1]
        travel_times = jax.vmap(self.network.get_travel_time)(current_from, current_to)

        # Check which vehicles have completed their current edge
        completed_edge = new_times >= travel_times

        # Compute updates for all vehicles
        new_edges, new_directions = self._update_completed_vehicles(
            self.fleet.route_ids,
            self.fleet.current_edges,
            self.fleet.directions,
            completed_edge,
        )

        # Reset times_on_edge for vehicles that completed their edge
        new_times = jnp.where(completed_edge, 0.0, new_times)

        # Create updated fleet
        new_fleet = replace(
            self.fleet,
            current_edges=new_edges,
            times_on_edge=new_times,
            directions=new_directions,
        )

        return replace(self, fleet=new_fleet)

    def _update_completed_vehicles(
        self,
        route_ids: Int[Array, " {self.fleet.num_vehicles}"],
        current_edges: Int[Array, "{self.fleet.num_vehicles} 2"],
        directions: Int[Array, " {self.fleet.num_vehicles}"],
        completed: Bool[Array, " {self.fleet.num_vehicles}"],
    ) -> tuple[Int[Array, "num_vehicles 2"], Int[Array, " {self.fleet.num_vehicles}"]]:
        # Get route information
        route_types = self.routes.types[route_ids]
        routes = self.routes.stops[route_ids]

        # For completed edges, determine if we need to reverse direction
        current_stops = current_edges[:, 1]  # Use destination of current edge
        is_fixed = route_types == RouteType.FIXED
        is_at_end = current_stops == routes[:, -1]  # Check if at last stop
        is_at_start = current_stops == routes[:, 0]  # Check if at first stop

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
        self,
    ) -> tuple[
        Float[Array, "num_routes num_nodes num_nodes"],
        Bool[Array, "num_routes num_nodes num_nodes"],
    ]:
        """
        Calculate travel times between all pairs of stops for each route and track direction.
        Returns both times and whether each time is from a forward pass (True) or backward pass
        (False).
        """
        num_nodes = self.network.num_nodes
        max_route_length = self.routes.max_route_length

        def calculate_single_route_times(
            route: Int[Array, " max_route_length"], route_type: Int[Array, ""]
        ) -> tuple[Float[Array, " num_nodes num_nodes"], Bool[Array, " num_nodes num_nodes"]]:
            # Initialize result matrices
            result_times = jnp.full((num_nodes, num_nodes), jnp.inf)
            result_times = result_times.at[jnp.arange(num_nodes), jnp.arange(num_nodes)].set(0.0)
            # True for forward direction, False for backward
            result_directions = jnp.zeros((num_nodes, num_nodes), dtype=bool)

            def accumulate_fn(
                i: Int[Array, ""], accumulated_times: Float[Array, " max_route_length"]
            ) -> Float[Array, " max_route_length"]:
                src = route[i]
                dest = route[i + 1]
                is_valid = (src >= 0) & (dest >= 0)
                segment_time = self.network.travel_times[src, dest]
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
                result: tuple[
                    Float[Array, " num_nodes num_nodes"], Bool[Array, " num_nodes num_nodes"]
                ],
            ) -> tuple[Float[Array, " num_nodes num_nodes"], Bool[Array, " num_nodes num_nodes"]]:
                times, directions = result
                src = route[i]
                is_valid_src = src >= 0

                def update_to_stop(
                    j: Int[Array, ""],
                    res: tuple[
                        Float[Array, " num_nodes num_nodes"], Bool[Array, " num_nodes num_nodes"]
                    ],
                ) -> tuple[
                    Float[Array, " num_nodes num_nodes"], Bool[Array, " num_nodes num_nodes"]
                ]:
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
                        jnp.where(is_valid, True, curr_directions[src, dest])
                    )

                    # For fixed routes, mirror the times with backward direction
                    is_fixed = route_type == RouteType.FIXED
                    curr_times = curr_times.at[dest, src].set(
                        jnp.where(is_valid & is_fixed, time, curr_times[dest, src])
                    )
                    curr_directions = curr_directions.at[dest, src].set(
                        jnp.where(is_valid & is_fixed, False, curr_directions[dest, src])
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
        result = batch_calculate(self.routes.stops, self.routes.types)
        assert isinstance(result, tuple)
        return result

    def calculate_journey_times(
        self, passenger_origin: Int[Array, ""], passenger_dest: Int[Array, ""]
    ) -> Float[Array, " {self.fleet.num_vehicles}"]:
        """Calculate journey times using pre-computed direction information."""
        route_times, route_directions = self.calculate_route_times()

        # Get vehicle data
        route_ids = self.fleet.route_ids
        current_positions = self.fleet.current_edges[:, 0]
        current_destinations = self.fleet.current_edges[:, 1]
        current_directions = self.fleet.directions

        # Get direction required to reach passenger
        required_directions = route_directions[route_ids, current_positions, passenger_origin]

        # Going wrong direction if current direction doesn't match required direction
        going_wrong_direction = current_directions != required_directions

        # Calculate times...
        direct_times = route_times[route_ids, current_positions, passenger_origin]
        end_stops = jnp.where(
            current_directions == VehicleDirection.FORWARD,
            self.routes.stops[route_ids, -1],  # last stop for forward
            self.routes.stops[route_ids, 0],  # first stop for backward
        )

        time_to_end = route_times[route_ids, current_positions, end_stops]
        time_back = route_times[route_ids, end_stops, passenger_origin]

        remaining_travel_time = jnp.where(
            current_destinations == passenger_origin,
            self.network.travel_times[current_positions, passenger_origin]
            - self.fleet.times_on_edge,
            jnp.where(
                self.routes.types[route_ids] == RouteType.FIXED,
                jnp.where(going_wrong_direction, time_to_end + time_back, direct_times),
                direct_times,
            ),
        )

        time_until_available = jnp.where(
            current_positions == passenger_origin, 0.0, remaining_travel_time
        )

        in_vehicle_times = route_times[route_ids, passenger_origin, passenger_dest]
        journey_times = time_until_available + in_vehicle_times

        return journey_times

    def assign_passengers(self, state: "State") -> "State":
        """Optimistic assignment considering both current and future vehicle availability."""

        # Get all waiting passengers
        waiting_mask = state.passengers.statuses == PassengerStatus.WAITING

        # Calculate route times once
        route_times, route_directions = state.calculate_route_times()

        def assign_single_passenger(
            state: State,
            passenger_idx: Int[Array, ""],
            route_times: Float[Array, "num_routes num_nodes num_nodes"],
            route_directions: Bool[Array, "num_routes num_nodes num_nodes"],
        ) -> State:
            origin = state.passengers.origins[passenger_idx]
            dest = state.passengers.destinations[passenger_idx]

            # Get data for all vehicles
            route_ids = state.fleet.route_ids
            current_positions = state.fleet.current_edges[:, 0]
            current_destinations = state.fleet.current_edges[:, 1]
            current_directions = state.fleet.directions

            # Calculate immediate boarding opportunities
            is_at_correct_stop = state.fleet.is_at_node & (
                state.fleet.current_edges[:, 0] == origin
            )
            immediate_boarding_possible = is_at_correct_stop & state.fleet.seat_is_available

            # Going wrong direction if current direction doesn't match required direction
            going_wrong_direction = (
                current_directions != route_directions[route_ids, current_positions, origin]
            )

            # Find last valid stop for each route (before -1 padding)
            route_lengths = (state.routes.stops[route_ids] != -1).sum(axis=1)
            last_valid_indices = route_lengths - 1  # subtract 1 to get 0-based index
            first_valid_indices = jnp.zeros_like(route_lengths)

            # Get end stops based on direction
            end_stops = jnp.where(
                current_directions == VehicleDirection.FORWARD,
                # Forward: use last valid stop
                state.routes.stops[route_ids, last_valid_indices],
                # Backward: use first valid stop
                state.routes.stops[route_ids, first_valid_indices],
            )

            time_to_end = route_times[route_ids, current_positions, end_stops]
            time_back = route_times[route_ids, end_stops, origin]

            direct_times = route_times[route_ids, current_positions, origin]
            remaining_travel_time = jnp.where(
                current_destinations == origin,
                state.network.travel_times[current_positions, origin] - state.fleet.times_on_edge,
                jnp.where(
                    (state.routes.types[route_ids] == RouteType.FIXED) & going_wrong_direction,
                    time_to_end + time_back,  # Must go to end and come back
                    direct_times,  # Direct path for all other cases
                ),
            )

            # Calculate total journey times
            in_vehicle_times = route_times[route_ids, origin, dest]  # [num_vehicles]
            total_times = remaining_travel_time + in_vehicle_times

            # Check route validity
            connects_od = jnp.isfinite(in_vehicle_times)

            # Calculate immediate and future options
            immediate_times = jnp.where(
                immediate_boarding_possible & connects_od, total_times, jnp.inf
            )
            future_times = jnp.where(
                connects_od & state.fleet.seat_is_available, total_times, jnp.inf
            )

            # Board if:
            # - immediate option exists AND
            # - no significantly better future option exists
            best_future_time = jnp.min(future_times)
            should_board = jnp.isfinite(immediate_times) & (best_future_time >= immediate_times)

            # Among valid options, pick one with shortest travel time
            valid_times = jnp.where(should_board, immediate_times, jnp.inf)
            best_time = jnp.min(valid_times)
            has_best_time = valid_times == best_time

            # Among those with best time, pick one with highest capacity
            # If still undecided, argmax will be the first indice
            best_vehicle = jnp.argmax(has_best_time * state.fleet.capacities_left)

            def update_state(state: State) -> State:
                new_fleet = state.fleet.add_passenger(best_vehicle, passenger_idx)
                new_passengers = state.passengers.update_passengers(
                    state.current_time,
                    jnp.array([passenger_idx]) == jnp.arange(state.passengers.num_passengers),
                    jnp.zeros_like(state.passengers.statuses, dtype=bool),
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

    def get_action_mask(
        self,
    ) -> Bool[Array, "{self.routes.num_flex_routes} {self.network.num_nodes+1}"]:
        action_mask = jnp.zeros(
            (self.routes.num_flex_routes, self.network.num_nodes + 1), dtype=bool
        )
        last_stops = self.routes.last_stops_flex_routes  # Shape: (num_flex_routes,)
        node_indices = jnp.arange(self.network.num_nodes)  # Shape: (num_nodes,)

        # Compute connected nodes
        connected_nodes = self.network.is_connected(last_stops[:, None], node_indices[None, :])

        # Exclude the last stop from the allowed actions
        is_not_last_stop = last_stops[:, None] != node_indices[None, :]
        connected_nodes = connected_nodes & is_not_last_stop  # Element-wise

        # Always allow the no-op action (e.g., doing nothing)
        no_op = jnp.ones((self.routes.num_flex_routes, 1), dtype=bool)

        # Concatenate the connected_nodes with the no_op to form the action_mask
        allowed_actions = jnp.concatenate([connected_nodes, no_op], axis=1)

        # For routes where last_stops == -1, allow all actions
        initial_routes = (last_stops == -1)[:, None]
        all_actions = jnp.ones_like(allowed_actions, dtype=bool)
        action_mask = jnp.where(initial_routes, all_actions, allowed_actions)

        return action_mask

    def get_observation(self) -> "Observation":
        """Creates observation from current state."""

        return Observation(
            network=self.network,
            routes=self.routes,
            fleet_positions=self.fleet.current_edges,
            origins=self.passengers.origins,
            destinations=self.passengers.destinations,
            desired_departure_times=self.passengers.desired_departure_times,
            passenger_statuses=self.passengers.statuses,
            current_time=float(self.current_time),
            action_mask=self.get_action_mask(),
        )


@dataclass
class Observation:
    """
    Represents the observable state provided to the agent.
    """

    network: "NetworkData"
    routes: "RouteBatch"
    fleet_positions: Float[Array, " num_vehicles 2"]
    origins: Int[Array, " num_passengers"]
    destinations: Int[Array, " num_passengers"]
    desired_departure_times: Float[Array, " num_passengers"]
    passenger_statuses: Int[Array, " num_passengers"]
    current_time: float
    action_mask: Bool[Array, "{self.routes.num_flex_routes} {self.netwowrk.num_nodes+1}"]


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
