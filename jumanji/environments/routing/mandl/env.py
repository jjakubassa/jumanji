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
from functools import cached_property
from typing import Final, Literal, Optional

import chex
import jax
import jax.numpy as jnp
import matplotlib
from beartype.typing import Sequence
from jaxtyping import Array, Bool, Int

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.types import (
    Observation,
    PassengerStatus,
    RouteBatch,
    RouteType,
    State,
    assign_passengers,
    calculate_route_times,
    find_best_transfer_route,
    floyd_warshall,
    get_last_stops,
    handle_completed_and_transferring_passengers,
    increment_in_vehicle_times,
    increment_wait_times,
    is_connected,
    move_vehicles,
    update_passengers_to_waiting,
    update_routes,
)
from jumanji.environments.routing.mandl.utils import (
    assign_routes_to_fleet,
    create_initial_fleet,
    create_initial_passengers,
    create_initial_routes,
    load_demand_data,
    load_network_data,
    load_solution_data,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Mandl(Environment[State, specs.BoundedArray, Observation]):
    def __init__(
        self,
        viewer: Optional[Viewer] = None,
        network_name: Literal["mandl1", "ceder1"] = "mandl1",
        runtime: float = 150.0,
        buffer_time_end: float = 100.0,
        buffer_time_start: float = 8,
        vehicle_capacity: int = 50,
        solution_name: Optional[str] = None,  # None means no solution from file
        num_fix_routes: int = 1,
        num_flex_routes: int = 16,
        max_route_length: int = 8,
        allow_actions_fixed_routes: bool = True,
        total_vehicles: int = 99,
        passenger_init_mode: Literal[
            "evenly_spaced", "rush_hour", "uniform_random", "all_at_start"
        ] = "evenly_spaced",
    ) -> None:
        self.network_name: Final = network_name
        self.runtime: Final = runtime
        self.num_flex_routes: Final = num_flex_routes
        self.passenger_init_mode: Final = passenger_init_mode
        self.vehicle_capacity: Final = vehicle_capacity
        self.buffer_time_start: Final = buffer_time_start
        self.buffer_time_end: Final = buffer_time_end
        self.total_vehicles: Final = total_vehicles
        self.allow_actions_fixed_routes: Final = allow_actions_fixed_routes
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )

        # Load all static data once during initialization
        self._network_data = load_network_data(network_name)
        self._routes: list[list[int]] = []
        if solution_name is not None:
            self._routes, vehicles_per_solution_route = load_solution_data(solution_name)
            # num_fix_routes now represents additional fixed routes beyond solution
            self.num_fix_routes = len(self._routes) + max(0, num_fix_routes)
            print(f"\nUsing solution with {len(self._routes)} routes")
            print(f"Adding {num_fix_routes} additional fixed routes")
            print(f"Total fixed routes: {self.num_fix_routes}")
        else:
            # If no solution specified, num_fix_routes is the total number of fixed routes
            self.num_fix_routes = max(0, num_fix_routes)
            vehicles_per_solution_route = []
            print("\nNo solution used")
            print(f"Creating {self.num_fix_routes} fixed routes")

        self._vehicles_per_route: tuple[int, ...]

        # Check if solution routes exceed max_stops
        max_solution_length = max(len(route) for route in self._routes) if self._routes else 0
        if max_solution_length > max_route_length:
            print(
                f"WARNING: Solution routes contain up to {max_solution_length} stops, "
                f"which exceeds the specified max_stops={max_route_length}. "
                f"Using {max_solution_length} as maximum."
            )
        self.max_route_length: Final = max(max_solution_length, max_route_length)

        # Create static components once
        self._route_batch = create_initial_routes(
            self._routes,
            num_fix_routes=self.num_fix_routes,
            num_flex_routes=self.num_flex_routes,
            network_data=self._network_data,  # Pass network data
            max_stops=self.max_route_length,
            key=None,  # Pass random key
        )

        self._initial_fleet, self._vehicles_per_route = create_initial_fleet(
            num_routes=self.num_fix_routes + self.num_flex_routes,
            num_flex_routes=self.num_flex_routes,
            total_vehicles=self.total_vehicles,
            vehicles_per_solution_route=vehicles_per_solution_route,
            vehicle_capacity=self.vehicle_capacity,
        )

        # Load passenger demand data
        self._demand_data = load_demand_data(network_name)

        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state."""
        initial_state = State(
            network=self._network_data,
            fleet=self._initial_fleet,
            passengers=create_initial_passengers(
                self._demand_data,
                key,
                runtime=self.runtime,
                buffer_time_start=self.buffer_time_start,
                buffer_time_end=self.buffer_time_end,
                mode=self.passenger_init_mode,
            ),
            routes=self._route_batch,
            current_time=jnp.array(0.0),
            key=key,
        )

        timestep = restart(
            observation=self.get_observation(initial_state),
            extras=self._calculate_metrics(initial_state),
        )

        return initial_state, timestep

    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep]:
        """
        Advance the simulation by one timestep using the provided actions.

        Args:
            state: Current state of the environment
            actions: Array of actions for flexible routes.

        Returns:
            Tuple containing the new State instance and a TimeStep instance.
        """
        # 1. Update routes according to action
        new_routes = update_routes(state.routes, state.network.num_nodes, action)
        state = replace(state, routes=new_routes)

        # 2. Check if grace period is over and fleet needs route assignment
        is_grace_period_over = state.current_time >= self.buffer_time_start
        needs_route_assignment = jnp.all(state.fleet.route_ids == -1)

        def assign_routes(state: State) -> State:
            """Assign routes to unassigned fleet using pre-calculated vehicle allocations."""
            new_fleet = assign_routes_to_fleet(
                state.fleet,
                state.routes,
                self._network_data,
                self._vehicles_per_route,  # Use pre-calculated vehicle allocations
                self.max_route_length,
            )
            return replace(state, fleet=new_fleet)

        state = jax.lax.cond(
            is_grace_period_over & needs_route_assignment,
            assign_routes,
            lambda s: s,
            state,
        )

        # 3. Move vehicles to new positions
        state = move_vehicles(state)

        # 4. Update passenger times
        new_passengers = increment_wait_times(state.passengers)
        new_passengers = increment_in_vehicle_times(new_passengers)
        state = replace(state, passengers=new_passengers)

        # 5. Handle completed and transferring passengers
        state = handle_completed_and_transferring_passengers(state)

        # 6. Update passenger statuses based on current time
        new_passengers = update_passengers_to_waiting(state.passengers, state.current_time)
        state = replace(state, passengers=new_passengers)

        # 7. Assign waiting passengers to vehicles
        state = assign_passengers(state)

        # 8. Calculate reward
        reward = -jnp.sum(
            state.passengers.time_waiting + state.passengers.time_in_vehicle, dtype=jnp.float32
        )

        # 9. Check if episode is done
        done = self._is_done(state)

        # 10. Increase simulation time
        new_time = state.current_time + 1.0
        state = replace(state, current_time=new_time)

        # 11. Create timestep
        obs = self.get_observation(state)
        metrics = self._calculate_metrics(state)
        timestep = jax.lax.cond(
            done,
            lambda: termination(observation=obs, reward=reward, extras=metrics),
            lambda: transition(observation=obs, reward=jnp.array(0.0), extras=metrics),
        )

        return state, timestep

    def _calculate_metrics(self, state: State) -> dict:
        """Calculate summary metrics at the end of an episode."""
        metrics = {}

        # 1. Passenger status counts
        not_in_system = jnp.sum(state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM)
        waiting = jnp.sum(state.passengers.statuses == PassengerStatus.WAITING)
        in_vehicle = jnp.sum(state.passengers.statuses == PassengerStatus.IN_VEHICLE)
        completed = jnp.sum(state.passengers.statuses == PassengerStatus.COMPLETED)
        total_passengers = state.passengers.num_passengers

        metrics["not_in_system_passengers"] = not_in_system
        metrics["waiting_passengers"] = waiting
        metrics["in_vehicle_passengers"] = in_vehicle
        metrics["completed_passengers"] = completed
        metrics["completion_rate"] = completed / jnp.maximum(total_passengers, 1)

        # 2. Waiting time statistics (for all passengers who have entered the system)
        in_system_mask = state.passengers.statuses != PassengerStatus.NOT_IN_SYSTEM
        in_system_count = jnp.sum(in_system_mask)

        waiting_times = state.passengers.time_waiting * in_system_mask  # Zeros for not-in-system
        metrics["total_waiting_time"] = jnp.sum(waiting_times)
        metrics["avg_waiting_time"] = jnp.sum(waiting_times) / jnp.maximum(in_system_count, 1)
        metrics["max_waiting_time"] = jnp.max(waiting_times)

        # 3. In-vehicle time statistics
        in_vehicle_times = state.passengers.time_in_vehicle * (state.passengers.time_in_vehicle > 0)
        in_vehicle_count = jnp.sum(state.passengers.time_in_vehicle > 0)

        metrics["total_in_vehicle_time"] = jnp.sum(in_vehicle_times)
        metrics["avg_in_vehicle_time"] = jnp.sum(in_vehicle_times) / jnp.maximum(
            in_vehicle_count, 1
        )
        metrics["max_in_vehicle_time"] = jnp.max(in_vehicle_times)

        # 4. Transfer statistics
        transfers = state.passengers.has_transferred
        metrics["total_transfers"] = jnp.sum(transfers)
        metrics["avg_transfers_per_passenger"] = jnp.sum(transfers) / jnp.maximum(
            in_system_count, 1
        )

        # 5. Vehicle utilization
        capacity_per_vehicle = state.fleet.passengers.shape[1]
        vehicle_occupancy = jnp.sum(state.fleet.passengers != -1, axis=1)
        metrics["avg_vehicle_utilization"] = jnp.mean(vehicle_occupancy) / capacity_per_vehicle
        metrics["percent_empty_vehicles"] = jnp.mean(vehicle_occupancy == 0)
        metrics["percent_full_vehicles"] = jnp.mean(vehicle_occupancy == capacity_per_vehicle)

        # 6. Travel distance metrics for completed passengers (using in-vehicle time as proxy)
        completed_mask = state.passengers.statuses == PassengerStatus.COMPLETED
        completed_count = jnp.sum(completed_mask)

        completed_distances = state.passengers.time_in_vehicle * completed_mask
        metrics["max_completed_distance"] = jnp.max(completed_distances)
        # Use a high value (inf) for min with a mask to avoid getting 0 as minimum
        min_distance = jnp.min(jnp.where(completed_mask, state.passengers.time_in_vehicle, jnp.inf))
        # Replace inf with 0 if no completions
        metrics["min_completed_distance"] = jnp.where(completed_count > 0, min_distance, 0.0)
        metrics["mean_completed_distance"] = jnp.sum(completed_distances) / jnp.maximum(
            completed_count, 1
        )

        # 7. Waiting time at transfer stop
        transfer_mask = state.passengers.has_transferred
        transfer_count = jnp.sum(transfer_mask)

        transfer_waiting_times = state.passengers.time_waiting * transfer_mask
        metrics["avg_transfer_waiting_time"] = jnp.sum(transfer_waiting_times) / jnp.maximum(
            transfer_count, 1
        )

        # 8. Total travel time (waiting + in-vehicle)
        total_travel_time = jnp.sum(waiting_times) + jnp.sum(in_vehicle_times)
        metrics["total_travel_time"] = total_travel_time

        return metrics

    def _get_empty_metrics(self) -> dict:
        """Return empty metrics dictionary with correct structure."""
        return {
            "not_in_system_passengers": jnp.array(0),
            "waiting_passengers": jnp.array(0),
            "in_vehicle_passengers": jnp.array(0),
            "completed_passengers": jnp.array(0),
            "completion_rate": jnp.array(0.0),
            "total_waiting_time": jnp.array(0.0),
            "avg_waiting_time": jnp.array(0.0),
            "max_waiting_time": jnp.array(0.0),
            "total_in_vehicle_time": jnp.array(0.0),
            "avg_in_vehicle_time": jnp.array(0.0),
            "max_in_vehicle_time": jnp.array(0.0),
            "total_transfers": jnp.array(0),
            "avg_transfers_per_passenger": jnp.array(0.0),
            "avg_vehicle_utilization": jnp.array(0.0),
            "percent_empty_vehicles": jnp.array(0.0),
            "percent_full_vehicles": jnp.array(0.0),
            "max_completed_distance": jnp.array(0.0),
            "min_completed_distance": jnp.array(0.0),
            "mean_completed_distance": jnp.array(0.0),
            "avg_transfer_waiting_time": jnp.array(0.0),
            "total_travel_time": jnp.array(0.0),
        }

    def _is_done(self, state: State) -> Bool[Array, ""]:
        """Check if episode is done.

        Episode ends when either:
        1. We've reached runtime
        2. All passengers have completed their journeys
        """
        time_done = state.current_time >= self.runtime - 1  # account for timestep t=0 -> t=1
        passengers_done = jnp.all(state.passengers.statuses == PassengerStatus.COMPLETED)
        return time_done | passengers_done

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec."""
        num_nodes = len(self._network_data.is_terminal)
        num_routes = self._route_batch.num_routes
        max_route_length = self.max_route_length
        num_vehicles = self._initial_fleet.num_vehicles

        return specs.Spec(
            Observation,
            "ObservationSpec",
            # Network data specs
            num_nodes=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=num_nodes,
            ),
            travel_times=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            is_terminal=specs.BoundedArray(
                shape=(num_nodes,),
                dtype=bool,
                minimum=False,
                maximum=True,
            ),
            # Route data specs
            num_routes=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=num_routes,
            ),
            max_route_length=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=max_route_length,
            ),
            route_types=specs.BoundedArray(
                shape=(num_routes,),
                dtype=int,
                minimum=0,
                maximum=1,  # RouteType.FIXED or RouteType.FLEXIBLE
            ),
            route_stops=specs.BoundedArray(
                shape=(num_routes, max_route_length),
                dtype=int,
                minimum=-1,  # -1 for padding
                maximum=num_nodes - 1,
            ),
            route_frequencies=specs.BoundedArray(
                shape=(num_routes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            num_flex_routes=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=num_routes,
            ),
            num_fix_routes=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=num_routes,
            ),
            direct_travel_times=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            transfer_travel_times=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            network_shortest_times=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            # Fleet data spec
            num_vehicles=specs.BoundedArray(
                shape=(),
                dtype=int,
                minimum=0,
                maximum=num_vehicles,
            ),
            fleet_positions=specs.BoundedArray(
                shape=(self._initial_fleet.num_vehicles, 2),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
            ),
            # Aggregated passenger demand specs
            future_demand=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            waiting_demand=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            transferring_demand=specs.BoundedArray(
                shape=(num_nodes * num_nodes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
            ),
            # Environment state specs
            current_time=specs.BoundedArray(
                shape=(),
                dtype=float,
                minimum=0.0,
                maximum=self.runtime,
            ),
            action_mask=specs.BoundedArray(
                shape=(num_routes, num_nodes + 1),
                dtype=bool,
                minimum=False,
                maximum=True,
            ),
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for flexible routes.

        The action space consists of one action per flexible route.
        Each action is an integer indicating:
        - Which node to add as the next stop (0 to num_nodes-1)
        - Or perform no-op (num_nodes)
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.full(
                shape=(self.num_flex_routes + self.num_fix_routes,),
                fill_value=self._network_data.num_nodes
                + 1,  # num_nodes + 1 possible actions per route
                dtype=jnp.int32,
            ),
            dtype=jnp.int32,
            name="actions",
        )

    def render(self, state: State) -> Optional[chex.ArrayNumpy]:
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self._viewer is not None:
            self._viewer.close()

    def get_observation(self, state: State) -> Observation:
        """Creates observation from current state with aggregated passenger information."""
        num_nodes = len(state.network.is_terminal)

        # Create passenger demand matrix (num_nodes x num_nodes)
        waiting_demand = jnp.zeros((num_nodes, num_nodes))
        transferring_demand = jnp.zeros((num_nodes, num_nodes))

        # Aggregate waiting passengers by OD pair
        future_mask = state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM
        waiting_mask = state.passengers.statuses == PassengerStatus.WAITING
        transferring_mask = state.passengers.statuses == PassengerStatus.TRANSFERRING
        origins = state.passengers.origins
        destinations = state.passengers.destinations

        future_demand = waiting_demand.at[origins, destinations].add(future_mask)
        waiting_demand = waiting_demand.at[origins, destinations].add(waiting_mask)
        transferring_demand = transferring_demand.at[origins, destinations].add(transferring_mask)

        # Calculate travel times
        route_times, route_directions = calculate_route_times(state)
        direct_times = jnp.min(route_times, axis=0)
        transfer_times = jax.vmap(
            jax.vmap(
                lambda o, d: find_best_transfer_route(state, o, d, route_times)[0],
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(jnp.arange(num_nodes), jnp.arange(num_nodes))
        network_shortest_times = floyd_warshall(state.network.travel_times)

        return Observation(
            # Network data
            num_nodes=jnp.array(num_nodes),
            travel_times=state.network.travel_times.flatten(),
            is_terminal=state.network.is_terminal,
            # Routes data
            num_routes=state.routes.num_routes,
            max_route_length=jnp.array(self.max_route_length),
            route_types=state.routes.types,
            route_stops=state.routes.stops,
            route_frequencies=state.routes.frequencies,
            num_flex_routes=state.routes.num_flex_routes,
            num_fix_routes=state.routes.num_fix_routes,
            direct_travel_times=direct_times.flatten(),
            transfer_travel_times=transfer_times.flatten(),
            network_shortest_times=network_shortest_times.flatten(),
            # Fleet data
            num_vehicles=jnp.array(state.fleet.num_vehicles),
            fleet_positions=state.fleet.current_edges,
            # Aggregated passenger data
            future_demand=future_demand.flatten(),
            waiting_demand=waiting_demand.flatten(),
            transferring_demand=transferring_demand.flatten(),
            # Environment state
            current_time=state.current_time,
            action_mask=self.get_action_mask(state),
        )

    def get_last_stops_flex_routes(self, routes: RouteBatch) -> Int[Array, " num_flex_routes"]:
        """Retrieve the last valid stop for each flexible route."""
        flex_indices = jnp.where(routes.types == RouteType.FLEXIBLE, size=self.num_flex_routes)[0]
        last_stops_all = get_last_stops(routes)  # Shape: (num_routes,)
        last_stops_flex = last_stops_all[flex_indices]  # Shape: (num_flex_routes,)
        return last_stops_flex

    def get_action_mask(
        self,
        state: State,
    ) -> Bool[Array, "num_routes num_nodes_plus_one"]:
        """Get action mask for all routes.

        For fixed routes, only allow no-op action.
        For flexible routes, allow connected nodes and no-op.
        No-op is disabled for the first two steps to force creation of at least one edge.
        """
        last_stops = get_last_stops(state.routes)  # Shape: (num_routes,)
        num_nodes = state.network.travel_times.shape[0]
        num_routes = state.routes.stops.shape[0]

        # Create an initial action mask with shape: (num_routes, num_nodes + 1)
        action_mask = jnp.zeros((num_routes, num_nodes + 1), dtype=bool)

        node_indices = jnp.arange(num_nodes)  # Shape: (num_nodes,)

        # Compute connected nodes for all routes
        # This creates a (num_routes, num_nodes) boolean matrix
        connected_nodes = is_connected(state.network, last_stops[:, None], node_indices[None, :])

        # Exclude the last stop from the allowed actions
        is_not_last_stop = last_stops[:, None] != node_indices[None, :]
        connected_nodes = connected_nodes & is_not_last_stop

        # Check if we're in the first two steps (where last_stops is -1)
        is_first_two_steps = jnp.sum(state.routes.stops != -1, axis=1) < 2

        # Allow no-op only after first two steps
        no_op = jnp.ones((num_routes, 1), dtype=bool) & (~is_first_two_steps[:, None])

        # Combine connected nodes with no-op
        allowed_actions = jnp.concatenate([connected_nodes, no_op], axis=1)

        # For routes where last_stops == -1, allow all actions except no-op in first two steps
        initial_routes = (last_stops == -1)[:, None]
        all_actions = jnp.ones_like(allowed_actions, dtype=bool)
        all_actions = all_actions.at[:, -1].set(
            ~is_first_two_steps
        )  # Disable no-op for first two steps

        if self.allow_actions_fixed_routes:
            is_fixed_route = (state.routes.types == RouteType.FIXED)[:, None]
            fixed_route_mask = jnp.zeros_like(allowed_actions)
            fixed_route_mask = fixed_route_mask.at[:, -1].set(
                ~is_first_two_steps
            )  # Disable no-op for first two steps

            # Combine all masks
            action_mask = jnp.where(
                is_fixed_route,
                fixed_route_mask,  # Fixed routes: only no-op after first two steps
                jnp.where(
                    initial_routes,
                    all_actions,  # Initial routes: all actions except no-op in first two steps
                    allowed_actions,  # Other cases: connected nodes + no-op
                ),
            )
        else:
            # Treat all routes the same way
            action_mask = jnp.where(
                initial_routes,
                all_actions,  # Initial routes: all actions except no-op in first two steps
                allowed_actions,  # Other cases: connected nodes + no-op
            )
        return action_mask
