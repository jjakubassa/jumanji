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
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import matplotlib
from beartype.typing import Sequence
from jaxtyping import Array, Bool, Int

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.generator import DefaultGenerator, Generator
from jumanji.environments.routing.mandl.types import (
    Observation,
    PassengerStatus,
    RouteBatch,
    RouteType,
    State,
    assign_passengers,
    calculate_shortest_route_times,
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
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Mandl(Environment[State, specs.BoundedArray, Observation]):
    def __init__(
        self,
        generator: Optional[Generator] = None,
        viewer: Optional[Viewer] = None,
    ) -> None:
        """Initialize the Mandl environment.

        Args:
            generator: Generator for creating problem instances. If None, uses DefaultGenerator
                with default parameters.
            viewer: Viewer for rendering. If None, uses MandlViewer with "human" render mode.
            allow_actions_fixed_routes: Whether to allow actions on fixed routes.
        """
        # Initialize generator with defaults if none provided
        self.generator = generator or DefaultGenerator()
        self.allow_actions_fixed_routes = self.generator.allow_actions_fixed_routes

        # Initialize viewer
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )

        # Store important parameters from generator for easy access
        self.num_flex_routes = self.generator.num_flex_routes
        self.num_fix_routes = self.generator.num_fix_routes
        self.num_solution_routes = self.generator.num_solution_routes
        self.total_vehicles = self.generator.total_vehicles
        self.vehicle_capacity = self.generator.vehicle_capacity
        self.runtime = self.generator.runtime
        self.buffer_time_start = self.generator.buffer_time_start
        self.buffer_time_end = self.generator.buffer_time_end
        self.max_route_length = self.generator.max_route_length

        # Store network data and dimensions
        self._network_data = self.generator.network_data
        self._network_shortest_times = floyd_warshall(self._network_data.travel_times).flatten()

        # Store important dimensions
        self.num_nodes = len(self._network_data.is_terminal)
        self.num_routes = self.num_fix_routes + self.num_flex_routes + self.num_solution_routes
        self.num_vehicles = self.total_vehicles

        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state.

        Args:
            key: Random key for initialization.

        Returns:
            A tuple containing:
                - The initial state
                - The initial timestep with observation
        """
        # Generate initial state using the generator
        initial_state = self.generator(key)

        # Create initial timestep
        timestep = restart(
            observation=self.get_observation(initial_state),
            extras=self._calculate_metrics(initial_state, self.get_observation(initial_state)),
        )

        return initial_state, timestep

    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep]:
        """
        Advance the simulation by one timestep using the provided actions.
        During warmup (before buffer_time_start), only update routes.
        """
        # 1. Update routes according to action
        new_routes = update_routes(state.routes, state.network.num_nodes, action)
        state = replace(state, routes=new_routes)

        # Check if we're still in warmup period
        is_warmup = state.current_time < self.buffer_time_start

        # During warmup, skip vehicle and passenger updates
        def normal_step(state: State) -> State:
            # 2. Check if grace period is over and fleet needs route assignment
            needs_route_assignment = jnp.all(state.fleet.route_ids == -1)

            state = jax.lax.cond(
                needs_route_assignment,
                lambda s: replace(
                    s,
                    fleet=assign_routes_to_fleet(
                        s.fleet,
                        s.routes,
                        self._network_data,
                        s.routes.vehicles_per_route,
                        self.max_route_length,
                    ),
                ),
                lambda s: s,
                state,
            )

            # 3. Move vehicles
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

            return state

        def warmup_step(state: State) -> State:
            # During warmup, only update routes and time
            return state

        # Use warmup or normal step based on current time
        state = jax.lax.cond(
            is_warmup,
            warmup_step,
            normal_step,
            state,
        )

        # 8. Calculate reward (zero during warmup)
        reward = jax.lax.cond(
            is_warmup,
            lambda: jnp.array(0.0),
            lambda: -jnp.sum(
                state.passengers.time_waiting + state.passengers.time_in_vehicle, dtype=jnp.float32
            ),
        )

        # 9. Check if episode is done
        done = self._is_done(state)

        # 10. Increase simulation time
        new_time = state.current_time + 1.0
        state = replace(state, current_time=new_time)

        # 11. Create timestep
        obs = self.get_observation(state)
        metrics = self._calculate_metrics(state, obs)
        timestep = jax.lax.cond(
            done,
            lambda: termination(observation=obs, reward=reward, extras=metrics),
            lambda: transition(observation=obs, reward=reward, extras=metrics),
        )
        return state, timestep

    def _calculate_metrics(self, state: State, obs: Observation) -> dict:
        """Calculate summary metrics at the end of an episode."""
        metrics = {}

        # Passenger status counts
        completed = jnp.sum(state.passengers.statuses == PassengerStatus.COMPLETED)
        total_passengers = state.passengers.num_passengers
        metrics["completion_rate"] = completed / jnp.maximum(total_passengers, 1)

        # Waiting time
        waiting_times = state.passengers.time_waiting
        metrics["total_waiting_time"] = jnp.sum(waiting_times)
        metrics["avg_waiting_time"] = jnp.mean(waiting_times)
        metrics["max_waiting_time"] = jnp.max(waiting_times)

        # In-vehicle time
        in_vehicle_times = state.passengers.time_in_vehicle
        metrics["total_in_vehicle_time"] = jnp.sum(in_vehicle_times)
        metrics["avg_in_vehicle_time"] = jnp.mean(in_vehicle_times)
        metrics["max_in_vehicle_time"] = jnp.max(in_vehicle_times)

        # Total travel time (waiting + in-vehicle)
        metrics["total_travel_time"] = jnp.sum(waiting_times) + jnp.sum(in_vehicle_times)
        metrics["avg_total_travel_time"] = jnp.mean(waiting_times + in_vehicle_times)

        # Transfer statistics
        transfers = state.passengers.has_transferred
        metrics["total_transfers"] = jnp.sum(transfers)
        metrics["avg_transfers_per_passenger"] = jnp.mean(transfers)

        # Vehicle utilization
        capacity_per_vehicle = state.fleet.passengers.shape[1]
        vehicle_occupancy = jnp.sum(state.fleet.passengers != -1, axis=1)
        metrics["avg_vehicle_utilization"] = jnp.mean(vehicle_occupancy) / capacity_per_vehicle
        metrics["percent_empty_vehicles"] = jnp.mean(vehicle_occupancy == 0)
        metrics["percent_full_vehicles"] = jnp.mean(vehicle_occupancy == capacity_per_vehicle)

        # Route optimality
        shortest_times_flat = self._network_shortest_times
        is_finite_direct = jnp.isfinite(obs.direct_travel_times)
        is_finite_transfer = jnp.isfinite(obs.transfer_travel_times)
        is_finite_shortest = jnp.isfinite(shortest_times_flat)
        valid_direct = is_finite_direct & is_finite_shortest
        valid_transfer = is_finite_transfer & is_finite_shortest
        metrics["ratio_travel_time_direct_to_shortest_path"] = jnp.where(
            valid_direct.sum() > 0,
            (obs.direct_travel_times * valid_direct).sum()
            / (shortest_times_flat * valid_direct).sum(),
            jnp.inf,
        )
        metrics["ratio_travel_time_transfers_to_shortest_path"] = jnp.where(
            valid_transfer.sum() > 0,
            (obs.transfer_travel_times * valid_transfer).sum()
            / (shortest_times_flat * valid_transfer).sum(),
            jnp.inf,
        )
        return metrics

    def _get_empty_metrics(self) -> dict:
        """Return empty metrics dictionary with correct structure."""
        return {
            "completion_rate": jnp.array(0.0),
            "total_waiting_time": jnp.array(0.0),
            "avg_waiting_time": jnp.array(0.0),
            "max_waiting_time": jnp.array(0.0),
            "total_in_vehicle_time": jnp.array(0.0),
            "avg_in_vehicle_time": jnp.array(0.0),
            "max_in_vehicle_time": jnp.array(0.0),
            "total_travel_time": jnp.array(0.0),
            "avg_total_travel_time": jnp.array(0.0),
            "total_transfers": jnp.array(0),
            "avg_transfers_per_passenger": jnp.array(0.0),
            "avg_vehicle_utilization": jnp.array(0.0),
            "percent_empty_vehicles": jnp.array(0.0),
            "percent_full_vehicles": jnp.array(0.0),
            "ratio_travel_time_direct_to_shortest_path": jnp.array(0.0),
            "ratio_travel_time_transfers_to_shortest_path": jnp.array(0.0),
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
        num_nodes = self.num_nodes
        num_routes = self.num_routes
        max_route_length = self.max_route_length
        num_vehicles = self.num_vehicles

        return specs.Spec(
            Observation,
            "ObservationSpec",
            # Network data specs
            num_nodes=specs.BoundedArray(
                shape=(1,),
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
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=num_routes,
            ),
            max_route_length=specs.BoundedArray(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=max_route_length,
            ),
            route_types=specs.BoundedArray(
                shape=(num_routes,),
                dtype=int,
                minimum=0,
                maximum=1,
            ),
            route_stops=specs.BoundedArray(
                shape=(num_routes * max_route_length,),  # Flattened
                dtype=int,
                minimum=-1,
                maximum=num_nodes - 1,
            ),
            route_frequencies=specs.BoundedArray(
                shape=(num_routes,),
                dtype=int,
                minimum=0,
                maximum=num_vehicles,
            ),
            num_flex_routes=specs.BoundedArray(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=num_routes,
            ),
            num_fix_routes=specs.BoundedArray(
                shape=(1,),
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
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=num_vehicles,
            ),
            fleet_positions=specs.BoundedArray(
                shape=(num_vehicles,),  # Flattened
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
                shape=(1,),
                dtype=float,
                minimum=0.0,
                maximum=self.runtime,
            ),
            action_mask=specs.BoundedArray(
                shape=(num_routes * (num_nodes + 1),),  # Flattened
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
                shape=(self.num_routes,),
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
        travel_times, _ = calculate_shortest_route_times(state)
        direct_times = jnp.min(travel_times, axis=0)

        transfer_times = jax.lax.cond(
            state.routes.num_flex_routes == state.routes.num_routes,
            lambda: jnp.full((num_nodes, num_nodes), jnp.inf),
            lambda: jax.vmap(
                jax.vmap(
                    lambda o, d: find_best_transfer_route(state, o, d, travel_times)[0],
                    in_axes=(None, 0),
                ),
                in_axes=(0, None),
            )(jnp.arange(num_nodes), jnp.arange(num_nodes)),
        )

        return Observation(
            # Network data
            num_nodes=jnp.array([num_nodes]),  # Make 1D
            travel_times=state.network.travel_times.flatten(),
            is_terminal=state.network.is_terminal,
            # Routes data
            num_routes=jnp.array([state.routes.num_routes]),
            max_route_length=jnp.array([self.max_route_length]),
            route_types=state.routes.types,
            route_stops=state.routes.stops.flatten(),
            route_frequencies=state.routes.vehicles_per_route,
            num_flex_routes=jnp.array([state.routes.num_flex_routes]),
            num_fix_routes=jnp.array([state.routes.num_fix_routes]),
            direct_travel_times=direct_times.flatten(),
            transfer_travel_times=transfer_times.flatten(),
            network_shortest_times=self._network_shortest_times.flatten(),
            # Fleet data
            num_vehicles=jnp.array([state.fleet.num_vehicles]),
            fleet_positions=state.fleet.current_edges,
            # Aggregated passenger data
            future_demand=future_demand.flatten(),
            waiting_demand=waiting_demand.flatten(),
            transferring_demand=transferring_demand.flatten(),
            # Environment state
            current_time=jnp.array([state.current_time]),
            action_mask=self.get_action_mask(state).flatten(),
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

        if not self.allow_actions_fixed_routes:
            is_fixed_route = (state.routes.types == RouteType.FIXED)[:, None]
            fixed_route_mask = jnp.zeros_like(allowed_actions)

            # Always allow no-op for solution routes, regardless of steps
            solution_routes_mask = (
                2 <= jnp.sum(state.routes.stops != -1, axis=1)[:, None]
            )  # Check if route has any stops
            fixed_route_mask = fixed_route_mask.at[:, -1].set(
                True
            )  # Always allow no-op for fixed routes

            # Combine all masks
            action_mask = jnp.where(
                is_fixed_route & solution_routes_mask,  # Solution routes
                fixed_route_mask,  # Always allow no-op
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

    def _generate_random_vehicle_allocation(
        self,
        key: chex.PRNGKey,
        num_routes: int,
        total_vehicles: int,
        min_vehicles_per_route: int = 1,
    ) -> tuple[int, ...]:
        """Generate random vehicle allocations ensuring minimum vehicles per route."""
        if total_vehicles < num_routes * min_vehicles_per_route:
            raise ValueError(
                f"Not enough vehicles ({total_vehicles}) to ensure minimum "
                f"of {min_vehicles_per_route} vehicles for {num_routes} routes"
            )

        # First, allocate minimum vehicles to each route
        remaining_vehicles = total_vehicles - (num_routes * min_vehicles_per_route)

        # Generate random proportions for remaining vehicles
        props = jax.random.uniform(key, shape=(num_routes,))
        props = props / props.sum()

        # Calculate additional vehicles per route
        additional_vehicles = jnp.floor(props * remaining_vehicles).astype(int)

        # Add any remaining vehicles to the route with highest proportion
        leftover = remaining_vehicles - additional_vehicles.sum()
        additional_vehicles = additional_vehicles.at[jnp.argmax(props)].add(leftover)

        # Add minimum vehicles to get final allocation
        final_allocation = additional_vehicles + min_vehicles_per_route

        return tuple(final_allocation)
