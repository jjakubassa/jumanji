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
    NetworkData,
    Observation,
    PassengerStatus,
    RouteBatch,
    RouteType,
    State,
    assign_passengers,
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
        runtime: float = 100.0,
        vehicle_capacity: int = 20,
        num_flex_routes: int = 50,
    ) -> None:
        self.network_name: Final = network_name
        self.runtime: Final = runtime
        self.num_flex_routes: Final = num_flex_routes
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )

        # Load all static data once during initialization
        self._network_data = load_network_data(network_name)
        self._routes, self._vehicles_per_route = load_solution_data(network_name)
        self.max_route_length = max(len(route) for route in self._routes)

        # Create static components once
        self._route_batch = create_initial_routes(
            self._routes, num_flex_routes=self.num_flex_routes
        )
        self._initial_fleet = create_initial_fleet(
            self._routes,
            self._vehicles_per_route,
            self._network_data.travel_times,
        )

        # Load passenger demand data
        self._demand_data = load_demand_data(network_name)

        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state."""
        initial_state = State(
            network=self._network_data,
            fleet=self._initial_fleet,
            passengers=create_initial_passengers(self._demand_data, key, runtime=self.runtime),
            routes=self._route_batch,
            current_time=jnp.array(0.0),
            key=key,
        )

        timestep = restart(
            observation=self.get_observation(initial_state),
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

        # 2. Move vehicles to new position
        state = move_vehicles(state)

        # 3. Increase time in vehicle for passengers in vehicles and waiting
        new_passengers = increment_wait_times(state.passengers)
        new_passengers = increment_in_vehicle_times(new_passengers)
        state = replace(state, passengers=new_passengers)

        # 4. For passengers at their goal or transfer stop: remove passengers from vehicles and
        # update passengers status
        state = handle_completed_and_transferring_passengers(state)

        # 5. Switch status to WAITING based on current time
        new_passengers = update_passengers_to_waiting(
            state.passengers,
            state.current_time,
        )
        state = replace(state, passengers=new_passengers)

        # 6. Assign waiting passengers to vehicles at stations
        state = assign_passengers(state)

        # 7. Calculate reward based on state transition
        # Negative reward based on waiting and in-vehicle times
        reward = -jnp.sum(
            state.passengers.time_waiting + state.passengers.time_in_vehicle, dtype=jnp.float32
        )

        # 8. Check if episode is done
        done = self._is_done(state)

        # 9. Increase simulation time
        new_time = state.current_time + 1.0
        state = replace(state, current_time=new_time)

        # 10. Create timestep
        obs = self.get_observation(state)
        timestep = jax.lax.cond(
            done,
            lambda: termination(observation=obs, reward=reward),
            lambda: transition(observation=obs, reward=reward),
        )

        # 11. Return updated state and timestep
        return state, timestep

    def _is_done(self, state: State) -> Bool[Array, ""]:
        """Check if episode is done.

        Episode ends when either:
        1. We've reached runtime
        2. All passengers have completed their journeys
        """
        time_done = state.current_time >= self.runtime
        passengers_done = jnp.all(state.passengers.statuses == PassengerStatus.COMPLETED)
        return time_done | passengers_done

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec."""
        num_nodes = self._network_data.num_nodes
        max_passengers = len(self._demand_data)

        return specs.Spec(
            Observation,
            "ObservationSpec",
            network=specs.Spec(
                NetworkData,
                "NetworkSpec",
                node_coordinates=specs.BoundedArray(
                    shape=(num_nodes, 2),
                    dtype=float,
                    minimum=0.0,
                    maximum=1.0,
                ),
                travel_times=specs.BoundedArray(
                    shape=(num_nodes, num_nodes),
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
            ),
            routes=specs.Spec(
                RouteBatch,
                "RouteBatchSpec",
                types=specs.BoundedArray(
                    shape=(self._route_batch.num_routes,),
                    dtype=int,
                    minimum=0,
                    maximum=1,  # RouteType.FIXED or RouteType.FLEXIBLE
                ),
                stops=specs.BoundedArray(
                    shape=(self._route_batch.num_routes, self.max_route_length),
                    dtype=int,
                    minimum=-1,  # -1 for padding
                    maximum=num_nodes - 1,
                ),
                frequencies=specs.BoundedArray(
                    shape=(self._route_batch.num_routes,),
                    dtype=float,
                    minimum=0.0,
                    maximum=float("inf"),
                ),
                num_flex_routes=specs.BoundedArray(
                    shape=(),
                    dtype=int,
                    minimum=0,
                    maximum=self._route_batch.num_routes,
                ),
                num_fix_routes=specs.BoundedArray(
                    shape=(),
                    dtype=int,
                    minimum=0,
                    maximum=self._route_batch.num_routes,
                ),
            ),
            fleet_positions=specs.BoundedArray(
                shape=(self._initial_fleet.num_vehicles, 2),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
            ),
            origins=specs.BoundedArray(
                shape=(max_passengers,),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
            ),
            destinations=specs.BoundedArray(
                shape=(max_passengers,),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
            ),
            desired_departure_times=specs.BoundedArray(
                shape=(max_passengers,),
                dtype=float,
                minimum=0.0,
                maximum=self.runtime,
            ),
            passenger_statuses=specs.BoundedArray(
                shape=(max_passengers,),
                dtype=int,
                minimum=0,
                maximum=3,  # Number of PassengerStatus values
            ),
            current_time=specs.BoundedArray(
                shape=(),
                dtype=float,
                minimum=0.0,
                maximum=self.runtime,
            ),
            action_mask=specs.BoundedArray(
                shape=(self._route_batch.num_routes, num_nodes + 1),
                dtype=bool,
                minimum=False,
                maximum=True,
            ),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec for flexible routes.

        The action space consists of one action per flexible route.
        Each action is an integer indicating:
        - Which node to add as the next stop (0 to num_nodes-1)
        - Or perform no-op (num_nodes)
        """
        return specs.BoundedArray(
            shape=(self._route_batch.num_routes,),  # One action per flexible route
            dtype=jnp.int32,
            minimum=0,  # First node index
            maximum=self._network_data.num_nodes,  # num_nodes is no-op action
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
        """Creates observation from current state."""

        return Observation(
            network=state.network,
            routes=state.routes,
            fleet_positions=state.fleet.current_edges,
            origins=state.passengers.origins,
            destinations=state.passengers.destinations,
            desired_departure_times=state.passengers.desired_departure_times,
            passenger_statuses=state.passengers.statuses,
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

        # Always allow the no-op action
        no_op = jnp.ones((num_routes, 1), dtype=bool)

        # Combine connected nodes with no-op
        allowed_actions = jnp.concatenate([connected_nodes, no_op], axis=1)

        # For routes where last_stops == -1, allow all actions
        initial_routes = (last_stops == -1)[:, None]
        all_actions = jnp.ones_like(allowed_actions, dtype=bool)

        # For fixed routes, only allow no-op
        is_fixed_route = (state.routes.types == RouteType.FIXED)[:, None]
        fixed_route_mask = jnp.zeros_like(allowed_actions)
        fixed_route_mask = fixed_route_mask.at[:, -1].set(True)  # Only no-op allowed

        # Combine all masks
        action_mask = jnp.where(
            is_fixed_route,
            fixed_route_mask,  # Fixed routes: only no-op
            jnp.where(
                initial_routes,
                all_actions,  # Initial routes: all actions
                allowed_actions,  # Other cases: connected nodes + no-op
            ),
        )
        return action_mask
