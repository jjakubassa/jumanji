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
from typing import Final, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import matplotlib
import pandas as pd
import pkg_resources
from jax import random

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.types import (
    DirectPaths,
    NetworkData,
    Observation,
    PassengerBatch,
    RouteBatch,
    State,
    TransferPaths,
    VehicleBatch,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer

from .profiler import profile_function


class Mandl(Environment[State, specs.BoundedArray, Observation]):
    def __init__(
        self,
        viewer: Optional[Viewer] = None,
        max_capacity: int = 40,
        simulation_steps: int = 60,
        num_flex_routes: int = 2,
        waiting_penalty_factor: float = 2.0,
    ) -> None:
        self.max_capacity: Final = max_capacity
        self.simulation_steps: Final = simulation_steps
        self.num_flex_routes: Final = num_flex_routes
        self.waiting_penalty_factor: Final = jnp.array(waiting_penalty_factor)
        self.num_nodes: Final = 15
        self.max_num_routes: Final = 99
        self.max_demand: Final = 1024
        self.max_edge_weight: Final = 10
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )
        self.num_passengers: int
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"Mandl(max_capacity={self.max_capacity}, "
            f"simulation_steps={self.simulation_steps}, "
            f"num_flex_routes={self.num_flex_routes}, "
            f"waiting_penalty_factor={self.waiting_penalty_factor})"
        )

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state."""
        # Load network and demand data
        network_data, demand = self._load_instance_data()

        # Load fixed routes
        fixed_routes = self._load_fixed_routes()
        num_fixed_routes = len(fixed_routes.ids)

        # Initialize flexible routes (empty initially)
        flexible_routes = RouteBatch(
            ids=jnp.arange(num_fixed_routes, num_fixed_routes + self.num_flex_routes),
            nodes=jnp.full((self.num_flex_routes, self.num_nodes), -1, dtype=int),
            frequencies=jnp.ones(self.num_flex_routes, dtype=int),
            on_demand=jnp.ones(self.num_flex_routes, dtype=bool),
        )

        # Combine fixed and flexible routes
        routes = RouteBatch(
            ids=jnp.concatenate([fixed_routes.ids, flexible_routes.ids]),
            nodes=jnp.concatenate([fixed_routes.nodes, flexible_routes.nodes]),
            frequencies=jnp.concatenate([fixed_routes.frequencies, flexible_routes.frequencies]),
            on_demand=jnp.concatenate([fixed_routes.on_demand, flexible_routes.on_demand]),
        )

        # Initialize vehicles
        total_routes = len(routes.ids)
        vehicles = self._init_vehicles(num_vehicles=total_routes, max_route_length=self.num_nodes)

        # Initialize passengers
        key, passengers_key = random.split(key)
        passengers = self._init_passengers(passengers_key, demand, self.simulation_steps)
        self.num_passengers = len(passengers.ids)

        # Create initial state
        state = State(
            network=network_data,
            vehicles=vehicles,
            routes=routes,
            passengers=passengers,
            current_time=0,
            key=key,
        )

        # Create initial timestep
        timestep = restart(observation=self._state_to_observation(state))

        return state, timestep

    def _load_instance_data(
        self,
    ) -> tuple[NetworkData, jnp.ndarray]:
        """Load network and demand data from files.

        Returns:
            NetworkData: Contains the network structure with nodes, links, and terminals
            demand: Matrix of shape (num_nodes, num_nodes) containing demand between nodes
        """
        # Load data from files
        nodes_path = pkg_resources.resource_filename(
            "jumanji", "environments/routing/mandl/mandl1_nodes.txt"
        )
        links_path = pkg_resources.resource_filename(
            "jumanji", "environments/routing/mandl/mandl1_links.txt"
        )
        demand_path = pkg_resources.resource_filename(
            "jumanji", "environments/routing/mandl/mandl1_demand.txt"
        )
        nodes_df = pd.read_csv(nodes_path)
        links_df = pd.read_csv(links_path)
        demand_df = pd.read_csv(demand_path)

        # Process nodes
        nodes = jnp.array(nodes_df[["lat", "lon"]].values)
        terminals = jnp.array(nodes_df["terminal"].values, dtype=bool)

        # Normalize node coordinates to [0, 1]
        min_coords = nodes.min(axis=0)
        max_coords = nodes.max(axis=0)
        nodes = (nodes - min_coords) / (max_coords - min_coords)

        # Create adjacency matrix with travel times
        n_nodes = len(nodes_df)
        links = jnp.full((n_nodes, n_nodes), jnp.inf)  # Initialize with inf
        links = links.at[jnp.diag_indices_from(links)].set(0)  # Set diagonal to 0

        # Fill in the travel times from the links data
        for _, row in links_df.iterrows():
            from_node = int(row["from"] - 1)  # Convert from 1-based to 0-based indexing
            to_node = int(row["to"] - 1)
            links = links.at[from_node, to_node].set(row["travel_time"])

        # Process demand into matrix form
        demand = jnp.zeros((n_nodes, n_nodes))
        for _, row in demand_df.iterrows():
            demand = demand.at[int(row["from"] - 1), int(row["to"] - 1)].set(row["demand"])

        network_data = NetworkData(
            nodes=nodes,  # Normalized coordinates of each node
            links=links,  # Travel times between nodes (inf if no direct connection)
            terminals=terminals,  # Boolean mask of terminal nodes
        )

        return network_data, demand

    def _load_fixed_routes(
        self,
    ) -> RouteBatch:
        """Load fixed routes from solution file.

        File format:
        First line: Comment about max stops
        Second line: Number of routes
        Next N lines: Routes as node sequences separated by dashes
        Last N lines: Frequencies for each route

        Returns:
            RouteBatch containing the fixed routes
        """
        path = pkg_resources.resource_filename(
            "jumanji", "environments/routing/mandl/mandl1_solution.txt"
        )

        with open(path, "r") as f:
            lines = f.readlines()

            # Skip comment line and read number of routes
            num_routes = int(lines[1])

            # Read routes
            routes = []
            for line in lines[2 : 2 + num_routes]:
                # Convert route string to list of node indices (subtract 1 for 0-based indexing)
                route = [int(x) - 1 for x in line.strip().split("-")]  # Note: using en dash
                # Pad with -1 to max length
                padded_route = route + [-1] * (self.num_nodes - len(route))
                routes.append(padded_route)

            # Read frequencies
            frequencies = []
            for line in lines[2 + num_routes : 2 + 2 * num_routes]:
                frequencies.append(int(line.strip()))

        return RouteBatch(
            ids=jnp.arange(num_routes),
            nodes=jnp.array(routes),
            frequencies=jnp.array(frequencies),
            on_demand=jnp.zeros(num_routes, dtype=bool),  # Fixed routes are not on-demand
        )

    def _init_passengers(
        self,
        key: chex.PRNGKey,
        demand_matrix: jnp.ndarray,
        simulation_period: int = 60,  # minutes
    ) -> PassengerBatch:
        """Generate passengers according to demand matrix."""
        n_nodes = demand_matrix.shape[0]

        # Use max_demand as our fixed size for all passenger arrays
        n_passengers = self.max_demand

        # Create a fixed set of passengers spread across all OD pairs
        od_pairs_per_passenger = jnp.mgrid[:n_nodes, :n_nodes].reshape(
            2, -1
        )  # Shape: (2, n_nodes*n_nodes)
        num_od_pairs = n_nodes * n_nodes

        # Repeat the OD pairs to fill max_demand slots
        repeat_factor = (n_passengers + num_od_pairs - 1) // num_od_pairs
        origins = jnp.tile(od_pairs_per_passenger[0], repeat_factor)[:n_passengers]
        destinations = jnp.tile(od_pairs_per_passenger[1], repeat_factor)[:n_passengers]

        # Create validity mask
        valid_mask = origins != destinations

        # Generate departure times
        key, departure_key = random.split(key)
        departure_times = random.uniform(
            departure_key, shape=(n_passengers,), minval=0, maxval=simulation_period
        )

        # Apply mask to all arrays
        origins = jnp.where(valid_mask, origins, 0)
        destinations = jnp.where(valid_mask, destinations, 0)
        departure_times = jnp.where(valid_mask, departure_times, 0.0)

        return PassengerBatch(
            ids=jnp.arange(n_passengers),
            origins=origins,
            destinations=destinations,
            departure_times=departure_times,
            time_waiting=jnp.zeros(n_passengers),
            time_in_vehicle=jnp.zeros(n_passengers),
            statuses=jnp.where(valid_mask, 0, -1).astype(jnp.int32),  # -1 for invalid passengers
        )

    def _init_vehicles(self, num_vehicles: int, max_route_length: int) -> VehicleBatch:
        """Initialize vehicles at the start of the simulation.

        Args:
            num_vehicles: Total number of vehicles to initialize
            max_route_length: Maximum length of a route (usually number of nodes)

        Returns:
            VehicleBatch: Initial state of all vehicles
        """
        return VehicleBatch(
            ids=jnp.arange(num_vehicles),
            route_ids=jnp.arange(num_vehicles),  # Each vehicle starts with unique route ID
            current_edges=jnp.zeros(
                (num_vehicles, 2), dtype=int
            ),  # All vehicles start at edge (0,0)
            times_on_edge=jnp.zeros(num_vehicles),  # No time spent on edges initially
            passengers=jnp.full(
                (num_vehicles, self.max_capacity), -1, dtype=int
            ),  # No passengers initially (-1 represents empty seat)
            capacities=jnp.full(
                num_vehicles, self.max_capacity
            ),  # All vehicles start with full capacity
            directions=jnp.ones(num_vehicles, dtype=int),  # All vehicles start moving forward
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Convert state to observation for the agent.

        Args:
            state: Current state of the environment
            t: Number of timesteps to look ahead for future demand

        Returns:
            Observation object containing relevant information for the agent
        """
        # Extract required fields from state and transform as needed
        network_mask = state.network.links < jnp.inf  # Convert links to boolean adjacency matrix

        return Observation(
            network=network_mask,
            travel_times=state.network.links,
            routes=state.routes.nodes,
            origins=state.passengers.origins,
            destinations=state.passengers.destinations,
            departure_times=state.passengers.departure_times,
            time_waiting=state.passengers.time_waiting,
            time_in_vehicle=state.passengers.time_in_vehicle,
            statuses=state.passengers.statuses,
            route_ids=state.vehicles.route_ids,
            current_edges=state.vehicles.current_edges,
            times_on_edge=state.vehicles.times_on_edge,
            capacities=state.vehicles.capacities,
            directions=state.vehicles.directions,
            frequencies=state.routes.frequencies,
            on_demand=state.routes.on_demand,
            current_time=state.current_time,
        )

    @profile_function
    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep[Observation]]:
        """Execute one environment step."""
        # Validate action shape
        if action.shape != (self.num_flex_routes,):
            raise ValueError(
                f"Action must have shape ({self.num_flex_routes},), got {action.shape}"
            )

        # First update the routes
        new_routes = self._update_flexible_routes(state.routes, action, state.network.links)

        # Create new state with updated routes
        state = replace(state, routes=new_routes)

        # Update vehicles
        new_vehicles = Mandl._move_vehicles(state.vehicles, state.routes.nodes, state.network.links)

        # Create new state with updated vehicles
        state = replace(state, vehicles=new_vehicles)

        # Update passenger status
        new_passengers, new_vehicles = Mandl._assign_passengers(
            state.passengers, state.vehicles, state.routes, state.network, state.current_time
        )

        # Create final state update
        new_state = replace(
            state,
            vehicles=new_vehicles,
            passengers=new_passengers,
            current_time=state.current_time + 1,
        )

        # Calculate done condition
        done = new_state.current_time >= self.simulation_steps

        def create_done_timestep(args):
            new_state, waiting_penalty_factor = args
            completed_time = jnp.sum(
                jnp.where(
                    new_state.passengers.statuses == 3,
                    new_state.passengers.time_waiting + new_state.passengers.time_in_vehicle,
                    0.0,
                )
            )

            waiting_penalty = (
                jnp.sum(
                    jnp.where(
                        new_state.passengers.statuses == 1, new_state.passengers.time_waiting, 0.0
                    )
                )
                * waiting_penalty_factor
            )

            reward = -(completed_time + waiting_penalty)

            return termination(
                observation=self._state_to_observation(new_state),
                reward=reward,
            )

        def create_transition_timestep(args):
            new_state, _ = args
            return transition(
                observation=self._state_to_observation(new_state),
                reward=jnp.array(0.0),
                discount=jnp.array(1.0),
            )

        # Pass both new_state and waiting_penalty_factor to the conditional functions
        args = (new_state, self.waiting_penalty_factor)
        timestep = jax.lax.cond(done, create_done_timestep, create_transition_timestep, args)

        return new_state, timestep

    def get_metrics(self, state: State) -> dict[str, float]:
        metrics = {
            "completed_passengers": jnp.sum(state.passengers.statuses == 3).item(),
            "waiting_passengers": jnp.sum(state.passengers.statuses == 1).item(),
            "in_vehicle_passengers": jnp.sum(state.passengers.statuses == 2).item(),
            "total_waiting_time": jnp.sum(state.passengers.time_waiting).item(),
            "total_in_vehicle_time": jnp.sum(state.passengers.time_in_vehicle).item(),
        }

        # Add route-related metrics
        for i in range(self.num_flex_routes):
            route = state.routes.nodes[len(state.routes.ids) - self.num_flex_routes + i]
            metrics[f"route_{i}_length"] = jnp.sum(route != -1).item()
            metrics[f"route_{i}_wait_actions"] = jnp.sum(route == -1).item()

        return metrics

    def _update_flexible_routes(
        self,
        routes: RouteBatch,
        action: jnp.ndarray,
        links: jnp.ndarray,
    ) -> RouteBatch:
        """Update flexible routes based on actions."""
        new_nodes = routes.nodes
        num_fixed_routes = len(routes.ids) - self.num_flex_routes

        def update_route(i, nodes):
            all_idx = num_fixed_routes + i
            route = nodes[all_idx]

            padding_mask = route == -1
            cumsum_mask = jnp.cumsum(padding_mask)
            empty_pos = jnp.argmax(cumsum_mask == 1)

            new_node = action[i]

            return jax.lax.cond(
                new_node >= 0,
                lambda: nodes.at[all_idx, empty_pos].set(new_node),
                lambda: nodes,
            )

        new_nodes = jax.lax.fori_loop(0, self.num_flex_routes, update_route, new_nodes)

        return routes._replace(nodes=new_nodes)

    def _is_valid_next_stop(self, route: jnp.ndarray, new_node: int, links: jnp.ndarray) -> bool:
        """Check if new_node is a valid next stop for the route."""
        # Wait action is always valid
        if new_node == -1:
            return True

        # Get last valid node in route
        valid_nodes = route[route != -1]
        if len(valid_nodes) == 0:
            return True  # Any node is valid for empty route

        last_node = valid_nodes[-1]
        # Check if nodes are connected in network
        return bool((links[last_node, new_node] < jnp.inf).item())

    @staticmethod
    @profile_function
    def _move_vehicles(
        vehicles: VehicleBatch, routes: jnp.ndarray, network_links: jnp.ndarray
    ) -> VehicleBatch:
        """Update vehicle positions based on their routes and current edges."""

        # Get all relevant arrays at once
        times_on_edge = vehicles.times_on_edge
        current_edges = vehicles.current_edges
        directions = vehicles.directions
        route_ids = vehicles.route_ids

        # Get travel times for all current edges
        from_nodes = current_edges[:, 0]
        to_nodes = current_edges[:, 1]
        edge_times = network_links[from_nodes, to_nodes]

        # Create masks for vehicles that need to move
        need_to_move = times_on_edge >= edge_times

        # Get current routes for all vehicles
        valid_routes = routes[route_ids]

        # Find current positions in routes
        valid_mask = valid_routes != -1
        route_lengths = jnp.sum(valid_mask, axis=1)
        current_positions = jnp.where(
            (valid_routes == current_edges[:, 1:2]) & valid_mask,
            jnp.arange(valid_routes.shape[1])[None, :],
            -1,
        ).max(axis=1)

        # Calculate next positions
        next_positions = current_positions + directions

        # Handle end conditions
        at_end = next_positions >= route_lengths
        at_start = next_positions < 0

        # Update directions
        new_directions = jnp.where(
            at_end,
            -1,  # reverse direction at end
            jnp.where(
                at_start,
                1,  # forward direction at start
                directions,  # keep current direction
            ),
        )

        # Adjust next positions based on direction changes
        next_positions = jnp.where(
            at_end,
            route_lengths - 2,  # second-to-last position
            jnp.where(
                at_start,
                1,  # second position
                next_positions,
            ),
        )

        # Get next nodes for vehicles that need to move
        valid_nodes = jnp.where(valid_mask, valid_routes, -1)
        next_nodes = jnp.take_along_axis(valid_nodes, next_positions[:, None], axis=1).squeeze()

        # Update vehicles that need to move
        new_current_edges = jnp.where(
            need_to_move[:, None],
            jnp.stack([current_edges[:, 1], next_nodes], axis=1),
            current_edges,
        )

        new_times_on_edge = jnp.where(need_to_move, 0, times_on_edge + 1)

        # Create new vehicle batch with updated values
        return vehicles._replace(
            current_edges=new_current_edges,
            times_on_edge=new_times_on_edge,
            directions=new_directions,
        )

    def _update_vehicle_position(
        self,
        vehicles: VehicleBatch,
        idx: int,
        current_edge: jnp.ndarray,
        routes: RouteBatch,
        direction: int,
    ) -> VehicleBatch:
        """Helper function to update a vehicle's position."""
        route = routes.nodes[vehicles.route_ids[idx]]

        # Create mask for valid nodes
        valid_mask = route != -1
        route_len = jnp.sum(valid_mask)
        valid_indices = jnp.arange(len(route))

        # Get position of current node in valid route
        current_pos = jnp.where((route == current_edge[1]) & valid_mask, valid_indices, -1).max()

        def process_valid_route(_):
            next_pos = current_pos + direction

            def handle_end(_):
                return vehicles.directions.at[idx].set(-1), route_len - 2

            def handle_start(_):
                return vehicles.directions.at[idx].set(1), 1

            def handle_middle(_):
                return vehicles.directions, next_pos

            directions, next_pos = jax.lax.cond(
                next_pos >= route_len,
                handle_end,
                lambda _: jax.lax.cond(next_pos < 0, handle_start, handle_middle, None),
                None,
            )

            valid_nodes = jnp.where(valid_mask, route, -1)
            next_node = valid_nodes[next_pos]

            return vehicles._replace(
                directions=directions,
                current_edges=vehicles.current_edges.at[idx].set(
                    jnp.array([current_edge[1], next_node], dtype=int)
                ),
                times_on_edge=vehicles.times_on_edge.at[idx].set(0),
            )

        return jax.lax.cond(route_len > 0, process_valid_route, lambda _: vehicles, None)

    @staticmethod
    def _update_passenger_status(state: State) -> State:
        """Update passenger statuses based on current time and position."""
        new_passengers = state.passengers
        new_vehicles = state.vehicles
        current_time = state.current_time

        # Update statuses based on current time and conditions
        new_statuses = new_passengers.statuses

        # 1. Update not-in-system to waiting (0 -> 1) when departure time is reached
        new_statuses = jnp.where(
            (new_statuses == 0) & (new_passengers.departure_times <= current_time), 1, new_statuses
        )

        # 2. Update in-vehicle to completed (2 -> 3) when destination is reached
        new_vehicle_passengers = new_vehicles.passengers

        def update_vehicle_passengers(
            vehicle_idx: int, carry: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            statuses, vehicle_passengers = carry
            current_edge = new_vehicles.current_edges[vehicle_idx]
            current_node = current_edge[1]

            def update_passenger(
                seat_idx: int, carry_inner: tuple[jnp.ndarray, jnp.ndarray]
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                statuses_inner, vehicle_passengers_inner = carry_inner
                passenger_idx = vehicle_passengers_inner[vehicle_idx, seat_idx]

                def complete_journey(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    new_statuses = statuses_inner.at[passenger_idx].set(3)
                    new_vehicle_passengers = vehicle_passengers_inner.at[vehicle_idx, seat_idx].set(
                        -1
                    )
                    return new_statuses, new_vehicle_passengers

                def keep_status(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    return statuses_inner, vehicle_passengers_inner

                return jax.lax.cond(
                    (passenger_idx != -1)
                    & (current_node == new_passengers.destinations[passenger_idx]),
                    complete_journey,
                    keep_status,
                    None,
                )

            return jax.lax.fori_loop(
                0,
                new_vehicles.passengers.shape[1],
                update_passenger,
                (statuses, vehicle_passengers),
            )

        # Process all vehicles
        new_statuses, new_vehicle_passengers = jax.lax.fori_loop(
            0,
            len(new_vehicles.ids),  # Use len instead of shape[1]
            update_vehicle_passengers,
            (new_statuses, new_vehicle_passengers),
        )

        # Update passengers with new statuses and times
        new_passengers = new_passengers._replace(
            statuses=new_statuses,
            time_waiting=jnp.where(
                new_statuses == 1,  # Only update waiting time for waiting passengers
                new_passengers.time_waiting + 1,
                new_passengers.time_waiting,
            ),
            time_in_vehicle=jnp.where(
                new_statuses == 2,  # Only update in-vehicle time for riding passengers
                new_passengers.time_in_vehicle + 1,
                new_passengers.time_in_vehicle,
            ),
        )

        # Update vehicles with new passenger assignments
        new_vehicles = new_vehicles._replace(passengers=new_vehicle_passengers)

        return replace(state, passengers=new_passengers, vehicles=new_vehicles)

    @staticmethod
    @profile_function
    def _assign_passengers(
        passengers: PassengerBatch,
        vehicles: VehicleBatch,
        routes: RouteBatch,
        network: NetworkData,
        current_time: int,
    ) -> tuple[PassengerBatch, VehicleBatch]:
        """Assign waiting passengers to vehicles based on routes and capacity."""
        # Pre-compute masks for valid assignments
        waiting_mask = passengers.statuses == 1
        capacity_mask = vehicles.passengers == -1

        # Get vehicle positions
        vehicle_positions = vehicles.current_edges[:, 1]
        available_seats = jnp.sum(capacity_mask, axis=1)

        # Compute costs matrix
        def compute_costs(p_idx):
            origin = passengers.origins[p_idx]
            dest = passengers.destinations[p_idx]
            is_waiting = waiting_mask[p_idx]

            def compute_vehicle_cost(v_idx):
                vehicle_pos = vehicle_positions[v_idx]
                has_capacity = available_seats[v_idx] > 0

                pickup_paths, _ = Mandl._find_paths(
                    network=network,
                    routes=routes,
                    start=vehicle_pos,
                    end=origin,
                )
                delivery_paths, _ = Mandl._find_paths(
                    network=network,
                    routes=routes,
                    start=origin,
                    end=dest,
                )

                pickup_cost = jnp.min(pickup_paths[:, 2])
                delivery_cost = jnp.min(delivery_paths[:, 2])
                total_cost = pickup_cost + delivery_cost

                return jnp.where(is_waiting & has_capacity, total_cost, jnp.inf)

            return jax.vmap(compute_vehicle_cost)(jnp.arange(len(vehicles.ids)))

        costs = jax.vmap(compute_costs)(jnp.arange(len(passengers.ids)))

        # Find best assignments
        best_vehicles = jnp.argmin(costs, axis=1)
        min_costs = jnp.min(costs, axis=1)

        # Process assignments using scan
        def scan_fn(carry, x):
            i, vehicle_idx, cost = x
            p_state, v_state = carry

            def assign():
                # Find first empty seat
                empty_seat = jnp.argmax(v_state.passengers[vehicle_idx] == -1)

                # Update passenger status
                new_p_statuses = p_state.statuses.at[i].set(2)
                new_p_state = p_state._replace(statuses=new_p_statuses)

                # Update vehicle passengers
                new_v_passengers = v_state.passengers.at[vehicle_idx, empty_seat].set(i)
                new_v_state = v_state._replace(passengers=new_v_passengers)

                return new_p_state, new_v_state

            return jax.lax.cond(
                cost < jnp.inf, lambda _: assign(), lambda _: (p_state, v_state), None
            ), None

        # Create scan input
        indices = jnp.arange(len(passengers.ids))
        scan_input = (indices, best_vehicles, min_costs)

        # Run scan
        (final_passengers, final_vehicles), _ = jax.lax.scan(
            scan_fn, (passengers, vehicles), scan_input
        )

        return final_passengers, final_vehicles

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            A nested `specs.Spec` matching the structure of `Observation`.
        """
        num_nodes = self.num_nodes
        max_num_routes = self.max_num_routes
        max_capacity = self.max_capacity
        max_demand = self.max_demand

        return specs.Spec(
            Observation,
            "ObservationSpec",
            network=specs.Array(
                shape=(num_nodes, num_nodes),
                dtype=bool,
                name="network",
            ),
            travel_times=specs.BoundedArray(
                shape=(num_nodes, num_nodes),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
                name="travel_times",
            ),
            routes=specs.BoundedArray(
                shape=(max_num_routes, num_nodes),
                dtype=int,
                minimum=-1,
                maximum=num_nodes - 1,
                name="routes",
            ),
            origins=specs.BoundedArray(
                shape=(max_demand,),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
                name="origins",
            ),
            destinations=specs.BoundedArray(
                shape=(max_demand,),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
                name="destinations",
            ),
            departure_times=specs.BoundedArray(
                shape=(max_demand,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
                name="departure_times",
            ),
            time_waiting=specs.BoundedArray(
                shape=(max_demand,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
                name="time_waiting",
            ),
            time_in_vehicle=specs.BoundedArray(
                shape=(max_demand,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
                name="time_in_vehicle",
            ),
            statuses=specs.BoundedArray(
                shape=(max_demand,),
                dtype=int,
                minimum=0,
                maximum=3,  # 0: not in system, 1: waiting, 2: in vehicle, 3: completed
                name="statuses",
            ),
            route_ids=specs.BoundedArray(
                shape=(max_num_routes,),
                dtype=int,
                minimum=0,
                maximum=max_num_routes - 1,
                name="route_ids",
            ),
            current_edges=specs.BoundedArray(
                shape=(max_num_routes, 2),
                dtype=int,
                minimum=0,
                maximum=num_nodes - 1,
                name="current_edges",
            ),
            times_on_edge=specs.BoundedArray(
                shape=(max_num_routes,),
                dtype=float,
                minimum=0.0,
                maximum=float("inf"),
                name="times_on_edge",
            ),
            capacities=specs.BoundedArray(
                shape=(max_num_routes,),
                dtype=int,
                minimum=0,
                maximum=max_capacity,
                name="capacities",
            ),
            directions=specs.BoundedArray(
                shape=(max_num_routes,),
                dtype=int,
                minimum=-1,
                maximum=1,
                name="directions",
            ),
            frequencies=specs.BoundedArray(
                shape=(max_num_routes,),
                dtype=int,
                minimum=0,
                maximum=max_num_routes,
                name="frequencies",
            ),
            on_demand=specs.Array(
                shape=(max_num_routes,),
                dtype=bool,
                name="on_demand",
            ),
            current_time=specs.BoundedArray(
                shape=(),  # scalar
                dtype=int,
                minimum=0,
                maximum=self.simulation_steps,
                name="current_time",
            ),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Action space for flexible routes:

        For each route, the action space is:
        - -1: wait/don't append stop
        - [0, num_nodes-1]: choose corresponding node as next stop
        """
        return specs.BoundedArray(
            shape=(self.num_flex_routes,),  # One action per flexible route
            dtype=jnp.int32,
            minimum=-1,  # -1 represents wait action
            maximum=self.num_nodes - 1,  # Node indices from 0 to num_nodes-1
            name="action",
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

    def _get_action_mask(
        self, routes: jnp.ndarray, action_space: jnp.ndarray, network: jnp.ndarray
    ) -> jnp.ndarray:
        """Return a mask that indicates which nodes should not be considered for next steps.

        Returns:
            Boolean mask of shape (num_routes, num_nodes+1) with True indicating valid actions
            Note: For each route, the wait action (-1) is always valid
        """

        def _body_fun(i: int, val: jnp.ndarray) -> jnp.ndarray:
            # Get mask for node selections
            node_mask = jnp.where(
                jnp.any(routes[i] != -1),
                self._get_mask_per_route(network, routes[i]),
                jnp.zeros_like(routes[i], jnp.bool),
            )
            # Add wait action (always valid)
            full_mask = jnp.concatenate([jnp.array([True]), node_mask])
            return val.at[i].set(full_mask)

        return jax.lax.fori_loop(
            0, routes.shape[0], _body_fun, jnp.zeros_like(action_space, jnp.bool)
        )

    def _get_mask_per_route(self, network: jnp.ndarray, route: jnp.ndarray) -> jnp.ndarray:
        """Get mask for valid next nodes for a given route.

        Args:
            network: Adjacency matrix representing network connections
            route: Current route nodes

        Returns:
            Boolean mask where True indicates valid next nodes
        """
        valid_nodes = route[route != -1]
        if len(valid_nodes) == 0:
            return jnp.zeros_like(network[0], dtype=bool)
        current_node = valid_nodes[-1]
        return network[current_node] > 0

    @staticmethod
    def _has_no_cycle(route: jnp.ndarray) -> bool:
        """Check if a route contains no cycles.

        Args:
            route: Array of node indices representing a route

        Returns:
            Boolean indicating whether the route is cycle-free
        """
        valid_nodes = route[route != -1]
        seen = set()
        for node in valid_nodes:
            if node in seen:
                return False
            seen.add(node)
        return True

    @staticmethod
    @profile_function
    def _find_paths(
        network: NetworkData,
        routes: RouteBatch,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> tuple[DirectPaths, TransferPaths]:
        """Find direct and transfer paths between start and end nodes."""
        # Get all paths (valid ones will be sorted first due to costs)
        direct_paths = Mandl._find_direct_paths(
            network=network, routes=routes, start=start, end=end
        )

        # Check if we have any valid direct paths (finite cost)
        has_valid_direct = jnp.any(jnp.isfinite(direct_paths[:, 2]))

        def find_transfers(_: None) -> TransferPaths:
            return Mandl._find_transfer_paths(
                network=network,
                routes=routes,
                start=start,
                end=end,
                transfer_penalty=transfer_penalty,
            )

        def no_transfers(_: None) -> TransferPaths:
            # Return empty transfer paths array
            num_routes = len(routes.ids)
            num_nodes = network.links.shape[0]
            max_paths = num_routes * num_routes * num_nodes
            return jnp.stack(
                [
                    jnp.zeros(max_paths),  # first route
                    jnp.zeros(max_paths),  # second route
                    jnp.zeros(max_paths),  # transfer stop
                    jnp.zeros(max_paths),  # validity
                    jnp.full(max_paths, jnp.inf),  # cost
                ],
                axis=1,
            )

        transfer_paths = jax.lax.cond(~has_valid_direct, find_transfers, no_transfers, None)

        return direct_paths, transfer_paths

    @staticmethod
    def _get_route_shortest_paths(network: NetworkData, route: jnp.ndarray) -> jnp.ndarray:
        """Calculate shortest paths between all pairs of nodes for a given route."""
        if route.ndim not in (1, 2):
            raise ValueError(
                f"_get_route_shortest_paths expects a route vector or a single route in batch form, "
                f"got {route.ndim} dimensions"
            )
        if route.ndim == 2 and route.shape[0] != 1:
            raise ValueError(
                f"_get_route_shortest_paths can only process one route at a time, "
                f"got {route.shape[0]} routes"
            )

        # If it's a batch with one route, squeeze out the batch dimension
        route = route.squeeze(0) if route.ndim == 2 else route

        n_nodes = network.links.shape[0]
        dist = jnp.full((n_nodes, n_nodes), jnp.inf)

        # Set diagonal elements to 0
        dist = dist.at[jnp.diag_indices_from(dist)].set(0)

        # Create mask for valid nodes (not -1)
        valid_mask = route != -1
        route_len = jnp.sum(valid_mask)

        # Create masked version of network links
        masked_links = jnp.full((n_nodes, n_nodes), jnp.inf)
        masked_links = masked_links.at[jnp.diag_indices_from(masked_links)].set(0)

        def process_edge(i: int, links: jnp.ndarray) -> jnp.ndarray:
            # Only process if current and next positions are valid
            is_valid_current = valid_mask[i]  # This should be a scalar
            is_valid_next = jnp.where(i + 1 < route_len, valid_mask[i + 1], False)

            # Combine conditions into a single scalar boolean
            is_valid = jnp.logical_and(is_valid_current, is_valid_next)

            def add_edge(links: jnp.ndarray) -> jnp.ndarray:
                from_node = route[i]
                to_node = route[i + 1]
                # Add edge in both directions
                new_links = links.at[from_node, to_node].set(network.links[from_node, to_node])
                new_links = new_links.at[to_node, from_node].set(network.links[to_node, from_node])
                return new_links

            def no_change(links: jnp.ndarray) -> jnp.ndarray:
                return links

            return jax.lax.cond(
                is_valid,  # Now this should be a scalar
                add_edge,
                no_change,
                links,
            )

        # Add edges between consecutive nodes in route
        masked_links = jax.lax.fori_loop(0, len(route) - 1, process_edge, masked_links)

        # Floyd-Warshall algorithm remains the same
        def update_dist(k: int, d: jnp.ndarray) -> jnp.ndarray:
            def update_ij(ij: int, d_inner: jnp.ndarray) -> jnp.ndarray:
                i, j = ij // n_nodes, ij % n_nodes
                new_dist = d_inner[i, k] + d_inner[k, j]

                def update(d_in: jnp.ndarray) -> jnp.ndarray:
                    return d_in.at[i, j].set(new_dist)

                def no_update(d_in: jnp.ndarray) -> jnp.ndarray:
                    return d_in

                return jax.lax.cond(
                    (d_inner[i, k] < jnp.inf)
                    & (d_inner[k, j] < jnp.inf)
                    & (new_dist < d_inner[i, j]),
                    update,
                    no_update,
                    d_inner,
                )

            return jax.lax.fori_loop(0, n_nodes * n_nodes, lambda ij, d: update_ij(ij, d), d)

        dist = jax.lax.fori_loop(0, n_nodes, update_dist, masked_links)

        return dist

    @staticmethod
    def _get_all_shortest_paths(network: NetworkData, routes: RouteBatch) -> jnp.ndarray:
        """Calculate shortest paths between all pairs of nodes for all routes."""
        num_routes = len(routes.ids)  # Use len instead of shape[1]
        route_paths = jnp.zeros((num_routes, network.links.shape[0], network.links.shape[0]))

        def compute_route_paths(i: int, paths: jnp.ndarray) -> jnp.ndarray:
            return paths.at[i].set(
                Mandl._get_route_shortest_paths(network, routes.nodes[i])
            )  # Remove [0]

        return jax.lax.fori_loop(0, num_routes, compute_route_paths, route_paths)

    @staticmethod
    def _find_direct_paths(
        network: NetworkData,
        routes: RouteBatch,
        start: int,
        end: int,
    ) -> DirectPaths:
        # Get all paths (valid ones will be sorted first due to costs)
        shortest_paths = Mandl._get_all_shortest_paths(network=network, routes=routes)
        num_routes = len(routes.ids)

        # Initialize results array with invalid paths
        results: DirectPaths = jnp.stack(
            [
                jnp.arange(num_routes),  # route indices
                jnp.zeros(num_routes),  # validity (0 = invalid)
                jnp.full(num_routes, jnp.inf),  # costs
            ],
            axis=1,
        )

        def process_route(i: int, res: jnp.ndarray) -> jnp.ndarray:
            path_costs = shortest_paths[i, start, end]

            def valid_path(r: jnp.ndarray) -> jnp.ndarray:
                return r.at[i].set(jnp.array([i, 1.0, path_costs]))

            def invalid_path(r: jnp.ndarray) -> jnp.ndarray:
                return r.at[i].set(jnp.array([i, 0.0, jnp.inf]))

            return jax.lax.cond(path_costs < jnp.inf, valid_path, invalid_path, res)

        results = jax.lax.fori_loop(0, num_routes, process_route, results)

        # Sort by cost - invalid paths (with inf cost) will be at the end
        return results[jnp.argsort(results[:, 2])]

    @staticmethod
    def _find_transfer_paths(
        network: NetworkData,
        routes: RouteBatch,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> TransferPaths:
        """Find transfer paths between two nodes."""
        shortest_paths_costs = Mandl._get_all_shortest_paths(network=network, routes=routes)
        num_routes = len(routes.ids)
        num_nodes = network.links.shape[0]

        # Create meshgrid of all possible combinations
        first_routes = jnp.arange(num_routes)
        second_routes = jnp.arange(num_routes)
        transfer_points = jnp.arange(num_nodes)

        r1, r2, tp = jnp.meshgrid(first_routes, second_routes, transfer_points, indexing="ij")

        # Flatten all arrays
        r1 = r1.flatten()
        r2 = r2.flatten()
        tp = tp.flatten()

        # Vectorized computation of path costs
        def compute_path_cost(first_route, second_route, transfer_point):
            first_leg = shortest_paths_costs[first_route, start, transfer_point]
            second_leg = shortest_paths_costs[second_route, transfer_point, end]

            # Check validity conditions
            route_different = first_route != second_route
            legs_valid = jnp.logical_and(first_leg < jnp.inf, second_leg < jnp.inf)
            path_valid = jnp.logical_and(route_different, legs_valid)

            # Calculate total cost
            total_cost = jnp.where(path_valid, first_leg + second_leg + transfer_penalty, jnp.inf)

            # Return invalid entries when path is not valid
            return jnp.where(
                path_valid,
                jnp.array(
                    [
                        first_route.astype(float),
                        second_route.astype(float),
                        transfer_point.astype(float),
                        1.0,  # valid
                        total_cost,
                    ]
                ),
                jnp.array(
                    [
                        jnp.inf,  # invalid route
                        jnp.inf,  # invalid route
                        jnp.inf,  # invalid transfer point
                        0.0,  # invalid
                        jnp.inf,  # infinite cost
                    ]
                ),
            )

        # Vectorize the computation
        compute_path_cost_v = jax.vmap(compute_path_cost)

        # Compute all paths at once
        results = compute_path_cost_v(r1, r2, tp)

        # Sort by cost - valid paths will be first due to infinite costs for invalid paths
        sorted_indices = jnp.argsort(results[:, 4])
        sorted_results = results[sorted_indices]

        return sorted_results
