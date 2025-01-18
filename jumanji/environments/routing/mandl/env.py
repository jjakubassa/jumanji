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
from importlib import resources
from typing import Final, Literal, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import matplotlib
import pandas as pd
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
        network_name: Literal["mandl1", "ceder1"] = "ceder1",
        max_capacity: int = 40,
        simulation_steps: int = 240,
        num_flex_routes: int = 4,
        waiting_penalty_factor: float = 10.0,
        max_num_stops_flex: int = 20,
    ) -> None:
        self.network_name: Final = network_name
        self.max_capacity: Final = max_capacity
        self.simulation_steps: Final = simulation_steps
        self.num_flex_routes: Final = num_flex_routes
        self.waiting_penalty_factor: Final = jnp.array(waiting_penalty_factor)
        self.num_nodes: Final = {"mandl1": 15, "ceder1": 4}[network_name]
        self.max_num_routes: Final = 99
        self.max_demand: Final = 1024
        self.max_edge_weight: Final = 10
        self.max_num_stops_flex = max_num_stops_flex
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

        # Filter out routes with zero frequency
        fixed_routes = RouteBatch(
            ids=fixed_routes.ids,
            nodes=jnp.where(fixed_routes.frequencies[:, None] > 0, fixed_routes.nodes, -1),
            frequencies=fixed_routes.frequencies,
            on_demand=fixed_routes.on_demand,
        )

        # Initialize flexible routes (empty initially)
        flexible_routes = RouteBatch(
            ids=jnp.arange(num_fixed_routes, num_fixed_routes + self.num_flex_routes),
            nodes=jnp.full((self.num_flex_routes, self.max_num_stops_flex), -1, dtype=int),
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
        metrics = self.get_metrics(state=state)
        timestep = restart(observation=self._state_to_observation(state), extras=metrics)

        return state, timestep

    def _load_instance_data(
        self,
    ) -> tuple[NetworkData, jnp.ndarray]:
        """Load network and demand data from files.

        Args:
            network_name: Name of the network to load (e.g., "mandl1", "ceder1")

        Returns:
            NetworkData: Contains the network structure with nodes, links, and terminals
            demand: Matrix of shape (num_nodes, num_nodes) containing demand between nodes
        """
        # Load data from files
        assets_package = f"jumanji.environments.routing.mandl.assets.{self.network_name}"
        nodes_file = f"{self.network_name}_nodes.txt"
        links_file = f"{self.network_name}_links.txt"
        demand_file = f"{self.network_name}_demand.txt"

        with resources.files(assets_package).joinpath(nodes_file).open("r") as f:
            nodes_df = pd.read_csv(f)

        with resources.files(assets_package).joinpath(links_file).open("r") as f:
            links_df = pd.read_csv(f)

        with resources.files(assets_package).joinpath(demand_file).open("r") as f:
            demand_df = pd.read_csv(f)

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
        """Load fixed routes from solution file."""
        assets_package = f"jumanji.environments.routing.mandl.assets.{self.network_name}"
        solution_file = f"{self.network_name}_solution.txt"

        with resources.files(assets_package).joinpath(solution_file).open("r") as f:
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

        if len(routes[0]) > self.max_num_stops_flex:
            raise ValueError(f"Fixed routes exceed max_stops ({self.max_num_stops_flex})")

        # Pad routes to max length
        padded_routes = []
        for route in routes:
            padded_route = route + [-1] * (self.max_num_stops_flex - len(route))
            padded_routes.append(padded_route)

        return RouteBatch(
            ids=jnp.arange(num_routes),
            nodes=jnp.array(padded_routes),
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

        # Generate departure times
        key, departure_key = random.split(key)
        departure_times = random.uniform(
            departure_key, shape=(n_passengers,), minval=0, maxval=simulation_period
        )

        return PassengerBatch(
            ids=jnp.arange(n_passengers),
            origins=origins,
            destinations=destinations,
            departure_times=departure_times,
            time_waiting=jnp.zeros(n_passengers),
            time_in_vehicle=jnp.zeros(n_passengers),
            statuses=jnp.zeros(n_passengers),
        )

    def _init_vehicles(self, num_vehicles: int, max_route_length: int) -> VehicleBatch:
        """Initialize vehicles at the start of the simulation."""
        return VehicleBatch(
            ids=jnp.arange(num_vehicles),
            route_ids=jnp.arange(num_vehicles),
            current_edges=jnp.full(
                (num_vehicles, 2), -1, dtype=int
            ),  # Start with invalid edge (-1, -1) to indicate no route
            times_on_edge=jnp.zeros(num_vehicles),
            passengers=jnp.full((num_vehicles, self.max_capacity), -1, dtype=int),
            capacities=jnp.full(num_vehicles, self.max_capacity),
            directions=jnp.ones(num_vehicles, dtype=int),
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

        # Calculate action mask for valid next stops
        action_mask = self._get_action_mask(state)

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
            action_mask=action_mask,
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

        # Assign passengers to vehicles
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

        def create_done_timestep(args: tuple[State, jnp.ndarray]) -> TimeStep[Observation]:
            new_state, waiting_penalty_factor = args
            # Include both completed and in-vehicle passengers' journey times
            journey_times = jnp.sum(
                jnp.where(
                    (new_state.passengers.statuses == 3) | (new_state.passengers.statuses == 2),
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

            reward = -(journey_times + waiting_penalty)
            metrics = self.get_metrics(new_state)
            return termination(
                observation=self._state_to_observation(new_state), reward=reward, extras=metrics
            )

        def create_transition_timestep(args: tuple[State, jnp.ndarray]) -> TimeStep[Observation]:
            new_state, _ = args
            metrics = self.get_metrics(new_state)
            return transition(
                observation=self._state_to_observation(new_state),
                reward=jnp.array(0.0),
                discount=jnp.array(1.0),
                extras=metrics,
            )

        # Pass both new_state and waiting_penalty_factor to the conditional functions
        args = (new_state, self.waiting_penalty_factor)
        timestep = jax.lax.cond(done, create_done_timestep, create_transition_timestep, args)

        # jax.debug.print("routes: {r}", r=state.routes)
        # jax.debug.print("network: {n}", n=state.network)
        # jax.debug.print("vehicles: {v}", v=state.vehicles)
        return new_state, timestep

    def get_metrics(self, state: State) -> dict[str, jnp.ndarray]:
        # Get indices for fixed and flex routes
        num_fixed_routes = len(state.routes.ids) - self.num_flex_routes

        # Create route type masks for vehicles
        vehicle_route_types = state.vehicles.route_ids < num_fixed_routes  # True for fixed routes

        # Initialize passenger counters
        def count_passengers_by_type(is_fixed: bool):
            # Count in-vehicle passengers
            def count_for_vehicle(acc, vehicle_idx):
                is_vehicle_type = vehicle_route_types[vehicle_idx] == is_fixed
                vehicle_passengers = state.vehicles.passengers[vehicle_idx]
                valid_passengers = vehicle_passengers >= 0
                return acc + jnp.sum(valid_passengers & is_vehicle_type)

            return jax.lax.fori_loop(0, len(state.vehicles.ids), count_for_vehicle, 0)

        # Count passengers by route type
        passengers_in_fixed = count_passengers_by_type(True)
        passengers_in_flex = count_passengers_by_type(False)

        metrics = {
            "completed_passengers": jnp.sum(state.passengers.statuses == 3),
            "waiting_passengers": jnp.sum(state.passengers.statuses == 1),
            "in_vehicle_passengers": jnp.sum(state.passengers.statuses == 2),
            "in_fixed_vehicles": passengers_in_fixed,
            "in_flex_vehicles": passengers_in_flex,
            "total_waiting_time": jnp.sum(state.passengers.time_waiting),
            "total_in_vehicle_time": jnp.sum(state.passengers.time_in_vehicle),
            "capacity_left": jnp.sum(state.vehicles.capacities),
            "capacity_left_relative": jnp.sum(state.vehicles.capacities)
            / (len(state.vehicles.ids) * self.max_capacity),
        }

        # Add route-related metrics
        for i in range(self.num_flex_routes):
            route = state.routes.nodes[len(state.routes.ids) - self.num_flex_routes + i]
            metrics[f"route_{i}_length"] = jnp.sum(route != -1)
            metrics[f"route_{i}_wait_actions"] = jnp.sum(route == -1)

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

        def update_route(i: int, nodes: jnp.ndarray) -> jnp.ndarray:
            all_idx = num_fixed_routes + i
            route = nodes[all_idx]

            # Find the first empty position (-1)
            valid_mask = route != -1
            empty_pos = jnp.argmin(valid_mask)

            new_node = action[i]

            # Get previous node (if it exists)
            prev_node = jnp.where(empty_pos > 0, route[empty_pos - 1], -1)

            # Conditions for adding the new node:
            # 1. New node is valid (>= 0)
            # 2. New node is different from previous node
            # 3. There is a valid connection in the network (if not first node)
            valid_connection = jax.lax.cond(
                prev_node >= 0, lambda: links[prev_node, new_node] < jnp.inf, lambda: True
            )

            can_add = (new_node >= 0) & (new_node != prev_node) & valid_connection

            return jax.lax.cond(
                can_add,
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

        def process_vehicle(i: int, vehicles: VehicleBatch) -> VehicleBatch:
            route = routes[route_ids[i]]
            current_edge = current_edges[i]

            # Count valid nodes (not -1)
            route_length = jnp.sum(route != -1)

            def handle_invalid_route():
                # Find first valid node using argmax
                first_valid_idx = jnp.argmax(route != -1)
                first_valid_node = route[first_valid_idx]

                # If route has at least one valid node, place vehicle there
                return jax.lax.cond(
                    route_length > 0,
                    lambda: vehicles._replace(
                        current_edges=vehicles.current_edges.at[i].set(
                            jnp.array([first_valid_node, first_valid_node])
                        )
                    ),
                    lambda: vehicles,  # Keep vehicle at invalid edge if no valid nodes
                )

            def process_valid_route():
                # If vehicle is at invalid edge, initialize it at first node
                def initialize_vehicle():
                    first_valid_idx = jnp.argmax(route != -1)
                    first_valid_node = route[first_valid_idx]
                    return vehicles._replace(
                        current_edges=vehicles.current_edges.at[i].set(
                            jnp.array([first_valid_node, first_valid_node])
                        ),
                        times_on_edge=vehicles.times_on_edge.at[i].set(0),
                    )

                # Normal movement logic for valid edges
                def move_vehicle():
                    # Find current position in route
                    current_pos = jnp.argmax(route == current_edge[1])

                    next_pos = current_pos + directions[i]

                    # Handle end conditions
                    at_end = next_pos >= route_length
                    at_start = next_pos < 0

                    new_direction = jnp.where(at_end, -1, jnp.where(at_start, 1, directions[i]))

                    next_pos = jnp.where(at_end, route_length - 2, jnp.where(at_start, 1, next_pos))

                    # Get next valid node by scanning through the route
                    next_valid_count = jnp.sum(jnp.arange(len(route)) <= next_pos)
                    next_node = jnp.where(route != -1, route, -1)[next_valid_count - 1]

                    edge_time = network_links[current_edge[1], next_node]

                    return jax.lax.cond(
                        times_on_edge[i] >= edge_time,
                        lambda: vehicles._replace(
                            current_edges=vehicles.current_edges.at[i].set(
                                jnp.array([current_edge[1], next_node])
                            ),
                            times_on_edge=vehicles.times_on_edge.at[i].set(0),
                            directions=vehicles.directions.at[i].set(new_direction),
                        ),
                        lambda: vehicles._replace(
                            times_on_edge=vehicles.times_on_edge.at[i].set(times_on_edge[i] + 1)
                        ),
                    )

                # Check if vehicle needs initialization
                return jax.lax.cond(current_edge[0] == -1, initialize_vehicle, move_vehicle)

            # Check if route has enough nodes for movement
            return jax.lax.cond(route_length > 1, process_valid_route, handle_invalid_route)

        # Process each vehicle
        return jax.lax.fori_loop(0, len(vehicles.ids), process_vehicle, vehicles)

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

        def process_valid_route(vehicles: VehicleBatch) -> VehicleBatch:
            next_pos = current_pos + direction

            def handle_end(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                return vehicles.directions.at[idx].set(-1), route_len - 2

            def handle_start(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                return vehicles.directions.at[idx].set(1), jnp.ones(1)

            def handle_middle(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        return process_valid_route(vehicles)

    @staticmethod
    @profile_function
    def _assign_passengers(
        passengers: PassengerBatch,
        vehicles: VehicleBatch,
        routes: RouteBatch,
        network: NetworkData,
        current_time: int,
    ) -> tuple[PassengerBatch, VehicleBatch]:
        # Update waiting/in-vehicle times
        new_time_waiting = jnp.where(
            passengers.statuses == 1, passengers.time_waiting + 1, passengers.time_waiting
        )
        new_time_in_vehicle = jnp.where(
            passengers.statuses == 2, passengers.time_in_vehicle + 1, passengers.time_in_vehicle
        )

        n_passengers = len(passengers.ids)
        n_vehicles = len(vehicles.ids)
        capacity = vehicles.passengers.shape[1]

        # Track passenger locations
        passenger_location = jnp.full((n_passengers,), -1, dtype=jnp.int32)

        def update_passenger_locations(vehicle_idx, locations):
            vehicle_passengers = vehicles.passengers[vehicle_idx]
            vehicle_node = vehicles.current_edges[vehicle_idx, 1]

            def update_location(p_idx, locs):
                passenger_id = vehicle_passengers[p_idx]
                return jnp.where(passenger_id >= 0, locs.at[passenger_id].set(vehicle_node), locs)

            return jax.lax.fori_loop(0, capacity, update_location, locations)

        # Update locations for all vehicles
        passenger_location = jax.lax.fori_loop(
            0, n_vehicles, update_passenger_locations, passenger_location
        )

        # Initialize new statuses with current values
        new_statuses = passengers.statuses
        new_vehicles = vehicles

        def process_vehicle(vehicle_idx, state):
            cur_statuses, cur_vehicles = state
            vehicle_node = cur_vehicles.current_edges[vehicle_idx, 1]
            vehicle_passengers = cur_vehicles.passengers[vehicle_idx]

            def process_waiting_passenger(p_idx, state):
                p_status, v_passengers = state

                is_waiting = p_status[p_idx] == 1
                is_at_node = passengers.origins[p_idx] == vehicle_node
                has_space = jnp.sum(v_passengers < 0) > 0

                empty_seats = jnp.where(v_passengers < 0, jnp.arange(capacity), capacity)
                first_empty = jnp.min(empty_seats)

                can_board = is_waiting * is_at_node * has_space * (first_empty < capacity)

                new_p_status = jnp.where(can_board, 2, p_status[p_idx])
                new_v_passengers = jnp.where(
                    can_board, v_passengers.at[first_empty].set(p_idx), v_passengers
                )

                return p_status.at[p_idx].set(new_p_status), new_v_passengers

            final_status, final_passengers = jax.lax.fori_loop(
                0, n_passengers, process_waiting_passenger, (cur_statuses, vehicle_passengers)
            )

            new_vehicles = cur_vehicles._replace(
                passengers=cur_vehicles.passengers.at[vehicle_idx].set(final_passengers)
            )

            return final_status, new_vehicles

        final_statuses, final_vehicles = jax.lax.fori_loop(
            0, n_vehicles, process_vehicle, (new_statuses, new_vehicles)
        )

        # Update completed journeys
        final_statuses = jnp.where(
            (final_statuses == 2) & (passenger_location == passengers.destinations),
            3,  # completed
            final_statuses,
        )

        # Create updated passenger batch
        updated_passengers = PassengerBatch(
            ids=passengers.ids,
            origins=passengers.origins,
            destinations=passengers.destinations,
            departure_times=passengers.departure_times,
            time_waiting=new_time_waiting,
            time_in_vehicle=new_time_in_vehicle,
            statuses=final_statuses,
        )

        return updated_passengers, final_vehicles

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
                shape=(max_num_routes, self.max_num_stops_flex),
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
            action_mask=specs.BoundedArray(
                shape=(self.num_flex_routes, self.num_nodes + 1),  # +1 for wait action
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
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

    def _get_action_mask(self, state: State) -> jnp.ndarray:
        """Get mask for valid actions in current state."""
        num_fixed_routes = len(state.routes.ids) - self.num_flex_routes

        def get_route_mask(i: int) -> jnp.ndarray:
            route = state.routes.nodes[num_fixed_routes + i]

            # Find the first empty position (-1)
            valid_mask = route != -1
            empty_pos = jnp.argmin(valid_mask)

            # Get previous node (if it exists)
            prev_node = jnp.where(empty_pos > 0, route[empty_pos - 1], -1)

            def mask_if_has_prev_node(prev_node):
                return state.network.links[prev_node] < jnp.inf

            def mask_if_empty():
                # Can add any node as first stop
                return jnp.ones(self.num_nodes, dtype=bool)

            route_mask = jax.lax.cond(
                prev_node >= 0,
                mask_if_has_prev_node,
                lambda _: mask_if_empty(),
                prev_node,
            )

            # Wait action (-1) is always valid
            return jnp.concatenate([jnp.array([True]), route_mask])

        masks = jax.vmap(get_route_mask)(jnp.arange(self.num_flex_routes))
        return masks

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
                f"_get_route_shortest_paths expects a route vector or a single route in batch"
                f"form,  got {route.ndim} dimensions"
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
        def compute_path_cost(
            first_route: jnp.ndarray, second_route: jnp.ndarray, transfer_point: jnp.ndarray
        ) -> jnp.ndarray:
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
