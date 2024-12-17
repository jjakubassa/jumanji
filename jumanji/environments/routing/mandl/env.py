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
from jax import random

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.types import (
    DirectPath,
    NetworkData,
    Observation,
    PassengerBatch,
    RouteBatch,
    State,
    TransferPath,
    VehicleBatch,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep, restart
from jumanji.viewer import Viewer


class Mandl(Environment[State, specs.DiscreteArray, Observation]):
    def __init__(
        self,
        viewer: Optional[Viewer] = None,
        max_capacity: int = 40,
        simulation_steps: int = 60,
        num_flex_routes: int = 2,
    ) -> None:
        self.max_capacity: Final = max_capacity
        self.simulation_steps: Final = simulation_steps
        self.num_flex_routes: Final = num_flex_routes
        self.num_nodes: Final = 15
        self.max_num_routes: Final = 99
        self.max_demand: Final = 1024
        self.max_edge_weight: Final = 10
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )
        self.num_passengers: int | None = None
        self._shortest_paths_cache: dict[int, jnp.ndarray] = {}
        super().__init__()

    def __repr__(self) -> str:
        raise NotImplementedError

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state.

        Args:
            key: Random key for initialization

        Returns:
            state: Initial state of the environment
            timestep: Initial timestep with observation
        """
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

        # Split key for different random operations
        key, passengers_key = random.split(key)

        # Initialize vehicles
        total_routes = len(routes.ids)
        vehicles = self._init_vehicles(num_vehicles=total_routes, max_route_length=self.num_nodes)

        # Initialize passengers
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
        timestep = restart(observation=self._state_to_observation(state, t=self.simulation_steps))

        return state, timestep

    def _load_instance_data(
        self, path: str = "jumanji/environments/routing/mandl"
    ) -> tuple[NetworkData, jnp.ndarray]:
        """Load network and demand data from files.

        Args:
            path: Path to the directory containing instance data files

        Returns:
            NetworkData: Contains the network structure with nodes, links, and terminals
            demand: Matrix of shape (num_nodes, num_nodes) containing demand between nodes
        """
        # Load data from files
        nodes_df = pd.read_csv(f"{path}/mandl1_nodes.txt")
        links_df = pd.read_csv(f"{path}/mandl1_links.txt")
        demand_df = pd.read_csv(f"{path}/mandl1_demand.txt")

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
        self, path: str = "jumanji/environments/routing/mandl/mandl1_solution.txt"
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
        """Generate passengers according to demand matrix.

        Args:
            key: Random key for generating departure times
            demand_matrix: Matrix of shape (num_nodes, num_nodes) containing hourly demand
                between nodes
            simulation_period: Length of simulation in minutes

        Returns:
            PassengerBatch: Initial state of all passengers

        Passenger status codes:
            0: not yet in system (before departure time)
            1: waiting for vehicle
            2: in vehicle
            3: completed journey
        """
        n_nodes = demand_matrix.shape[0]
        origins = []
        destinations = []
        departure_times = []
        passenger_id = 0

        # Convert hourly demand to simulation period
        period_demand = demand_matrix * (simulation_period / 60)

        # For each OD pair
        for origin in range(n_nodes):
            for destination in range(n_nodes):
                if origin != destination:
                    n_passengers = int(period_demand[origin, destination])

                    if n_passengers > 0:
                        key, subkey = random.split(key)
                        departure_times_batch = random.uniform(
                            subkey, shape=(n_passengers,), minval=0, maxval=simulation_period
                        )

                        origins.extend([origin] * n_passengers)
                        destinations.extend([destination] * n_passengers)
                        departure_times.extend(departure_times_batch)
                        passenger_id += n_passengers

        # Convert lists to arrays
        origins = jnp.array(origins)
        destinations = jnp.array(destinations)
        departure_times = jnp.array(departure_times)
        num_passengers = len(origins)

        return PassengerBatch(
            ids=jnp.arange(num_passengers),
            origins=origins,
            destinations=destinations,
            departure_times=departure_times,
            time_waiting=jnp.zeros(num_passengers),  # Initially no waiting time
            time_in_vehicle=jnp.zeros(num_passengers),  # Initially no in-vehicle time
            statuses=jnp.zeros(
                num_passengers, dtype=int
            ),  # All passengers start as "not yet in system" (0)
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

    def _state_to_observation(self, state: State, t: int) -> Observation:
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
        )

    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep[Observation]]:
        """Execute one environment step.

        Args:
            state: Current state of the environment
            action: Action to take

        Returns:
            new_state: Updated state after taking action
            timestep: New timestep with updated observation
        """

        # Validate action shape
        if not isinstance(action, (jnp.ndarray, jnp.ndarray)) or action.shape != (
            self.num_flex_routes,
        ):
            raise ValueError(
                f"Action must have shape ({self.num_flex_routes},), got {action.shape}"
            )

        self._shortest_paths_cache.clear()
        new_routes = self._update_flexible_routes(state.routes, action, state.network.links)
        new_state = replace(state, routes=new_routes)
        new_vehicles = self._move_vehicles(new_state)
        new_state = self._update_passenger_status(replace(new_state, vehicles=new_vehicles))
        new_state = self._assign_passengers(new_state)
        new_state = replace(new_state, current_time=state.current_time + 1)

        # Create timestep with new observation
        timestep = restart(
            observation=self._state_to_observation(new_state, t=self.simulation_steps)
        )

        return new_state, timestep

    def _update_flexible_routes(
        self, routes: RouteBatch, action: jnp.ndarray, links: jnp.ndarray
    ) -> RouteBatch:
        """Update flexible routes based on actions."""
        new_nodes = routes.nodes.copy()
        flexible_mask = routes.on_demand

        for flex_idx, all_idx in enumerate(jnp.where(flexible_mask)[0]):
            if action[flex_idx] != self.num_nodes:  # If not wait action
                route = routes.nodes[all_idx]
                empty_pos = jnp.where(route == -1)[0][0]
                new_node = action[flex_idx].item()
                if self._is_valid_next_stop(route, new_node, links):
                    new_nodes = new_nodes.at[all_idx, empty_pos].set(new_node)

        return routes._replace(nodes=new_nodes)

    def _is_valid_next_stop(self, route: jnp.ndarray, new_node: int, links: jnp.ndarray) -> bool:
        """Check if new_node is a valid next stop for the route."""
        # Get last valid node in route
        valid_nodes = route[route != -1]
        if len(valid_nodes) == 0:
            return True  # Any node is valid for empty route

        last_node = valid_nodes[-1]

        # Check if nodes are connected in network
        return bool((links[last_node, new_node] < jnp.inf).item())

    def _move_vehicles(self, state: State) -> VehicleBatch:
        """Update vehicle positions based on their routes and current edges.

        Args:
            state: Current state of the environment

        Returns:
            Updated vehicle batch with new positions
        """
        new_vehicles = state.vehicles._replace()  # Create a copy to modify

        # For each vehicle
        for idx in range(len(state.vehicles.ids)):
            current_time = state.vehicles.times_on_edge[idx]
            current_edge = state.vehicles.current_edges[idx]
            edge_time = state.network.links[current_edge[0], current_edge[1]]

            # If vehicle has completed current edge
            if current_time >= edge_time:
                # Get vehicle's route
                route = state.routes.nodes[state.vehicles.route_ids[idx]]
                valid_nodes = route[route != -1]

                if len(valid_nodes) > 0:  # Only process if route is not empty
                    # Find next node in route
                    current_pos = jnp.where(valid_nodes == current_edge[1])[0]
                    if len(current_pos) > 0:  # Only process if current node is in route
                        current_pos = current_pos[0]
                        next_pos = current_pos + state.vehicles.directions[idx]

                        if next_pos >= len(valid_nodes):
                            # Reverse direction at end of route
                            new_vehicles = new_vehicles._replace(
                                directions=new_vehicles.directions.at[idx].set(-1)
                            )
                            next_pos = len(valid_nodes) - 2
                        elif next_pos < 0:
                            # Reverse direction at start of route
                            new_vehicles = new_vehicles._replace(
                                directions=new_vehicles.directions.at[idx].set(1)
                            )
                            next_pos = 1

                        # Update edge
                        next_node = valid_nodes[next_pos]
                        new_vehicles = new_vehicles._replace(
                            current_edges=new_vehicles.current_edges.at[idx].set(
                                jnp.array([current_edge[1], next_node], dtype=int)
                            ),
                            times_on_edge=new_vehicles.times_on_edge.at[idx].set(0),
                        )
            else:
                # Increment time on current edge
                new_vehicles = new_vehicles._replace(
                    times_on_edge=new_vehicles.times_on_edge.at[idx].set(current_time + 1)
                )

        return new_vehicles

    def _update_passenger_status(self, state: State) -> State:
        """Update passenger statuses based on current time and position.

        Args:
            state: Current state of the environment

        Returns:
            Updated state with new passenger statuses and vehicle occupancy
        """
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
        # and remove passengers from vehicles
        new_vehicle_passengers = new_vehicles.passengers

        for vehicle_idx, vehicle_passengers in enumerate(state.vehicles.passengers):
            current_edge = state.vehicles.current_edges[vehicle_idx]
            current_node = current_edge[1]  # Destination node of current edge

            # Check each passenger in the vehicle
            for seat_idx, passenger_idx in enumerate(vehicle_passengers):
                if passenger_idx != -1:  # Skip empty seats
                    # Check if vehicle is at passenger's destination
                    if current_node == new_passengers.destinations[passenger_idx]:
                        # Update passenger status to completed
                        new_statuses = new_statuses.at[passenger_idx].set(3)

                        # Remove passenger from vehicle
                        new_vehicle_passengers = new_vehicle_passengers.at[
                            vehicle_idx, seat_idx
                        ].set(-1)

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

    def _assign_passengers(self, state: State) -> State:
        """Assign passengers to vehicles based on current state.

        Args:
            state: Current state of the environment

        Returns:
            Updated state with new passenger assignments
        """
        new_vehicles = state.vehicles
        new_passengers = state.passengers

        # Update passenger statuses based on current time
        # Change status from 0 (not in system) to 1 (waiting) when departure time is reached
        new_statuses = jnp.where(
            (new_passengers.statuses == 0) & (new_passengers.departure_times <= state.current_time),
            1,
            new_passengers.statuses,
        )
        new_passengers = new_passengers._replace(statuses=new_statuses)

        # For each waiting passenger (status 1), try to assign to a vehicle
        waiting_passengers = jnp.where(new_passengers.statuses == 1)[0]

        for passenger_idx in waiting_passengers:
            origin = new_passengers.origins[passenger_idx].item()
            destination = new_passengers.destinations[passenger_idx].item()

            # Find possible paths for this passenger
            direct_paths, transfer_paths = self._find_paths(
                network=state.network,
                routes=state.routes,
                start=origin,
                end=destination,
                transfer_penalty=2.0,
            )

            # Try to assign to direct path first
            assigned = False
            for path in direct_paths:
                # Find vehicles serving this route
                route_vehicles = jnp.where(new_vehicles.route_ids == path.route)[0]

                for vehicle_idx in route_vehicles:
                    # Check if vehicle has capacity and is at or approaching the origin
                    current_edge = new_vehicles.current_edges[vehicle_idx]
                    at_origin = current_edge[0] == origin or current_edge[1] == origin
                    has_capacity = jnp.any(new_vehicles.passengers[vehicle_idx] == -1)

                    if at_origin and has_capacity:
                        # Assign passenger to vehicle
                        empty_seat = jnp.where(new_vehicles.passengers[vehicle_idx] == -1)[0][0]
                        new_vehicles = new_vehicles._replace(
                            passengers=new_vehicles.passengers.at[vehicle_idx, empty_seat].set(
                                passenger_idx
                            )
                        )
                        # Update passenger status to in-vehicle (2)
                        new_passengers = new_passengers._replace(
                            statuses=new_passengers.statuses.at[passenger_idx].set(2)
                        )
                        assigned = True
                        break
                if assigned:
                    break

            # TODO: Handle transfer paths if needed
            # For now, passengers remain waiting if no direct assignment is possible

        return replace(state, vehicles=new_vehicles, passengers=new_passengers)

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
            nodes=specs.Array(
                shape=(max_num_routes, num_nodes),
                dtype=bool,
                name="nodes",
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
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """
        Action space for each flexible route:
        - 0 to num_nodes-1: choose that node as next stop
        - num_nodes: wait/don't append stop
        """
        return specs.DiscreteArray(
            num_values=self.num_nodes + 1,  # includes wait action
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
        """Return a mask that indicates which nodes should not be considered for the next step in
        any route.

        Args:
            routes: A matrix of shape (num_routes, 2) with node indices representing origin and
                 destination for each route
            action_space: A matrix of shape (num_routes, num_nodes) indicating valid actions
            network: A matrix of shape (num_nodes, num_nodes) with 1 indicating connections between
                 nodes

        Returns:
            A boolean mask of shape (num_routes, num_nodes) with True indicating nodes that should
            be considered for each route

        Example:
            >>> mandl = Mandl()
            >>> network = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            >>> routes = jnp.array([[0, 1], [-1, -1]])
            >>> action_space = jnp.ones((2, 3))
            >>> mandl.get_action_mask(routes, action_space, network)
            Array([[False,  True, False],
                   [False, False, False]], dtype=bool)
        """

        def _body_fun(i: int, val: jnp.ndarray) -> jnp.ndarray:
            mask = jnp.where(
                jnp.any(routes[i] != -1),
                self._get_mask_per_route(network, routes[i]),
                jnp.zeros_like(routes[i], jnp.bool),
            )
            return val.at[i].set(mask)

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

    def _find_paths(
        self,
        network: NetworkData,
        routes: RouteBatch,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> tuple[list[DirectPath], list[TransferPath]]:
        # Always check direct paths first
        direct_paths = self._find_direct_paths(network=network, routes=routes, start=start, end=end)

        # Only look for transfer paths if no direct paths exist
        transfer_paths = (
            []
            if direct_paths
            else self._find_transfer_paths(
                network=network,
                routes=routes,
                start=start,
                end=end,
                transfer_penalty=transfer_penalty,
            )
        )

        return direct_paths, transfer_paths

    @staticmethod
    def _get_route_shortest_paths(network: NetworkData, route: jnp.ndarray) -> jnp.ndarray:
        """Calculate shortest paths between all pairs of nodes for a given route."""
        n_nodes = network.links.shape[0]
        dist = jnp.full((n_nodes, n_nodes), jnp.inf)

        # Get valid nodes (remove padding)
        valid_nodes = route[route != -1]

        if len(valid_nodes) <= 1:
            return dist.at[jnp.diag_indices_from(dist)].set(0)

        # Create a masked version of the network links
        masked_links = jnp.full((n_nodes, n_nodes), jnp.inf)

        # Set diagonal elements to 0
        masked_links = masked_links.at[jnp.diag_indices_from(masked_links)].set(0)

        # Only add links between consecutive nodes in the route
        for i in range(len(valid_nodes) - 1):
            from_node = valid_nodes[i]
            to_node = valid_nodes[i + 1]
            masked_links = masked_links.at[from_node, to_node].set(
                network.links[from_node, to_node]
            )
            masked_links = masked_links.at[to_node, from_node].set(
                network.links[to_node, from_node]
            )

        # Floyd-Warshall only for nodes in the route
        route_nodes_set = set(valid_nodes.tolist())
        dist = masked_links
        for k in range(n_nodes):
            if k not in route_nodes_set:
                continue
            for i in range(n_nodes):
                if i not in route_nodes_set:
                    continue
                for j in range(n_nodes):
                    if j not in route_nodes_set:
                        continue
                    if dist[i, k] < jnp.inf and dist[k, j] < jnp.inf:
                        new_dist = dist[i, k] + dist[k, j]
                        if new_dist < dist[i, j]:
                            dist = dist.at[i, j].set(new_dist)

        return dist

    def _get_all_shortest_paths(self, network: NetworkData, routes: RouteBatch) -> jnp.ndarray:
        # Create a cache key from the routes
        cache_key = hash(routes.nodes.tobytes())

        if cache_key in self._shortest_paths_cache:
            return self._shortest_paths_cache[cache_key]

        # If not in cache, compute the paths
        num_routes = len(routes.ids)
        route_paths = jnp.zeros((num_routes, network.links.shape[0], network.links.shape[1]))
        for i in range(num_routes):
            route_paths = route_paths.at[i].set(
                self._get_route_shortest_paths(network, routes.nodes[i])
            )

        # Store in cache
        self._shortest_paths_cache[cache_key] = route_paths
        return route_paths

    def _find_direct_paths(
        self, network: NetworkData, routes: RouteBatch, start: int, end: int
    ) -> list[DirectPath]:
        """Find all direct paths between start and end nodes.

        Args:
            route_paths: List of route batches containing route definitions
            start: Starting node index
            end: Ending node index

        Returns:
            List of direct paths sorted by cost
        """
        shortest_paths_costs = self._get_all_shortest_paths(network=network, routes=routes)
        direct_paths = []

        paths_start_end = shortest_paths_costs[:, start, end]

        # Check each route for possible direct paths
        for route_idx, path_costs in enumerate(paths_start_end):
            # If there's a valid path from start to end in this route
            if path_costs < jnp.inf:
                direct_paths.append(
                    DirectPath(start=start, end=end, route=route_idx, cost=path_costs.item())
                )

        # Sort paths by cost
        return sorted(direct_paths, key=lambda p: p.cost)

    def _find_transfer_paths(
        self,
        network: NetworkData,
        routes: RouteBatch,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> list[TransferPath]:
        """Find all transfer paths between start and end nodes.

        Args:
            network: Network structure containing nodes and links
            routes: Batch of routes
            start: Starting node index
            end: Ending node index
            transfer_penalty: Cost penalty for making a transfer

        Returns:
            List of transfer paths sorted by total cost
        """
        shortest_paths_costs = self._get_all_shortest_paths(network=network, routes=routes)
        transfer_paths = []
        num_routes = len(routes.ids)

        # Check all possible pairs of routes for transfers
        for first_route in range(num_routes):
            for second_route in range(num_routes):
                if first_route == second_route:
                    continue

                # Check each possible transfer point
                for transfer_point in range(network.nodes.shape[0]):
                    # Check if transfer point is reachable in both routes
                    first_leg = shortest_paths_costs[first_route][start, transfer_point]
                    second_leg = shortest_paths_costs[second_route][transfer_point, end]

                    # Check if we can get from start to transfer point on first route
                    # and from transfer point to end on second route
                    if (first_leg < jnp.inf) and (second_leg < jnp.inf):
                        total_cost = first_leg + second_leg + transfer_penalty
                        transfer_paths.append(
                            TransferPath(
                                start=start,
                                end=end,
                                first_route=first_route,
                                second_route=second_route,
                                transfer_stop=transfer_point,
                                total_cost=float(total_cost),
                            )
                        )

        # Sort paths by total cost
        return sorted(transfer_paths, key=lambda p: p.total_cost)
