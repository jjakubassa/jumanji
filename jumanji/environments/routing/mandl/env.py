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
    Passenger,
    State,
    TransferPath,
    Vehicle,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep, restart
from jumanji.viewer import Viewer


class Mandl(Environment[State, specs.DiscreteArray, Observation]):
    def __init__(
        self,
        viewer: Optional[Viewer] = None,
        max_capacity: int = 40,
        simulation_peridod: float = 60.0,
    ) -> None:
        self.max_capacity: Final = max_capacity
        self.simulation_peridod: Final = simulation_peridod
        self.num_nodes: Final = 15
        self.max_num_routes: Final = 99
        self.max_demand: Final = 1024
        self.max_edge_weight: Final = 10
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )
        super().__init__()

    def __repr__(self) -> str:
        raise NotImplementedError

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        network_data = self._load_network()
        passengers = self._generate_passengers(key, network_data.demand, self.simulation_peridod)
        state = State(
            network=network_data,
            vehicles=[],
            passengers=passengers,
            current_time=0.0,
            save_path=None,
            key=key,
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: chex.Numeric) -> tuple[State, TimeStep[Observation]]:
        # Update vehicle positions
        new_vehicles = []
        for vehicle in state.vehicles:
            new_vehicle = self._move_vehicle(vehicle, state.network.links)
            new_vehicles.append(new_vehicle)
        state = replace(state, vehicles=new_vehicles, current_time=state.current_time + 1)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def _move_vehicle(self, vehicle: Vehicle, travel_times: jnp.ndarray) -> Vehicle:
        """Move the vehicle along its route based on the time spent on the current edge."""
        time_on_edge = vehicle.time_on_edge + 1  # Increment time on edge by 1 minute
        start_node, end_node = vehicle.current_edge
        total_time = travel_times[start_node, end_node]

        print(
            f"Vehicle {vehicle.id} moving from {start_node} to {end_node}"
            f"with time on edge {time_on_edge}/{total_time}"
        )

        if time_on_edge >= total_time:
            print(f"Vehicle {vehicle.id} moving to next edge from {start_node} to {end_node}")
            # Move to the next edge
            route_nodes = vehicle.route.nodes
            current_index = route_nodes.tolist().index(start_node)
            next_index = (current_index + 1) % len(route_nodes)
            new_edge = (route_nodes[next_index], route_nodes[(next_index + 1) % len(route_nodes)])
            time_on_edge = 0  # Reset time on edge
        else:
            print(f"Vehicle {vehicle.id} staying on edge from {start_node} to {end_node}")
            new_edge = vehicle.current_edge

        return vehicle._replace(current_edge=new_edge, time_on_edge=time_on_edge)

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - network: BoundedArray (int) of shape (num_nodes, num_nodes).
            - demand_original: BoundedArray (int) of shape (num_nodes, num_nodes).
            - demand_now: BoundedArray (int) of shape (num_nodes, num_nodes).
            - routes: BoundedArray (int) of shape (max_num_routes, 2).
            - capacity_left: BoundedArray (int) of shape (max_num_routes,).
            - action_mask: BoundedArray (bool) of shape (max_num_routes, num_nodes).
        """
        network = specs.BoundedArray(
            shape=(self.num_nodes, self.num_nodes),
            minimum=0,
            maximum=self.max_edge_weight,
            dtype=int,
            name="network",
        )

        demand_original = specs.BoundedArray(
            shape=(self.num_nodes, self.num_nodes),
            minimum=0,
            maximum=self.max_demand,
            dtype=int,
            name="demand_original",
        )

        demand_now = specs.BoundedArray(
            shape=(self.num_nodes, self.num_nodes),
            minimum=0,
            maximum=self.max_demand,
            dtype=int,
            name="demand_now",
        )

        routes = specs.BoundedArray(
            shape=(self.max_num_routes, 2),
            minimum=0,
            maximum=self.num_nodes - 1,
            dtype=int,
            name="routes",
        )

        capacity_left = specs.BoundedArray(
            shape=(self.max_num_routes,),
            minimum=0,
            maximum=self.max_capacity,
            dtype=int,
            name="capacity_left",
        )

        action_mask = specs.BoundedArray(
            shape=(self.max_num_routes, self.num_nodes),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            demand_original=demand_original,
            demand_now=demand_now,
            network=network,
            routes=routes,
            capacity_left=capacity_left,
            action_mask=action_mask,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(2, name="action")

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
        raise NotImplementedError

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
        raise NotImplementedError

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state into an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """
        raise NotImplementedError

    def _assign_passengers(self, state: State) -> None:
        raise NotImplementedError

    @staticmethod
    def _has_no_cycle(route: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _find_paths(
        self,
        network: NetworkData,
        routes: jnp.ndarray,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> tuple[list[DirectPath], list[TransferPath]]:
        """Find all valid paths between start and end, including transfers"""
        # Calculate shortest paths for each route
        route_paths = []
        for r in range(routes.shape[0]):
            route_paths.append(self._get_route_shortest_paths(network, routes[r]))

        # Always check direct paths first
        direct_paths = self._find_direct_paths(route_paths, start, end)

        # Only look for transfer paths if no direct paths exist
        transfer_paths = (
            []
            if direct_paths
            else self._find_transfer_paths(route_paths, start, end, transfer_penalty)
        )

        return direct_paths, transfer_paths

    @staticmethod
    def _get_route_shortest_paths(network: NetworkData, route: jnp.ndarray) -> jnp.ndarray:
        """Get shortest paths for a route by masking the network"""
        dist = jnp.where(route == 1, network.links, jnp.inf)

        # Set diagonal elements to 0 initially
        dist = dist.at[jnp.diag_indices_from(dist)].set(0)

        # Floyd-Warshall
        n_stops = len(dist)
        for intermediate in range(n_stops):
            for start in range(n_stops):
                for end in range(n_stops):
                    dist_via_intermediate = dist[start, intermediate] + dist[intermediate, end]
                    dist = dist.at[start, end].set(
                        jnp.minimum(dist[start, end], dist_via_intermediate)
                    )
        return dist

    @staticmethod
    def _find_direct_paths(
        route_paths: list[jnp.ndarray], start: int, end: int
    ) -> list[DirectPath]:
        direct_paths = []
        for r in range(len(route_paths)):
            cost = route_paths[r][start, end]
            if cost < jnp.inf:
                direct_paths.append(DirectPath(route=r, cost=float(cost), start=start, end=end))
        return sorted(direct_paths, key=lambda x: x.cost)

    @staticmethod
    def _find_transfer_paths(
        route_paths: list[jnp.ndarray],
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> list[TransferPath]:
        transfer_paths = []
        n_routes = len(route_paths)
        n_stops = len(route_paths[0])

        for r1 in range(n_routes):
            for r2 in range(n_routes):
                if r1 == r2:
                    continue
                for transfer_stop in range(n_stops):
                    # Skip transfers at start and end nodes
                    if transfer_stop == start or transfer_stop == end:
                        continue

                    cost1 = route_paths[r1][start, transfer_stop]
                    cost2 = route_paths[r2][transfer_stop, end]

                    if cost1 < jnp.inf and cost2 < jnp.inf:
                        total_cost = cost1 + cost2 + transfer_penalty
                        transfer_paths.append(
                            TransferPath(
                                first_route=r1,
                                second_route=r2,
                                total_cost=float(total_cost),
                                transfer_stop=transfer_stop,
                                start=start,
                                end=end,
                            )
                        )
        return sorted(transfer_paths, key=lambda x: x.total_cost)

    def _load_network(self, path: str = "jumanji/environments/routing/mandl") -> NetworkData:
        """Load network data from files"""

        nodes_df = pd.read_csv(f"{path}/mandl1_nodes.txt")
        links_df = pd.read_csv(f"{path}/mandl1_links.txt")
        demand_df = pd.read_csv(f"{path}/mandl1_demand.txt")
        icon_path = f"{path}/bus-transportation-public-svgrepo-com.png"

        # Process nodes
        nodes = jnp.array(nodes_df[["lat", "lon"]].values)
        terminals = jnp.array(nodes_df["terminal"].values, dtype=bool)

        # Normalize coordinates
        min_coords = nodes.min(axis=0)
        max_coords = nodes.max(axis=0)
        nodes = (nodes - min_coords) / (max_coords - min_coords)

        # Create adjacency matrix with travel times
        n_nodes = len(nodes_df)
        travel_times = jnp.full((n_nodes, n_nodes), jnp.inf)
        travel_times = travel_times.at[jnp.diag_indices_from(travel_times)].set(0)
        for _, row in links_df.iterrows():
            travel_times = travel_times.at[int(row["from"] - 1), int(row["to"] - 1)].set(
                row["travel_time"]
            )

        # Process demand into matrix form
        demand = jnp.zeros((n_nodes, n_nodes))
        for _, row in demand_df.iterrows():
            demand = demand.at[int(row["from"] - 1), int(row["to"] - 1)].set(row["demand"])

        return NetworkData(
            nodes=nodes, links=travel_times, demand=demand, terminals=terminals, icon_path=icon_path
        )

    def _generate_passengers(
        self,
        key: chex.PRNGKey,
        demand_matrix: jnp.ndarray,
        simulation_period: float = 60.0,  # minutes
    ) -> list[Passenger]:
        """Generate passengers according to demand matrix"""
        n_nodes = demand_matrix.shape[0]
        passengers = []
        passenger_id = 0

        # Convert hourly demand to simulation period
        period_demand = demand_matrix * (simulation_period / 60.0)

        # For each OD pair
        for origin in range(n_nodes):
            for destination in range(n_nodes):
                if origin != destination:
                    n_passengers = int(period_demand[origin, destination])

                    if n_passengers > 0:
                        key, subkey = random.split(key)
                        departure_times = random.uniform(
                            subkey, shape=(n_passengers,), minval=0, maxval=simulation_period
                        )

                        for t in departure_times:
                            passengers.append(
                                Passenger(
                                    id=passenger_id,
                                    origin=origin,
                                    destination=destination,
                                    departure_time=float(t),
                                    status=0,
                                )
                            )
                            passenger_id += 1

        passengers.sort(key=lambda p: p.departure_time)
        return passengers
