from functools import cached_property
from typing import Final, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import matplotlib

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.types import (
    DirectPath,
    Network,
    Observation,
    Routes,
    State,
    TransferPath,
)
from jumanji.types import TimeStep, restart


class Mandl(Environment[State, specs.DiscreteArray, Observation]):
    def __init__(self, max_capacity: int = 40) -> None:
        self.max_capacity = max_capacity
        self.num_nodes: Final = 15
        self.max_num_routes: Final = 99
        self.max_demand: Final = 1024
        self.max_edge_weight: Final = 10
        super().__init__()

    def __repr__(self) -> str:
        raise NotImplementedError

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        state = State(
            # fmt: off
            demand=jnp.array(
                [
                    #   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                    [0, 400, 200, 60, 80, 150, 75, 75, 30, 160, 30, 25, 35, 0, 0],  # 1
                    [400, 0, 50, 120, 20, 180, 90, 90, 15, 130, 20, 10, 10, 5, 0],  # 2
                    [200, 50, 0, 40, 60, 180, 90, 90, 15, 45, 20, 10, 10, 5, 0],  # 3
                    [60, 120, 40, 0, 50, 100, 50, 50, 15, 240, 40, 25, 10, 5, 0],  # 4
                    [80, 20, 60, 50, 0, 50, 25, 25, 10, 120, 20, 15, 5, 0, 0],  # 5
                    [
                        150,
                        180,
                        180,
                        100,
                        50,
                        0,
                        100,
                        100,
                        30,
                        880,
                        60,
                        15,
                        15,
                        10,
                        0,
                    ],  # 6
                    [75, 90, 90, 50, 25, 100, 0, 50, 15, 440, 35, 10, 10, 5, 0],  # 7
                    [75, 90, 90, 50, 25, 100, 50, 0, 15, 440, 35, 10, 10, 5, 0],  # 8
                    [30, 15, 15, 15, 10, 30, 15, 15, 0, 140, 20, 5, 0, 0, 0],  # 9
                    [
                        160,
                        130,
                        45,
                        240,
                        120,
                        880,
                        440,
                        440,
                        140,
                        0,
                        600,
                        250,
                        500,
                        200,
                        0,
                    ],  # 10
                    [30, 20, 20, 40, 20, 60, 35, 35, 20, 600, 0, 75, 95, 15, 0],  # 11
                    [25, 10, 10, 25, 15, 15, 10, 10, 5, 250, 75, 0, 70, 0, 0],  # 12
                    [35, 10, 10, 10, 5, 15, 10, 10, 0, 500, 95, 70, 0, 45, 0],  # 13
                    [0, 5, 5, 5, 0, 10, 5, 5, 0, 200, 15, 0, 45, 0, 0],  # 14
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                ],
                dtype=jnp.int32,
            ),
            # fmt: on
            # fmt: off
            network=jnp.array(  # not verified yet
                [
                    # 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
                    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                    [8, 0, 2, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                    [0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                    [0, 3, 0, 0, 4, 4, 0, 0, 0, 0, 0, 10, 0, 0, 0],  # 4
                    [0, 6, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                    [0, 0, 3, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3],  # 6
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 2],  # 7
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 2],  # 8
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],  # 9
                    [0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 5, 0, 10, 8, 0],  # 10
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 10, 5, 0, 0],  # 11
                    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0],  # 12
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 0, 0, 2, 0],  # 13
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0],  # 14
                    [0, 0, 0, 0, 0, 3, 2, 2, 8, 0, 0, 0, 0, 0, 0],  # 15
                ],
                dtype=jnp.int32,
            ),
            # fmt: on
            routes=jnp.full((self.max_num_routes, 2), -1, dtype=jnp.int32),
            capacity=jnp.full(
                (self.max_num_routes,), self.max_capacity, dtype=jnp.int32
            ),
            key=key,
        )

        timestep = restart(observation=self._state_to_observation(state))

        return state, timestep

    def step(
        self, state: State, action: chex.Numeric
    ) -> tuple[State, TimeStep[Observation]]:
        raise NotImplementedError

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
        raise NotImplementedError

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def get_action_mask(
        self, routes: jnp.ndarray, action_space: jnp.ndarray, network: jnp.ndarray
    ) -> jnp.ndarray:
        """Return a mask that indicates which nodes should not be considered for the next step in any route.

        Args:
            routes: A matrix of shape (num_routes, 2) with node indices representing origin and destination for each route
            action_space: A matrix of shape (num_routes, num_nodes) indicating valid actions
            network: A matrix of shape (num_nodes, num_nodes) with 1 indicating connections between nodes

        Returns:
            A boolean mask of shape (num_routes, num_nodes) with True indicating nodes that should not be
            considered for each route

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

    def _get_mask_per_route(
        self, network: jnp.ndarray, route: jnp.ndarray
    ) -> jnp.ndarray:
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

    def get_route_shortest_paths(
        self, network: Network, routes: Routes, route_idx: int
    ) -> jnp.ndarray:
        """Get shortest paths for a route by masking the network"""
        route = routes.route_edges[route_idx]
        dist = jnp.where(route == 1, network.weights, jnp.inf)

        # Set diagonal elements to 0 initially
        dist = dist.at[jnp.diag_indices_from(dist)].set(0)

        # Simple Floyd-Warshall with JAX
        n_stops = len(dist)
        for intermediate in range(n_stops):
            for start in range(n_stops):
                for end in range(n_stops):
                    dist_via_intermediate = (
                        dist[start, intermediate] + dist[intermediate, end]
                    )
                    dist = dist.at[start, end].set(
                        jnp.minimum(dist[start, end], dist_via_intermediate)
                    )
        return dist

    def find_direct_paths(
        self, route_paths: list[jnp.ndarray], start: int, end: int
    ) -> list[DirectPath]:
        direct_paths = []
        for r in range(len(route_paths)):
            cost = route_paths[r][start, end]
            if cost < jnp.inf:
                direct_paths.append(
                    DirectPath(route=r, cost=float(cost), start=start, end=end)
                )
        return sorted(direct_paths, key=lambda x: x.cost)

    def find_transfer_paths(
        self,
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

    def find_paths(
        self,
        network: Network,
        routes: Routes,
        start: int,
        end: int,
        transfer_penalty: float = 2.0,
    ) -> tuple[list[DirectPath], list[TransferPath]]:
        """Find all valid paths between start and end, including transfers"""
        # Calculate shortest paths for each route
        route_paths = []
        for r in range(routes.n_routes):
            route_paths.append(self.get_route_shortest_paths(network, routes, r))

        # Always check direct paths first
        direct_paths = self.find_direct_paths(route_paths, start, end)

        # Only look for transfer paths if no direct paths exist
        transfer_paths = (
            []
            if direct_paths
            else self.find_transfer_paths(route_paths, start, end, transfer_penalty)
        )

        return direct_paths, transfer_paths
