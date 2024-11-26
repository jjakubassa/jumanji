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

import jax
import pytest
from jax import numpy as jnp

from jumanji.environments.routing.mandl.env import Mandl
from jumanji.environments.routing.mandl.types import (
    DirectPath,
    NetworkData,
    TransferPath,
)


@pytest.fixture
def mandl_env() -> Mandl:
    return Mandl()


class TestMandl:
    # def test_mandl_reset(self, mandl_env: Mandl) -> None:
    #     raise NotImplementedError

    @pytest.mark.skip()
    def test_has_no_cycles(self) -> None:
        test_routes_0 = jnp.array([[1, 2]])  # Not a cycle
        test_routes_1 = jnp.array([[2, 3]])  # Not a cycle
        test_routes_2 = jnp.array([[1, 0]])  # Not a cycle with termination
        test_routes_3 = jnp.array([[-1, -1]])  # Empty route
        test_routes_4 = jnp.array([[1, 2], [2, 3]])  # Two routes without cycles
        test_routes_5 = jnp.array([[1, 2], [5, 6]])  # Two routes without cycles
        test_routes_6 = jnp.array([[1, 2], [2, 1]])  # Cycle between two routes
        test_routes_7 = jnp.array([[1, 2], [2, 3], [3, 1]])  # Cycle among three routes

        # Test unjitted version
        assert Mandl._has_no_cycle(test_routes_0)
        assert Mandl._has_no_cycle(test_routes_1)
        assert Mandl._has_no_cycle(test_routes_2)
        assert Mandl._has_no_cycle(test_routes_3)
        assert Mandl._has_no_cycle(test_routes_4)
        assert Mandl._has_no_cycle(test_routes_5)
        assert not Mandl._has_no_cycle(test_routes_6)
        assert not Mandl._has_no_cycle(test_routes_7)

        # Test jitted version
        jitted_has_no_cycle = jax.jit(Mandl._has_no_cycle)
        assert jitted_has_no_cycle(test_routes_0)
        assert jitted_has_no_cycle(test_routes_1)
        assert jitted_has_no_cycle(test_routes_2)
        assert jitted_has_no_cycle(test_routes_3)
        assert jitted_has_no_cycle(test_routes_4)
        assert jitted_has_no_cycle(test_routes_5)
        assert not jitted_has_no_cycle(test_routes_6)
        assert not jitted_has_no_cycle(test_routes_7)


class TestGetRouteShortestPaths:
    def test_single_edge_route(self) -> None:
        """Test shortest path calculation for a route with a single edge."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_linear_route(self) -> None:
        """Test shortest path calculation for a linear route (stops connected in sequence)."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf, jnp.inf],
                    [2, 0, 3, jnp.inf],
                    [jnp.inf, 3, 0, 1],
                    [jnp.inf, jnp.inf, 1, 0],
                ]
            ),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array([[0, 2, 5, 6], [2, 0, 3, 4], [5, 3, 0, 1], [6, 4, 1, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_circular_route(self) -> None:
        """Test shortest path calculation for a circular route."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [[0, 2, jnp.inf, 3], [2, 0, 2, jnp.inf], [jnp.inf, 2, 0, 2], [3, jnp.inf, 2, 0]]
            ),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array([[0, 2, 4, 3], [2, 0, 2, 4], [4, 2, 0, 2], [3, 4, 2, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_disconnected_route(self) -> None:
        """Test shortest path calculation for a route with disconnected segments."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf, jnp.inf],
                    [2, 0, jnp.inf, jnp.inf],
                    [jnp.inf, jnp.inf, 0, 1],
                    [jnp.inf, jnp.inf, 1, 0],
                ]
            ),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array(
            [
                [0, 2, jnp.inf, jnp.inf],
                [2, 0, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, 0, 1],
                [jnp.inf, jnp.inf, 1, 0],
            ]
        )
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_empty_route(self) -> None:
        """Test shortest path calculation for an empty route (no edges)."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array(
            [[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]
        )
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_multiple_paths_between_stops(self) -> None:
        """Test that the shortest path is chosen when multiple paths exist between stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array([[0, 2, 5, jnp.inf], [2, 0, 1, 4], [5, 1, 0, 2], [jnp.inf, 4, 2, 0]]),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        # Verify optimal path lengths between various stops
        assert float(shortest_paths[0, 2]) == 3.0  # 0->1->2 is shorter than 0->2
        assert float(shortest_paths[2, 0]) == 3.0  # 2->1->0 is shorter than 2->0
        assert float(shortest_paths[1, 3]) == 3.0  # 1->2->3 is shorter than 1->3
        assert float(shortest_paths[3, 1]) == 3.0  # 3->2->1 is shorter than 3->1

    def test_asymmetric_weights(self) -> None:
        """Test shortest path calculation with asymmetric weights between stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [3, 0, 1], [jnp.inf, 2, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array([[0, 2, 3], [3, 0, 1], [5, 2, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_zero_weight_edges(self) -> None:
        """Test shortest path calculation with zero-weight edges."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [
                    [0, 0, jnp.inf, jnp.inf],
                    [0, 0, 0, jnp.inf],
                    [jnp.inf, 0, 0, 0],
                    [jnp.inf, jnp.inf, 0, 0],
                ]
            ),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        route = jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        expected_paths = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_self_loops(self) -> None:
        """Test shortest path calculation with self-loops in the route."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[1, 2, jnp.inf], [2, 1, 3], [jnp.inf, 3, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        route = jnp.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])

        shortest_paths = Mandl._get_route_shortest_paths(network, route)

        # Verify self-loops don't affect shortest paths
        assert float(shortest_paths[0, 1]) == 2.0
        assert float(shortest_paths[1, 2]) == 3.0


class TestFindDirectPaths:
    @pytest.fixture
    def mandl_env(self) -> Mandl:
        return Mandl()

    def test_single_direct_path(self) -> None:
        """Test finding a single direct path between two nodes."""
        route_paths = [jnp.array([[0, 5, jnp.inf], [5, 0, 3], [jnp.inf, 3, 0]])]

        paths = Mandl._find_direct_paths(route_paths, start=0, end=1)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=0, cost=5.0, start=0, end=1)

    def test_no_direct_paths(self) -> None:
        """Test when no direct paths exist between nodes."""
        route_paths = [
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]])
        ]

        paths = Mandl._find_direct_paths(route_paths, start=0, end=1)

        assert len(paths) == 0

    def test_multiple_paths_sorted_by_cost(self) -> None:
        """Test that multiple direct paths are found and sorted by cost."""
        route_paths = [
            jnp.array([[0, 5, jnp.inf], [5, 0, 3], [jnp.inf, 3, 0]]),
            jnp.array([[0, 3, jnp.inf], [3, 0, 4], [jnp.inf, 4, 0]]),
        ]

        paths = Mandl._find_direct_paths(route_paths, start=0, end=1)

        assert len(paths) == 2
        assert paths[0] == DirectPath(route=1, cost=3.0, start=0, end=1)
        assert paths[1] == DirectPath(route=0, cost=5.0, start=0, end=1)

    def test_single_valid_path_among_multiple_routes(self) -> None:
        """Test finding single valid path when multiple routes exist but one has a valid path."""
        route_paths = [
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
        ]

        paths = Mandl._find_direct_paths(route_paths, start=0, end=1)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=1, cost=2.0, start=0, end=1)

    def test_reverse_direction(self) -> None:
        """Test finding direct paths in reverse direction."""
        route_paths = [jnp.array([[0, 5, jnp.inf], [5, 0, 3], [jnp.inf, 3, 0]])]

        paths = Mandl._find_direct_paths(route_paths, start=1, end=0)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=0, cost=5.0, start=1, end=0)


class TestFindTransferPaths:
    def test_single_transfer_path(self) -> None:
        """Test finding a single transfer path between two nodes."""
        route_paths = [
            jnp.array(
                [  # Route 0
                    [0, 2, jnp.inf],
                    [2, 0, jnp.inf],
                    [jnp.inf, jnp.inf, 0],
                ]
            ),
            jnp.array(
                [  # Route 1
                    [0, jnp.inf, jnp.inf],
                    [jnp.inf, 0, 3],
                    [jnp.inf, 3, 0],
                ]
            ),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=0, end=2, transfer_penalty=2.0)

        assert len(paths) == 1
        assert paths[0] == TransferPath(
            first_route=0,
            second_route=1,
            total_cost=7.0,  # 2 + 3 + 2 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )

    def test_no_transfer_paths(self) -> None:
        """Test when no transfer paths exist between nodes."""
        route_paths = [
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=0, end=2, transfer_penalty=2.0)

        assert len(paths) == 0

    def test_multiple_transfer_paths_sorted(self) -> None:
        """Test that multiple transfer paths are found and sorted by total cost."""
        route_paths = [
            jnp.array(
                [  # Route 0
                    [0, 2, jnp.inf],
                    [2, 0, jnp.inf],
                    [jnp.inf, jnp.inf, 0],
                ]
            ),
            jnp.array(
                [  # Route 1
                    [0, 1, jnp.inf],
                    [1, 0, 3],
                    [jnp.inf, 3, 0],
                ]
            ),
            jnp.array(
                [  # Route 2
                    [0, 3, jnp.inf],
                    [3, 0, 2],
                    [jnp.inf, 2, 0],
                ]
            ),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=0, end=2, transfer_penalty=2.0)

        assert len(paths) == 4
        # Verify paths are sorted by total cost
        assert paths[0].total_cost == 5.0  # 1 + 2 + 2 (transfer penalty)
        assert paths[1].total_cost == 6.0  # 2 + 2 + 2 (transfer penalty)
        assert paths[2].total_cost == 7.0  # 2 + 3 + 2 (transfer penalty)
        assert paths[3].total_cost == 8.0  # 3 + 3 + 2 (transfer penalty)

    def test_different_transfer_penalties(self) -> None:
        """Test that different transfer penalties correctly affect the total cost."""
        route_paths = [
            jnp.array(
                [  # Route 0
                    [0, 2, jnp.inf],
                    [2, 0, jnp.inf],
                    [jnp.inf, jnp.inf, 0],
                ]
            ),
            jnp.array(
                [  # Route 1
                    [0, jnp.inf, jnp.inf],
                    [jnp.inf, 0, 3],
                    [jnp.inf, 3, 0],
                ]
            ),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=0, end=2, transfer_penalty=4.0)

        assert len(paths) == 1
        assert paths[0].total_cost == 9.0  # 2 + 3 + 4 (transfer penalty)

    def test_multiple_transfer_stops(self) -> None:
        """Test finding paths with multiple possible transfer stops."""
        route_paths = [
            jnp.array(
                [  # Route 0
                    [0, 2, 4],
                    [2, 0, 2],
                    [4, 2, 0],
                ]
            ),
            jnp.array(
                [  # Route 1
                    [0, 1, 3],
                    [1, 0, 1],
                    [3, 1, 0],
                ]
            ),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=0, end=2, transfer_penalty=2.0)

        assert len(paths) == 2
        assert paths[0].total_cost == 5.0  # 2 + 1 + 2 (transfer penalty)
        assert paths[1].total_cost == 5.0  # 1 + 2 + 2 (transfer penalty)

    def test_reverse_direction(self) -> None:
        """Test finding transfer paths in reverse direction."""
        route_paths = [
            jnp.array(
                [  # Route 0
                    [0, 2, jnp.inf],
                    [2, 0, 3],
                    [jnp.inf, 3, 0],
                ]
            ),
            jnp.array(
                [  # Route 1
                    [0, 3, jnp.inf],
                    [3, 0, 2],
                    [jnp.inf, 2, 0],
                ]
            ),
        ]

        paths = Mandl._find_transfer_paths(route_paths, start=2, end=0, transfer_penalty=2.0)

        assert len(paths) == 2
        assert paths[0].total_cost == 6.0  # 2 + 2 + 2 (transfer penalty)
        assert paths[1].total_cost == 8.0  # 3 + 3 + 2 (transfer penalty)


class TestFindPaths:
    def test_direct_path_exists(self, mandl_env: Mandl) -> None:
        """Test that when a direct path exists, it is found and preferred over transfer paths."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        routes = jnp.array(
            [
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            ]
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 1
        assert len(transfer_paths) == 0
        assert direct_paths[0] == DirectPath(route=0, cost=5.0, start=0, end=2)

    def test_only_transfer_path_exists(self, mandl_env: Mandl) -> None:
        """Test that when only a transfer path exists, it is found correctly."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        routes = jnp.array(
            [
                [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            ]
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 0
        assert len(transfer_paths) == 1
        assert transfer_paths[0] == TransferPath(
            first_route=0,
            second_route=1,
            total_cost=7.0,  # 2 + 3 + 2 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )

    def test_no_paths_exist(self, mandl_env: Mandl) -> None:
        """Test that when no paths exist (direct or transfer), empty lists are returned."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        routes = jnp.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 0
        assert len(transfer_paths) == 0

    def test_direct_path_preferred_over_transfers(self, mandl_env: Mandl) -> None:
        """Test that when both direct and transfer paths exist, direct path is preferred."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [[0, 2, jnp.inf, 5], [2, 0, 3, jnp.inf], [jnp.inf, 3, 0, 2], [5, jnp.inf, 2, 0]]
            ),
            demand=jnp.zeros((4, 4)),
            terminals=jnp.array([False, False, False, False]),
            icon_path="",
        )
        routes = jnp.array(
            [
                [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]],
            ]
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=3, transfer_penalty=2.0
        )

        assert len(direct_paths) == 1
        assert len(transfer_paths) == 0
        assert direct_paths[0] == DirectPath(route=0, cost=5.0, start=0, end=3)

    def test_transfer_penalty_affects_cost(self, mandl_env: Mandl) -> None:
        """Test that different transfer penalties correctly affect the total path cost."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            demand=jnp.zeros((3, 3)),
            terminals=jnp.array([False, False, False]),
            icon_path="",
        )
        routes = jnp.array(
            [
                [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            ]
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network,
            routes=routes,
            start=0,
            end=2,
            transfer_penalty=5.0,  # Higher penalty
        )

        assert len(direct_paths) == 0
        assert len(transfer_paths) == 1
        assert transfer_paths[0] == TransferPath(
            first_route=0,
            second_route=1,
            total_cost=10.0,  # 2 + 3 + 5 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )
