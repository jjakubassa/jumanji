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

# Copyright 2022 InstaDeep Ltd. All rights reerved.
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
    RouteBatch,
    State,
    TransferPath,
)


@pytest.fixture
def mandl_env() -> Mandl:
    return Mandl()


class TestMandlEnv:
    def test_env_initialization(self) -> None:
        """Test environment initialization and properties."""
        env = Mandl(num_flex_routes=2)
        assert env.num_nodes == 15
        assert env.num_flex_routes == 2
        assert env.max_capacity == 40
        assert env.simulation_steps == 60

    def test_reset(self) -> None:
        """Test environment reset."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, timestep = env.reset(key)

        # Check state components
        assert isinstance(state, State)
        assert state.current_time == 0

        # Check routes initialization
        fixed_routes = sum(~state.routes.on_demand)  # Count non-flexible routes
        flex_routes = sum(state.routes.on_demand)  # Count flexible routes
        assert fixed_routes == 4  # From solution file
        assert flex_routes == env.num_flex_routes

        # Check vehicles initialization
        total_routes = len(state.routes.ids)
        assert len(state.vehicles.ids) == total_routes
        assert jnp.all(state.vehicles.capacities == env.max_capacity)

    def test_action_space(self) -> None:
        """Test action space properties."""
        env = Mandl(num_flex_routes=2)
        assert env.action_spec.num_values == env.num_nodes + 1  # Includes wait action
        assert env.action_spec.minimum == 0
        assert env.action_spec.maximum == env.num_nodes

    def test_step_wait_action(self) -> None:
        """Test step function with wait action."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Create wait action for all flexible routes
        action = jnp.full((env.num_flex_routes,), env.num_nodes)  # All wait

        new_state, _ = env.step(state, action)

        # Check that flexible routes haven't changed
        flex_routes_mask = state.routes.on_demand
        assert jnp.array_equal(
            state.routes.nodes[flex_routes_mask], new_state.routes.nodes[flex_routes_mask]
        )

        # Check time increment
        assert new_state.current_time == state.current_time + 1

    def test_step_valid_node_action(self) -> None:
        """Test step function with valid node addition."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Add node 0 to first flexible route
        action = jnp.array([0, env.num_nodes])  # First route: add node 0, Second route: wait
        new_state, _ = env.step(state, action)

        # Check that first flexible route has node 0
        flex_routes_mask = state.routes.on_demand
        flex_route_idx = jnp.where(flex_routes_mask)[0][0]
        assert new_state.routes.nodes[flex_route_idx][0] == 0

    def test_invalid_action_shape(self) -> None:
        """Test error handling for invalid action shape."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Wrong shape action
        invalid_action = jnp.array([0])  # Only one action when we need two

        with pytest.raises(ValueError, match=r"Action must have shape"):
            env.step(state, invalid_action)

    def test_passenger_status_updates(self) -> None:
        """Test passenger status updates."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Get initial passenger counts by status
        initial_waiting = sum(state.passengers.statuses == 1)

        # Step with wait action
        action = jnp.full((env.num_flex_routes,), env.num_nodes)
        new_state, _ = env.step(state, action)

        # Check that some passengers might have changed status
        new_waiting = sum(new_state.passengers.statuses == 1)
        assert new_waiting >= initial_waiting  # More or same number of waiting passengers

    def test_vehicle_movement(self) -> None:
        """Test vehicle movement along routes."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Record initial vehicle positions
        initial_positions = state.vehicles.current_edges.copy()

        # Step with wait action
        action = jnp.full((env.num_flex_routes,), env.num_nodes)
        new_state, _ = env.step(state, action)

        # Check that some vehicles have moved
        assert not jnp.array_equal(initial_positions, new_state.vehicles.current_edges)


class TestGetRouteShortestPaths:
    def test_single_edge_route(self) -> None:
        """Test shortest path calculation for a route with a single edge."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, -1, -1]]),  # Single edge 0->1
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

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
            terminals=jnp.array([False, False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, 3]]),  # Linear route 0->1->2->3
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, 5, 6], [2, 0, 3, 4], [5, 3, 0, 1], [6, 4, 1, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    @pytest.mark.skip()
    def test_circular_route(self) -> None:
        """Test shortest path calculation for a circular route."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [[0, 2, jnp.inf, 3], [2, 0, 2, jnp.inf], [jnp.inf, 2, 0, 2], [3, jnp.inf, 2, 0]]
            ),
            terminals=jnp.array([False, False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, 3, 0, 1]]),  # Circular route 0->1->2->3->0->1
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, 4, 6], [2, 0, 2, 4], [4, 2, 0, 2], [6, 4, 2, 0]])
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
            terminals=jnp.array([False, False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, -1, -1], [2, 3, -1, -1]]),  # Two disconnected segments
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

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
            terminals=jnp.array([False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[-1, -1, -1]]),  # Empty route
            frequencies=jnp.array([0]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array(
            [[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]
        )
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_multiple_paths_between_stops(self) -> None:
        """Test that the shortest path is chosen when multiple paths exist between stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array([[0, 2, 5, jnp.inf], [2, 0, 1, 4], [5, 1, 0, 2], [jnp.inf, 4, 2, 0]]),
            terminals=jnp.array([False, False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, 3]]),  # Route with multiple possible paths
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

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
            terminals=jnp.array([False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2]]),  # Route with asymmetric weights
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, 3], [3, 0, 1], [5, 2, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    def test_self_loops(self) -> None:
        """Test shortest path calculation with self-loops in the route."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        route = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 1, 2]]),  # Route with self-loop at node 1
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = Mandl._get_route_shortest_paths(network, route.nodes)

        # Verify self-loops don't affect shortest paths
        assert float(shortest_paths[0, 1]) == 2.0  # Direct path 0->1
        assert float(shortest_paths[1, 2]) == 3.0  # Direct path 1->2

    def test_multiple_paths_sorted_between_all_pairs(self, mandl_env: Mandl) -> None:
        """Test shortest paths calculation for routes with direct and indirect paths
        between nodes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 3, 5], [3, 0, 5], [5, 5, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 2, -1, -1],  # Route 0: 0->2 direct (cost 5)
                    [0, 1, 2, -1],  # Route 1: 0->1->2 (cost 8 = 3+5)
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        shortest_paths = mandl_env._get_all_shortest_paths(network=network, routes=routes)

        # Check dimensions
        assert shortest_paths.shape == (2, 3, 3)  # (num_routes, num_nodes, num_nodes)

        # Expected shortest paths for direct route (0->2)
        expected_direct = jnp.array([[0, jnp.inf, 5], [jnp.inf, 0, jnp.inf], [5, jnp.inf, 0]])

        # Expected shortest paths for indirect route (0->1->2)
        expected_indirect = jnp.array([[0, 3, 8], [3, 0, 5], [8, 5, 0]])

        # Check both routes have correct shortest paths
        assert jnp.allclose(shortest_paths[0], expected_direct, equal_nan=True)
        assert jnp.allclose(shortest_paths[1], expected_indirect, equal_nan=True)

        # Verify specific path costs
        assert float(shortest_paths[0, 0, 2]) == 5.0  # Route 0: direct path 0->2
        assert float(shortest_paths[1, 0, 2]) == 8.0  # Route 1: indirect path 0->1->2 (3+5)


class TestGetAllShortestPaths:
    def test_single_route(self, mandl_env: Mandl) -> None:
        """Test shortest paths calculation for a single route."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, -1]]),  # Route 0->1->2
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )
        shortest_paths = mandl_env._get_all_shortest_paths(network=network, routes=routes)

        assert shortest_paths.shape == (1, 3, 3)  # (num_routes, num_nodes, num_nodes)
        assert jnp.allclose(shortest_paths[0], jnp.array([[0, 2, 5], [2, 0, 3], [5, 3, 0]]))

    def test_multiple_routes(self, mandl_env: Mandl) -> None:
        """Test shortest paths calculation for multiple routes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0->1
                    [1, 2, -1, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        shortest_paths = mandl_env._get_all_shortest_paths(network=network, routes=routes)

        assert shortest_paths.shape == (2, 3, 3)
        # First route shortest paths
        assert jnp.allclose(
            shortest_paths[0],
            jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            equal_nan=True,
        )
        # Second route shortest paths
        assert jnp.allclose(
            shortest_paths[1],
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, 3], [jnp.inf, 3, 0]]),
            equal_nan=True,
        )

    def test_empty_routes(self, mandl_env: Mandl) -> None:
        """Test shortest paths calculation for empty routes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[-1, -1, -1]]),  # Empty route
            frequencies=jnp.array([0]),
            on_demand=jnp.array([False]),
        )

        shortest_paths = mandl_env._get_all_shortest_paths(network=network, routes=routes)

        assert shortest_paths.shape == (1, 3, 3)
        assert jnp.allclose(
            shortest_paths[0],
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            equal_nan=True,
        )

    def test_caching(self, mandl_env: Mandl) -> None:
        """Test that results are cached and reused."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, -1]]),
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        # First call should compute and cache
        first_result = mandl_env._get_all_shortest_paths(network=network, routes=routes)
        # Second call should use cached result
        second_result = mandl_env._get_all_shortest_paths(network=network, routes=routes)

        assert jnp.array_equal(first_result, second_result)
        assert len(mandl_env._shortest_paths_cache) == 1

    def test_long_routes_both_directions(self, mandl_env: Mandl) -> None:
        """Test shortest path calculation for longer routes with symmetric weights."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf, jnp.inf, jnp.inf],
                    [2, 0, 3, jnp.inf, jnp.inf],
                    [jnp.inf, 3, 0, 1, jnp.inf],
                    [jnp.inf, jnp.inf, 1, 0, 4],
                    [jnp.inf, jnp.inf, jnp.inf, 4, 0],
                ]
            ),
            terminals=jnp.array([False, False, False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, 2, 3, 4],  # Route 0: 0->1->2->3->4
                    [4, 3, 2, 1, 0],  # Route 1: 4->3->2->1->0 (reverse direction)
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        shortest_paths = mandl_env._get_route_shortest_paths(network, routes.nodes)

        # Forward direction (route 0)
        assert float(shortest_paths[0, 4]) == 10.0  # 0->1->2->3->4 = 2+3+1+4 = 10
        assert float(shortest_paths[0, 2]) == 5.0  # 0->1->2 = 2+3 = 5
        assert float(shortest_paths[1, 3]) == 4.0  # 1->2->3 = 3+1 = 4

        # Reverse direction (route 1)
        assert float(shortest_paths[4, 0]) == 10.0  # 4->3->2->1->0 = 4+1+3+2 = 10
        assert float(shortest_paths[4, 2]) == 5.0  # 4->3->2 = 4+1 = 5
        assert float(shortest_paths[3, 1]) == 4.0  # 3->2->1 = 1+3 = 4

    def test_multiple_paths_between_stops(self, mandl_env: Mandl) -> None:
        """Test shortest path calculation with multiple possible routes between stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf, jnp.inf, 7],
                    [2, 0, 3, jnp.inf, jnp.inf],
                    [jnp.inf, 3, 0, 1, jnp.inf],
                    [jnp.inf, jnp.inf, 1, 0, 4],
                    [7, jnp.inf, jnp.inf, 4, 0],
                ]
            ),
            terminals=jnp.array([False, False, False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, 2, 3, 4],  # Route 0: 0->1->2->3->4 (longer path)
                    [0, 4, -1, -1, -1],  # Route 1: 0->4 (direct path)
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        shortest_paths = mandl_env._get_route_shortest_paths(network, routes.nodes)

        # Verify optimal path lengths for different routes
        assert float(shortest_paths[0, 4]) == 7.0  # Direct path 0->4 = 7
        assert float(shortest_paths[4, 0]) == 7.0  # Direct path 4->0 = 7
        assert float(shortest_paths[1, 4]) == 8.0  # Route 0: 1->2->3->4 = 3+1+4 = 8
        assert float(shortest_paths[4, 1]) == 8.0  # Route 0: 4->3->2->1 = 4+1+3 = 8


class TestFindDirectPaths:
    @pytest.fixture
    def mandl_env(self) -> Mandl:
        return Mandl()

    def test_single_direct_path(self, mandl_env: Mandl) -> None:
        """Test finding a single direct path between two nodes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, -1]]),  # Route 0->1->2
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=1)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=0, cost=2.0, start=0, end=1)

    def test_no_direct_paths(self, mandl_env: Mandl) -> None:
        """Test when no direct paths exist between nodes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[-1, -1, -1, -1]]),  # Empty route
            frequencies=jnp.array([0]),
            on_demand=jnp.array([False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=1)

        assert len(paths) == 0

    def test_multiple_paths_sorted_by_cost(self, mandl_env: Mandl) -> None:
        """Test that multiple direct paths are found and sorted by cost."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 3, 5], [3, 0, 5], [5, 5, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 2, -1, -1],  # Route 0: 0->2 direct (cost 5)
                    [0, 1, 2, -1],  # Route 1: 0->1->2 (cost 8)
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=2)

        assert len(paths) == 2
        assert paths[0] == DirectPath(route=0, cost=5.0, start=0, end=2)
        assert paths[1] == DirectPath(route=1, cost=8.0, start=0, end=2)

    def test_single_valid_path_among_multiple_routes(self, mandl_env: Mandl) -> None:
        """Test finding single valid path when multiple routes exist but one has a valid path."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [-1, -1, -1, -1],  # Route 0: empty
                    [0, 1, -1, -1],  # Route 1: 0->1
                ]
            ),
            frequencies=jnp.array([0, 1]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=1)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=1, cost=2.0, start=0, end=1)

    def test_reverse_direction(self, mandl_env: Mandl) -> None:
        """Test finding direct paths in reverse direction."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 2, -1]]),  # Route 0->1->2
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=1, end=0)

        assert len(paths) == 1
        assert paths[0] == DirectPath(route=0, cost=2.0, start=1, end=0)

    def test_path_with_transfer_not_found(self, mandl_env: Mandl) -> None:
        """Test that paths requiring transfers are not found as direct paths."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0->1
                    [1, 2, -1, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=2)

        assert len(paths) == 0  # No direct path exists, would need transfer at node 1

    @pytest.mark.skip()
    def test_paths_with_loops(self, mandl_env: Mandl) -> None:
        """Test finding direct paths in routes with loops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[0, 1, 0, 1, 2]]),  # Route with loop: 0->1->0->1->2
            frequencies=jnp.array([1]),
            on_demand=jnp.array([False]),
        )

        paths = mandl_env._find_direct_paths(routes=routes, network=network, start=0, end=2)

        assert len(paths) == 1
        assert paths[0] == DirectPath(
            route=0, cost=8.0, start=0, end=2
        )  # Should find shortest path despite loop


class TestFindTransferPaths:
    def test_single_transfer_path(self, mandl_env: Mandl) -> None:
        """Test finding a single transfer path between two nodes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf],  # 0->1: 2, 0->2: inf
                    [2, 0, 3],  # 1->0: 2, 1->2: 3
                    [jnp.inf, 3, 0],  # 2->0: inf, 2->1: 3
                ]
            ),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1],  # Route 0: 0->1
                    [1, 2, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_transfer_paths(
            routes=routes, network=network, start=0, end=2, transfer_penalty=2.0
        )

        assert len(paths) == 1
        assert paths[0] == TransferPath(
            first_route=0,
            second_route=1,
            total_cost=7.0,  # 2 (0->1) + 3 (1->2) + 2 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )

    def test_no_transfer_paths(self, mandl_env: Mandl) -> None:
        """Test when no transfer paths exist between nodes."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [-1, -1, -1],  # Empty route
                    [-1, -1, -1],  # Empty route
                ]
            ),
            frequencies=jnp.array([0, 0]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )
        assert len(paths) == 0

    def test_multiple_transfer_paths_sorted(self, mandl_env: Mandl) -> None:
        """Test that multiple transfer paths are found and sorted by total cost."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [[0, 2, jnp.inf, jnp.inf], [2, 0, 3, 4], [jnp.inf, 3, 0, 2], [jnp.inf, 4, 2, 0]]
            ),
            terminals=jnp.array([False, False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1, 2]),
            nodes=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0->1
                    [1, 2, -1, -1],  # Route 1: 1->2
                    [1, 3, 2, -1],  # Route 2: 1->3->2 (alternative path)
                ]
            ),
            frequencies=jnp.array([1, 1, 1]),
            on_demand=jnp.array([False, False, False]),
        )

        paths = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(paths) == 2
        # Path 1: 0->1 (2) + 1->2 (3) + transfer (2) = 7
        assert paths[0].total_cost == 7.0
        assert paths[0].transfer_stop == 1
        assert paths[0].first_route == 0
        assert paths[0].second_route == 1

        # Path 2: 0->1 (2) + 1->3->2 (6) + transfer (2) = 10
        assert paths[1].total_cost == 10.0
        assert paths[1].transfer_stop == 1
        assert paths[1].first_route == 0
        assert paths[1].second_route == 2

    def test_different_transfer_penalties(self, mandl_env: Mandl) -> None:
        """Test that different transfer penalties correctly affect the total cost."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1],  # Route 0: 0->1
                    [1, 2, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        # Test with higher transfer penalty
        paths_high = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=4.0
        )

        assert len(paths_high) == 1
        assert paths_high[0].total_cost == 9.0  # 2 + 3 + 4 (higher penalty)

        # Test with lower transfer penalty
        paths_low = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=1.0
        )

        assert len(paths_low) == 1
        assert paths_low[0].total_cost == 6.0  # 2 + 3 + 1 (lower penalty)

    def test_multiple_transfer_stops(self, mandl_env: Mandl) -> None:
        """Test finding paths with multiple possible transfer stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array([[0, 2, jnp.inf, 5], [2, 0, 3, 4], [jnp.inf, 3, 0, 2], [5, 4, 2, 0]]),
            terminals=jnp.array([False, False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, 3, -1],  # Route 0: 0->1->3
                    [1, 2, 3, -1],  # Route 1: 1->2->3
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        paths = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(paths) == 2
        # Transfer at node 1
        assert any(p.transfer_stop == 1 and p.total_cost == 7.0 for p in paths)  # 2 + 3 + 2
        # Transfer at node 3
        assert any(p.transfer_stop == 3 and p.total_cost == 10.0 for p in paths)  # 6 + 2 + 2

    def test_reverse_direction(self, mandl_env: Mandl) -> None:
        """Test transfer paths work in reverse direction."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )
        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1],  # Route 0: 0->1
                    [1, 2, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        # Test forward direction (0->2)
        paths_forward = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )
        assert len(paths_forward) == 1
        assert paths_forward[0].total_cost == 7.0  # 2 + 3 + 2

        # Test reverse direction (2->0)
        paths_reverse = mandl_env._find_transfer_paths(
            network=network, routes=routes, start=2, end=0, transfer_penalty=2.0
        )
        assert len(paths_reverse) == 1
        assert paths_reverse[0].total_cost == 7.0  # 3 + 2 + 2


class TestFindPaths:
    def test_direct_path_exists(self, mandl_env: Mandl) -> None:
        """Test that when a direct path exists, it is found and preferred over transfer paths."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )

        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, 2, -1],  # Route 0: complete path 0->1->2
                    [-1, -1, -1, -1],  # Route 1: empty
                ]
            ),
            frequencies=jnp.array([2, 0]),  # Two vehicles on route 0, none on route 1
            on_demand=jnp.array([False, False]),
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 1
        assert direct_paths[0] == DirectPath(
            route=0,
            cost=5.0,  # 2 (0->1) + 3 (1->2)
            start=0,
            end=2,
        )
        assert len(transfer_paths) == 0  # No transfer paths needed

    def test_only_transfer_path_exists(self, mandl_env: Mandl) -> None:
        """Test that when only a transfer path exists, it is found correctly."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )

        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0->1
                    [1, 2, -1, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 0
        assert len(transfer_paths) == 1
        assert transfer_paths[0] == TransferPath(
            first_route=0,
            second_route=1,
            total_cost=7.0,  # 2 (0->1) + 3 (1->2) + 2 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )

    def test_no_paths_exist(self, mandl_env: Mandl) -> None:
        """Test that when no paths exist (direct or transfer), empty lists are returned."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )

        routes = RouteBatch(
            ids=jnp.array([0]),
            nodes=jnp.array([[-1, -1, -1, -1]]),  # Empty route
            frequencies=jnp.array([0]),
            on_demand=jnp.array([False]),
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert len(direct_paths) == 0
        assert len(transfer_paths) == 0

    def test_direct_path_preferred_over_transfers(self, mandl_env: Mandl) -> None:
        """Test that when a direct path exists, transfer paths are not even calculated."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            links=jnp.array(
                [[0, 2, jnp.inf, 5], [2, 0, 3, jnp.inf], [jnp.inf, 3, 0, 2], [5, jnp.inf, 2, 0]]
            ),
            terminals=jnp.array([False, False, False, False]),
        )

        routes = RouteBatch(
            ids=jnp.array([0, 1, 2]),
            nodes=jnp.array(
                [
                    [0, 3, -1, -1],  # Route 0: direct path 0->3
                    [0, 1, -1, -1],  # Route 1: potential transfer path part 1
                    [1, 2, 3, -1],  # Route 2: potential transfer path part 2
                ]
            ),
            frequencies=jnp.array([1, 1, 1]),
            on_demand=jnp.array([False, False, False]),
        )

        direct_paths, transfer_paths = mandl_env._find_paths(
            network=network, routes=routes, start=0, end=3, transfer_penalty=2.0
        )

        # Direct path exists
        assert len(direct_paths) == 1
        assert direct_paths[0] == DirectPath(route=0, cost=5.0, start=0, end=3)

        # Transfer paths should be empty since a direct path exists
        assert len(transfer_paths) == 0

    def test_transfer_penalty_affects_cost(self, mandl_env: Mandl) -> None:
        """Test that different transfer penalties correctly affect the total path cost."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2]]),
            links=jnp.array([[0, 2, jnp.inf], [2, 0, 3], [jnp.inf, 3, 0]]),
            terminals=jnp.array([False, False, False]),
        )

        routes = RouteBatch(
            ids=jnp.array([0, 1]),
            nodes=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0->1
                    [1, 2, -1, -1],  # Route 1: 1->2
                ]
            ),
            frequencies=jnp.array([1, 1]),
            on_demand=jnp.array([False, False]),
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
            total_cost=10.0,  # 2 (0->1) + 3 (1->2) + 5 (transfer penalty)
            transfer_stop=1,
            start=0,
            end=2,
        )


if __name__ == "__main__":
    pytest.main([__file__])
