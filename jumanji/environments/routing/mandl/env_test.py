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

import chex
import jax
import pytest
from jax import numpy as jnp

from jumanji.environments.routing.mandl.env import Mandl
from jumanji.environments.routing.mandl.types import (
    NetworkData,
    PassengerBatch,
    RouteBatch,
    State,
)


@pytest.fixture()
def mandl_env() -> Mandl:
    return Mandl()


class TestMandlEnv(chex.TestCase):
    def test_env_initialization(self) -> None:
        """Test environment initialization and properties."""
        env = Mandl(num_flex_routes=2)
        assert env.num_nodes == 15
        assert env.num_flex_routes == 2
        assert env.max_capacity == 40
        assert env.simulation_steps == 60

    @chex.variants(with_jit=True, without_jit=True)
    def test_reset(self) -> None:
        """Test environment reset."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, timestep = reset(key)

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
        assert env.action_spec.minimum == -1  # Allow wait action (-1)
        assert env.action_spec.maximum == env.num_nodes - 1  # Node indices from 0 to num_nodes-1

    @chex.variants(with_jit=True, without_jit=True)
    def test_step_wait_action(self) -> None:
        """Test step function with wait action."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Create wait action for all flexible routes
        action = jnp.full((env.num_flex_routes,), -1)  # All wait actions are -1 now

        new_state, _ = env.step(state, action)

        # Check that flexible routes haven't changed
        flex_routes_mask = state.routes.on_demand
        assert jnp.array_equal(
            state.routes.nodes[flex_routes_mask], new_state.routes.nodes[flex_routes_mask]
        )

        # Check time increment
        assert new_state.current_time == state.current_time + 1

    @chex.variants(with_jit=True, without_jit=True)
    def test_step_valid_node_action(self) -> None:
        """Test step function with valid node addition."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Add node 0 to first flexible route
        action = jnp.array([0, env.num_nodes])  # First route: add node 0, Second route: wait
        new_state, _ = env.step(state, action)

        # Check that first flexible route has node 0
        flex_routes_mask = state.routes.on_demand
        flex_route_idx = jnp.where(flex_routes_mask)[0][0]
        assert new_state.routes.nodes[flex_route_idx][0] == 0

    @chex.variants(with_jit=True, without_jit=True)
    def test_invalid_action_shape(self) -> None:
        """Test error handling for invalid action shape."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Wrong shape action
        invalid_action = jnp.array([0])  # Only one action when we need two

        step = self.variant(env.step)
        with pytest.raises(ValueError, match=r"Action must have shape"):
            step(state, invalid_action)

    @chex.variants(with_jit=True, without_jit=True)
    def test_passenger_status_transitions(self) -> None:
        """Test that passenger statuses transition correctly through different states.

        Tests four key scenarios:
        1. Status 0 -> 1: Passenger becomes eligible but no vehicle available (should accumulate
        waiting time)
        2. Status 0 -> 1 -> 2: Passenger becomes eligible and gets immediate pickup (no
        waiting time)
        3. Status 2: Passenger in vehicle (should accumulate in-vehicle time)
        4. Status 3: Completed passenger (should maintain status and times)
        """
        env = Mandl(num_flex_routes=2, simulation_steps=10)
        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Set up test passengers
        test_passengers = state.passengers._replace(
            # Four test passengers for different scenarios
            statuses=jnp.array([0, 0, 2, 3] + [0] * (len(state.passengers.statuses) - 4)),
            departure_times=jnp.array(
                [1.0, 1.0, 0.0, 0.0] + [100.0] * (len(state.passengers.departure_times) - 4)
            ),
            origins=jnp.array([0, 1, 2, 3] + [0] * (len(state.passengers.origins) - 4)),
            destinations=jnp.array([1, 2, 3, 4] + [1] * (len(state.passengers.destinations) - 4)),
            time_waiting=jnp.array(
                [0.0, 0.0, 0.0, 2.0] + [0.0] * (len(state.passengers.time_waiting) - 4)
            ),
            time_in_vehicle=jnp.array(
                [0.0, 0.0, 1.0, 3.0] + [0.0] * (len(state.passengers.time_in_vehicle) - 4)
            ),
        )

        # Set up vehicles for testing immediate pickup
        test_vehicles = state.vehicles._replace(
            current_edges=jnp.array([[0, 1], [0, 0]] + [[0, 0]] * (len(state.vehicles.ids) - 2)),
            passengers=jnp.full((len(state.vehicles.ids), env.max_capacity), -1),
        )

        # Set initial state
        state = replace(
            state, passengers=test_passengers, vehicles=test_vehicles, current_time=1
        )  # Set time to when passengers become eligible

        # Take a step
        action = jnp.full((env.num_flex_routes,), -1)  # Wait action
        step = self.variant(env.step)
        new_state, _ = step(state, action)

        # 1. Verify passenger that becomes eligible but no vehicle available
        chex.assert_trees_all_equal(
            new_state.passengers.statuses[0], 1
        )  # Should transition to waiting
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[0], 1.0
        )  # Should accumulate waiting time

        # 2. Verify passenger that gets immediate pickup
        chex.assert_trees_all_equal(
            new_state.passengers.statuses[1], 2
        )  # Should transition to in-vehicle
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[1], 0.0
        )  # Should have no waiting time

        # 3. Verify passenger in vehicle
        chex.assert_trees_all_equal(new_state.passengers.statuses[2], 2)  # Should stay in-vehicle
        chex.assert_trees_all_equal(
            new_state.passengers.time_in_vehicle[2],
            2.0,  # Previous time (1.0) + 1 timestep
        )

        # 4. Verify completed passenger
        chex.assert_trees_all_equal(new_state.passengers.statuses[3], 3)  # Should stay completed
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[3], 2.0
        )  # Should maintain waiting time
        chex.assert_trees_all_equal(
            new_state.passengers.time_in_vehicle[3], 3.0
        )  # Should maintain in-vehicle time

    @chex.variants(with_jit=True, without_jit=True)
    def test_vehicle_movement(self) -> None:
        """Test vehicle movement along routes."""
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Record initial vehicle positions
        initial_positions = state.vehicles.current_edges.copy()

        # Step with wait action
        action = jnp.full((env.num_flex_routes,), env.num_nodes)
        step = self.variant(env.step)
        new_state, _ = step(state, action)

        # Check that some vehicles have moved
        assert not jnp.array_equal(initial_positions, new_state.vehicles.current_edges)

    @chex.variants(with_jit=True, without_jit=True)
    def test_reward_calculation_components(self) -> None:
        """Test reward calculation with three passengers"""
        env = Mandl(num_flex_routes=2, simulation_steps=1, waiting_penalty_factor=2.0)
        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, _ = reset(key)

        test_passengers = PassengerBatch(
            ids=jnp.arange(3),
            statuses=jnp.array([3, 2, 1]),
            origins=jnp.array([0, 1, 5]),
            destinations=jnp.array([1, 2, 6]),
            departure_times=jnp.array([0.0, 0.0, 0.0]),
            time_waiting=jnp.array([1.0, 1.0, 2.0]),
            time_in_vehicle=jnp.array([2.0, 3.0, 0.0]),
        )

        state = replace(state, passengers=test_passengers)

        # Take one step to reach terminal state
        step = self.variant(env.step)
        _, timestep = step(state, jnp.array([-1, -1]))

        # Reward calculation: -(journey_times + waiting_penalty)
        # passenger 0: 2 in vehicle, 1 waiting
        # passenger 1: 4 in vehicle, 1 waiting
        # passenger 2: 0 in vehicle, 3 waiting
        # Journey times: waiting(2.0) + in_vehicle(6.0) = 8.0
        # Waiting penalty: waiting(3.0) * factor(2.0) = 6.0
        # Total = -(6.0 + 8.0) = -14.0
        chex.assert_trees_all_close(timestep.reward, jnp.array(-14.0))


class TestCederEnv(chex.TestCase):
    def test_env_initialization(self) -> None:
        """Test environment initialization and properties."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        assert env.num_nodes == 15
        assert env.num_flex_routes == 2
        assert env.max_capacity == 40
        assert env.simulation_steps == 60

    @chex.variants(with_jit=True, without_jit=True)
    def test_reset(self) -> None:
        """Test environment reset."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, timestep = reset(key)

        # Check state components
        assert isinstance(state, State)
        assert state.current_time == 0

        # Check routes initialization
        fixed_routes = sum(~state.routes.on_demand)  # Count non-flexible routes
        flex_routes = sum(state.routes.on_demand)  # Count flexible routes
        assert fixed_routes == 2  # From solution file
        assert flex_routes == env.num_flex_routes

        # Check vehicles initialization
        total_routes = len(state.routes.ids)
        assert len(state.vehicles.ids) == total_routes
        assert jnp.all(state.vehicles.capacities == env.max_capacity)

    def test_action_space(self) -> None:
        """Test action space properties."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        assert env.action_spec.minimum == -1  # Allow wait action (-1)
        assert env.action_spec.maximum == env.num_nodes - 1  # Node indices from 0 to num_nodes-1

    @chex.variants(with_jit=True, without_jit=True)
    def test_step_wait_action(self) -> None:
        """Test step function with wait action."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Create wait action for all flexible routes
        action = jnp.full((env.num_flex_routes,), -1)  # All wait actions are -1 now

        new_state, _ = env.step(state, action)

        # Check that flexible routes haven't changed
        flex_routes_mask = state.routes.on_demand
        assert jnp.array_equal(
            state.routes.nodes[flex_routes_mask], new_state.routes.nodes[flex_routes_mask]
        )

        # Check time increment
        assert new_state.current_time == state.current_time + 1

    @chex.variants(with_jit=True, without_jit=True)
    def test_step_valid_node_action(self) -> None:
        """Test step function with valid node addition."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Add node 0 to first flexible route
        action = jnp.array([0, env.num_nodes])  # First route: add node 0, Second route: wait
        new_state, _ = env.step(state, action)

        # Check that first flexible route has node 0
        flex_routes_mask = state.routes.on_demand
        flex_route_idx = jnp.where(flex_routes_mask)[0][0]
        assert new_state.routes.nodes[flex_route_idx][0] == 0

    @chex.variants(with_jit=True, without_jit=True)
    def test_invalid_action_shape(self) -> None:
        """Test error handling for invalid action shape."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Wrong shape action
        invalid_action = jnp.array([0])  # Only one action when we need two

        step = self.variant(env.step)
        with pytest.raises(ValueError, match=r"Action must have shape"):
            step(state, invalid_action)

    @chex.variants(with_jit=True, without_jit=True)
    def test_passenger_status_transitions(self) -> None:
        """Test that passenger statuses transition correctly through different states.

        Tests four key scenarios:
        1. Status 0 -> 1: Passenger becomes eligible but no vehicle available (should accumulate
        waiting time)
        2. Status 0 -> 1 -> 2: Passenger becomes eligible and gets immediate pickup (no
        waiting time)
        3. Status 2: Passenger in vehicle (should accumulate in-vehicle time)
        4. Status 3: Completed passenger (should maintain status and times)
        """
        env = Mandl(network_name="ceder1", num_flex_routes=2, simulation_steps=10)
        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Set up test passengers
        test_passengers = state.passengers._replace(
            # Four test passengers for different scenarios
            statuses=jnp.array([0, 0, 2, 3] + [0] * (len(state.passengers.statuses) - 4)),
            departure_times=jnp.array(
                [1.0, 1.0, 0.0, 0.0] + [100.0] * (len(state.passengers.departure_times) - 4)
            ),
            origins=jnp.array([0, 1, 2, 3] + [0] * (len(state.passengers.origins) - 4)),
            destinations=jnp.array([1, 2, 3, 4] + [1] * (len(state.passengers.destinations) - 4)),
            time_waiting=jnp.array(
                [0.0, 0.0, 0.0, 2.0] + [0.0] * (len(state.passengers.time_waiting) - 4)
            ),
            time_in_vehicle=jnp.array(
                [0.0, 0.0, 1.0, 3.0] + [0.0] * (len(state.passengers.time_in_vehicle) - 4)
            ),
        )

        # Set up vehicles for testing immediate pickup
        test_vehicles = state.vehicles._replace(
            current_edges=jnp.array([[0, 1], [0, 0]] + [[0, 0]] * (len(state.vehicles.ids) - 2)),
            passengers=jnp.full((len(state.vehicles.ids), env.max_capacity), -1),
        )

        # Set initial state
        state = replace(
            state, passengers=test_passengers, vehicles=test_vehicles, current_time=1
        )  # Set time to when passengers become eligible

        # Take a step
        action = jnp.full((env.num_flex_routes,), -1)  # Wait action
        step = self.variant(env.step)
        new_state, _ = step(state, action)

        # 1. Verify passenger that becomes eligible but no vehicle available
        chex.assert_trees_all_equal(
            new_state.passengers.statuses[0], 1
        )  # Should transition to waiting
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[0], 1.0
        )  # Should accumulate waiting time

        # 2. Verify passenger that gets immediate pickup
        chex.assert_trees_all_equal(
            new_state.passengers.statuses[1], 2
        )  # Should transition to in-vehicle
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[1], 0.0
        )  # Should have no waiting time

        # 3. Verify passenger in vehicle
        chex.assert_trees_all_equal(new_state.passengers.statuses[2], 2)  # Should stay in-vehicle
        chex.assert_trees_all_equal(
            new_state.passengers.time_in_vehicle[2],
            2.0,  # Previous time (1.0) + 1 timestep
        )

        # 4. Verify completed passenger
        chex.assert_trees_all_equal(new_state.passengers.statuses[3], 3)  # Should stay completed
        chex.assert_trees_all_equal(
            new_state.passengers.time_waiting[3], 2.0
        )  # Should maintain waiting time
        chex.assert_trees_all_equal(
            new_state.passengers.time_in_vehicle[3], 3.0
        )  # Should maintain in-vehicle time

    @chex.variants(with_jit=True, without_jit=True)
    def test_vehicle_movement(self) -> None:
        """Test vehicle movement along routes."""
        env = Mandl(network_name="ceder1", num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        reset = self.variant(env.reset)
        state, _ = reset(key)

        # Record initial vehicle positions
        initial_positions = state.vehicles.current_edges.copy()

        # Step with wait action
        action = jnp.full((env.num_flex_routes,), env.num_nodes)
        step = self.variant(env.step)
        new_state, _ = step(state, action)

        # Check that some vehicles have moved
        assert not jnp.array_equal(initial_positions, new_state.vehicles.current_edges)

    @chex.variants(with_jit=True, without_jit=True)
    def test_reward_calculation_components(self) -> None:
        """Test reward calculation with three passengers"""
        env = Mandl(
            network_name="ceder1", num_flex_routes=2, simulation_steps=1, waiting_penalty_factor=2.0
        )

        key = jax.random.PRNGKey(0)

        reset = self.variant(env.reset)
        state, _ = reset(key)

        test_passengers = PassengerBatch(
            ids=jnp.arange(3),
            statuses=jnp.array([3, 2, 1]),
            origins=jnp.array([0, 1, 5]),
            destinations=jnp.array([1, 2, 6]),
            departure_times=jnp.array([0.0, 0.0, 0.0]),
            time_waiting=jnp.array([1.0, 1.0, 2.0]),
            time_in_vehicle=jnp.array([2.0, 3.0, 0.0]),
        )

        state = replace(state, passengers=test_passengers)

        # Take one step to reach terminal state
        step = self.variant(env.step)
        _, timestep = step(state, jnp.array([-1, -1]))

        # Reward calculation: -(journey_times + waiting_penalty)
        # passenger 0: 2 in vehicle, 1 waiting
        # passenger 1: 4 in vehicle, 1 waiting
        # passenger 2: 0 in vehicle, 3 waiting
        # Journey times: waiting(2.0) + in_vehicle(6.0) = 8.0
        # Waiting penalty: waiting(3.0) * factor(2.0) = 6.0
        # Total = -(6.0 + 8.0) = -14.0
        chex.assert_trees_all_close(timestep.reward, jnp.array(-14.0))


class TestGetRouteShortestPaths(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, jnp.inf], [2, 0, jnp.inf], [jnp.inf, jnp.inf, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, 4, 6], [2, 0, 2, 4], [4, 2, 0, 2], [6, 4, 2, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array(
            [[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]
        )
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        # Verify optimal path lengths between various stops
        assert float(shortest_paths[0, 2]) == 3.0  # 0->1->2 is shorter than 0->2
        assert float(shortest_paths[2, 0]) == 3.0  # 2->1->0 is shorter than 2->0
        assert float(shortest_paths[1, 3]) == 3.0  # 1->2->3 is shorter than 1->3
        assert float(shortest_paths[3, 1]) == 3.0  # 3->2->1 is shorter than 3->1

    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        expected_paths = jnp.array([[0, 2, 3], [3, 0, 1], [5, 2, 0]])
        assert jnp.allclose(shortest_paths, expected_paths, equal_nan=True)

    @chex.variants(with_jit=True, without_jit=True)
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

        _get_route_shortest_paths = self.variant(Mandl._get_route_shortest_paths)
        shortest_paths = _get_route_shortest_paths(network, route.nodes)

        # Verify self-loops don't affect shortest paths
        assert float(shortest_paths[0, 1]) == 2.0  # Direct path 0->1
        assert float(shortest_paths[1, 2]) == 3.0  # Direct path 1->2


class TestGetAllShortestPaths(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_single_route(self) -> None:
        mandl_env = Mandl()
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
        _get_all_shortest_paths = self.variant(mandl_env._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

        assert shortest_paths.shape == (1, 3, 3)  # (num_routes, num_nodes, num_nodes)
        assert jnp.allclose(shortest_paths[0], jnp.array([[0, 2, 5], [2, 0, 3], [5, 3, 0]]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_routes(self) -> None:
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

        _get_all_shortest_paths = self.variant(Mandl._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

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

    @chex.variants(with_jit=True, without_jit=True)
    def test_empty_routes(self) -> None:
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

        _get_all_shortest_paths = self.variant(Mandl._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

        assert shortest_paths.shape == (1, 3, 3)
        assert jnp.allclose(
            shortest_paths[0],
            jnp.array([[0, jnp.inf, jnp.inf], [jnp.inf, 0, jnp.inf], [jnp.inf, jnp.inf, 0]]),
            equal_nan=True,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_long_routes_both_directions(self) -> None:
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

        _get_all_shortest_paths = self.variant(Mandl._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

        # Forward direction (route 0)
        assert float(shortest_paths[0, 0, 4]) == 10.0  # 0->1->2->3->4 = 2+3+1+4 = 10
        assert float(shortest_paths[0, 0, 2]) == 5.0  # 0->1->2 = 2+3 = 5
        assert float(shortest_paths[0, 1, 3]) == 4.0  # 1->2->3 = 3+1 = 4

        # Reverse direction (route 1)
        assert float(shortest_paths[1, 4, 0]) == 10.0  # 4->3->2->1->0 = 4+1+3+2 = 10
        assert float(shortest_paths[1, 4, 2]) == 5.0  # 4->3->2 = 4+1 = 5
        assert float(shortest_paths[1, 3, 1]) == 4.0  # 3->2->1 = 1+3 = 4

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_paths_between_stops(self) -> None:
        """Test shortest path calculation with multiple possible routes between stops."""
        network = NetworkData(
            nodes=jnp.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
            links=jnp.array(
                [
                    [0, 2, jnp.inf, jnp.inf, 7],  # From node 0
                    [2, 0, 3, jnp.inf, jnp.inf],  # From node 1
                    [jnp.inf, 3, 0, 1, jnp.inf],  # From node 2
                    [jnp.inf, jnp.inf, 1, 0, 4],  # From node 3
                    [7, jnp.inf, jnp.inf, 4, 0],  # From node 4
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

        _get_all_shortest_paths = self.variant(Mandl._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

        # Verify optimal path lengths for route 0 (longer path)
        assert float(shortest_paths[0, 0, 4]) == 10.0  # Route 0: 0->1->2->3->4 = 2+3+1+4 = 10
        assert float(shortest_paths[0, 4, 0]) == 10.0  # Route 0: 4->3->2->1->0 = 4+1+3+2 = 10

        # Verify optimal path lengths for route 1 (direct path)
        assert float(shortest_paths[1, 0, 4]) == 7.0  # Route 1: direct path 0->4 = 7
        assert float(shortest_paths[1, 4, 0]) == 7.0  # Route 1: direct path 4->0 = 7

        # Verify intermediate path segments for route 0
        assert float(shortest_paths[0, 1, 4]) == 8.0  # Route 0: 1->2->3->4 = 3+1+4 = 8
        assert float(shortest_paths[0, 4, 1]) == 8.0  # Route 0: 4->3->2->1 = 4+1+3 = 8
        assert float(shortest_paths[0, 2, 4]) == 5.0  # Route 0: 2->3->4 = 1+4 = 5
        assert float(shortest_paths[0, 4, 2]) == 5.0  # Route 0: 4->3->2 = 4+1 = 5
        assert float(shortest_paths[0, 3, 4]) == 4.0  # Route 0: 3->4 = 4
        assert float(shortest_paths[0, 4, 3]) == 4.0  # Route 0: 4->3 = 4

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_paths_sorted_between_all_pairs(self) -> None:
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

        _get_all_shortest_paths = self.variant(Mandl._get_all_shortest_paths)
        shortest_paths = _get_all_shortest_paths(network=network, routes=routes)

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


class TestFindDirectPaths(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_single_direct_path(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=1)

        # Check paths array shape and first path
        assert paths.shape[1] == 3  # [route_idx, valid, cost]
        assert jnp.allclose(paths[0], jnp.array([0, 1.0, 2.0]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_no_direct_paths(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=1)
        assert jnp.all(paths[:, 2] == jnp.inf)  # All costs are inf

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_paths_sorted_by_cost(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=2)

        # Check we got 2 paths sorted by cost
        assert paths.shape[0] == 2
        assert jnp.allclose(paths[0], jnp.array([0, 1.0, 5.0]))  # First path cheaper
        assert jnp.allclose(paths[1], jnp.array([1, 1.0, 8.0]))  # Second path more expensive

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_valid_path_among_multiple_routes(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=1)

        assert paths.shape[0] == 2  # Still get all paths
        # But only the second one is valid
        assert jnp.allclose(paths[0], jnp.array([1, 1.0, 2.0]))  # Valid path
        assert paths[1, 2] == jnp.inf  # Invalid path has infinite cost

    @chex.variants(with_jit=True, without_jit=True)
    def test_reverse_direction(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=1, end=0)

        assert paths.shape[0] == 1
        assert jnp.allclose(paths[0], jnp.array([0, 1.0, 2.0]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_path_with_transfer_not_found(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=2)
        assert jnp.all(paths[:, 2] == jnp.inf)  # All paths should have infinite cost

    @pytest.mark.skip()
    def test_paths_with_loops(self) -> None:
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

        _find_direct_paths = self.variant(Mandl._find_direct_paths)
        paths = _find_direct_paths(routes=routes, network=network, start=0, end=2)

        assert paths.shape[0] == 1
        assert jnp.allclose(
            paths[0], jnp.array([0, 1.0, 8.0])
        )  # Should find shortest path despite loop


class TestFindTransferPaths(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_single_transfer_path(self) -> None:
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

        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths = _find_transfer_paths(
            routes=routes, network=network, start=0, end=2, transfer_penalty=2.0
        )

        assert paths.shape[1] == 5  # [first_route, second_route, transfer_stop, valid, cost]
        assert jnp.allclose(
            paths[0], jnp.array([0, 1, 1, 1.0, 7.0])
        )  # 2 (0->1) + 3 (1->2) + 2 (penalty)

    @chex.variants(with_jit=True, without_jit=True)
    def test_no_transfer_paths(self) -> None:
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

        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )
        assert jnp.all(paths[:, 4] == jnp.inf)  # All costs should be infinite

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_transfer_paths_sorted(self) -> None:
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

        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        # Verify we get two paths sorted by cost
        assert paths.shape[0] >= 2
        # Path 1: 0->1 (2) + 1->2 (3) + transfer (2) = 7
        assert jnp.allclose(paths[0], jnp.array([0, 1, 1, 1.0, 7.0]))
        # Path 2: 0->1 (2) + 1->3->2 (6) + transfer (2) = 10
        assert jnp.allclose(paths[1], jnp.array([0, 2, 1, 1.0, 10.0]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_different_transfer_penalties(self) -> None:
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
        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths_high = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=4.0
        )
        assert jnp.allclose(
            paths_high[0], jnp.array([0, 1, 1, 1.0, 9.0])
        )  # 2 + 3 + 4 (higher penalty)

        # Test with lower transfer penalty
        paths_low = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=1.0
        )
        assert jnp.allclose(
            paths_low[0], jnp.array([0, 1, 1, 1.0, 6.0])
        )  # 2 + 3 + 1 (lower penalty)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_transfer_stops(self) -> None:
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

        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        assert paths.shape[0] >= 2
        # Check there's a path transferring at node 1
        assert jnp.any(
            jnp.logical_and(paths[:, 2] == 1, jnp.isclose(paths[:, 4], 7.0))
        )  # 2 + 3 + 2
        # Check there's a path transferring at node 3
        assert jnp.any(
            jnp.logical_and(paths[:, 2] == 3, jnp.isclose(paths[:, 4], 10.0))
        )  # 6 + 2 + 2

    @chex.variants(with_jit=True, without_jit=True)
    def test_reverse_direction(self) -> None:
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
        _find_transfer_paths = self.variant(Mandl._find_transfer_paths)
        paths_forward = _find_transfer_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )
        assert jnp.allclose(paths_forward[0], jnp.array([0, 1, 1, 1.0, 7.0]))  # 2 + 3 + 2

        # Test reverse direction (2->0)
        paths_reverse = _find_transfer_paths(
            network=network, routes=routes, start=2, end=0, transfer_penalty=2.0
        )
        assert jnp.allclose(paths_reverse[0], jnp.array([1, 0, 1, 1.0, 7.0]))  # 3 + 2 + 2


class TestFindPaths(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_direct_path_exists(self) -> None:
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

        _find_paths = self.variant(Mandl._find_paths)
        direct_paths, transfer_paths = _find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        # Check direct path
        assert jnp.allclose(direct_paths[0], jnp.array([0, 1.0, 5.0]))  # [route_idx, valid, cost]

        # Check transfer paths are all invalid (infinite cost)
        assert jnp.all(transfer_paths[:, 4] == jnp.inf)

    @chex.variants(with_jit=True, without_jit=True)
    def test_only_transfer_path_exists(self) -> None:
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

        _find_paths = self.variant(Mandl._find_paths)
        direct_paths, transfer_paths = _find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        # Check direct paths are all invalid
        assert jnp.all(direct_paths[:, 2] == jnp.inf)

        # Check transfer path
        assert jnp.allclose(
            transfer_paths[0],
            jnp.array(
                [0, 1, 1, 1.0, 7.0]
            ),  # [first_route, second_route, transfer_stop, valid, cost]
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_no_paths_exist(self) -> None:
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

        _find_paths = self.variant(Mandl._find_paths)
        direct_paths, transfer_paths = _find_paths(
            network=network, routes=routes, start=0, end=2, transfer_penalty=2.0
        )

        # Check both path types are invalid
        assert jnp.all(direct_paths[:, 2] == jnp.inf)
        assert jnp.all(transfer_paths[:, 4] == jnp.inf)

    @chex.variants(with_jit=True, without_jit=True)
    def test_direct_path_preferred_over_transfers(self) -> None:
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

        _find_paths = self.variant(Mandl._find_paths)
        direct_paths, transfer_paths = _find_paths(
            network=network, routes=routes, start=0, end=3, transfer_penalty=2.0
        )

        # Check direct path exists and is valid
        assert jnp.allclose(direct_paths[0], jnp.array([0, 1.0, 5.0]))

        # Check transfer paths are all invalid since direct path exists
        assert jnp.all(transfer_paths[:, 4] == jnp.inf)

    @chex.variants(with_jit=True, without_jit=True)
    def test_transfer_penalty_affects_cost(self) -> None:
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

        _find_paths = self.variant(Mandl._find_paths)
        direct_paths, transfer_paths = _find_paths(
            network=network,
            routes=routes,
            start=0,
            end=2,
            transfer_penalty=5.0,  # Higher penalty
        )

        # Check direct paths are invalid
        assert jnp.all(direct_paths[:, 2] == jnp.inf)

        # Check transfer path with higher penalty
        assert jnp.allclose(
            transfer_paths[0],
            jnp.array([0, 1, 1, 1.0, 10.0]),  # 2 (0->1) + 3 (1->2) + 5 (transfer penalty)
        )


class TestUpdateFlexibleRoutes(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_initial_route_creation(self) -> None:
        """Test adding first node to an empty flexible route."""
        # Load actual fixed routes from file
        env = Mandl(num_flex_routes=1)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        # Get number of fixed routes
        num_fixed = len(state.routes.ids) - env.num_flex_routes

        # Create test routes by replacing the flexible route portion
        routes = state.routes._replace(
            nodes=jnp.concatenate(
                [
                    state.routes.nodes[:num_fixed],  # Keep original fixed routes
                    jnp.full((1, env.num_nodes), -1, dtype=int),  # Empty flexible route
                ]
            ),
        )

        _update_flexible_routes = self.variant(env._update_flexible_routes)

        # Add node 0 to flexible route
        action = jnp.array([0])
        new_routes = _update_flexible_routes(routes, action, state.network.links)

        # Check fixed routes weren't modified
        assert jnp.array_equal(new_routes.nodes[:num_fixed], routes.nodes[:num_fixed])

        # Check that node was added correctly to flexible route
        expected_flex_route = jnp.full((env.num_nodes,), -1, dtype=int).at[0].set(0)
        assert jnp.array_equal(new_routes.nodes[num_fixed], expected_flex_route)

    @chex.variants(with_jit=True, without_jit=True)
    def test_append_valid_node(self) -> None:
        """Test appending a valid node to existing flexible route."""
        # Initialize environment with real fixed routes
        env = Mandl(num_flex_routes=1)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        num_fixed = len(state.routes.ids) - env.num_flex_routes

        # Create test routes with one node in flexible route
        initial_flex_route = jnp.full((env.num_nodes,), -1, dtype=int).at[0].set(0)
        routes = state.routes._replace(
            nodes=jnp.concatenate(
                [
                    state.routes.nodes[:num_fixed],
                    initial_flex_route[None, :],  # Add batch dimension
                ]
            ),
        )

        _update_flexible_routes = self.variant(env._update_flexible_routes)

        # Add node 1 to flexible route (assuming it's connected to node 0)
        action = jnp.array([1])
        new_routes = _update_flexible_routes(routes, action, state.network.links)

        # Check fixed routes weren't modified
        assert jnp.array_equal(new_routes.nodes[:num_fixed], routes.nodes[:num_fixed])

        # Check that node was appended correctly
        expected_flex_route = jnp.full((env.num_nodes,), -1, dtype=int)
        expected_flex_route = expected_flex_route.at[0].set(0).at[1].set(1)
        assert jnp.array_equal(new_routes.nodes[num_fixed], expected_flex_route)

    @chex.variants(with_jit=True, without_jit=True)
    def test_wait_action(self) -> None:
        """Test that wait action (-1) doesn't modify route."""
        # Initialize environment with real fixed routes
        env = Mandl(num_flex_routes=1)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        num_fixed = len(state.routes.ids) - env.num_flex_routes

        # Create test routes with existing flexible route
        initial_flex_route = jnp.full((env.num_nodes,), -1, dtype=int)
        initial_flex_route = initial_flex_route.at[0].set(0).at[1].set(1)
        routes = state.routes._replace(
            nodes=jnp.concatenate(
                [
                    state.routes.nodes[:num_fixed],
                    initial_flex_route[None, :],
                ]
            ),
        )

        _update_flexible_routes = self.variant(env._update_flexible_routes)

        # Wait action
        action = jnp.array([-1])
        new_routes = _update_flexible_routes(routes, action, state.network.links)

        # Check that all routes remained unchanged
        assert jnp.array_equal(new_routes.nodes, routes.nodes)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multiple_flexible_routes(self) -> None:
        """Test updating multiple flexible routes simultaneously."""
        # Initialize environment with real fixed routes
        env = Mandl(num_flex_routes=2)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        num_fixed = len(state.routes.ids) - env.num_flex_routes

        # Create test routes with partially filled flexible routes
        flex_route1 = jnp.full((env.num_nodes,), -1, dtype=int).at[0].set(0)
        flex_route2 = jnp.full((env.num_nodes,), -1, dtype=int).at[0].set(1)

        routes = state.routes._replace(
            nodes=jnp.concatenate(
                [
                    state.routes.nodes[:num_fixed],
                    flex_route1[None, :],
                    flex_route2[None, :],
                ]
            ),
        )

        _update_flexible_routes = self.variant(env._update_flexible_routes)

        # Add nodes to both flexible routes
        action = jnp.array([1, 2])  # Add node 1 to first route, node 2 to second route
        new_routes = _update_flexible_routes(routes, action, state.network.links)

        # Check fixed routes weren't modified
        assert jnp.array_equal(new_routes.nodes[:num_fixed], routes.nodes[:num_fixed])

        # Check that nodes were added correctly to flexible routes
        expected_flex_route1 = jnp.full((env.num_nodes,), -1, dtype=int)
        expected_flex_route1 = expected_flex_route1.at[0].set(0).at[1].set(1)
        assert jnp.array_equal(new_routes.nodes[num_fixed], expected_flex_route1)

        expected_flex_route2 = jnp.full((env.num_nodes,), -1, dtype=int)
        expected_flex_route2 = expected_flex_route2.at[0].set(1).at[1].set(2)
        assert jnp.array_equal(new_routes.nodes[num_fixed + 1], expected_flex_route2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_full_route(self) -> None:
        """Test attempting to add a node to a full route."""
        # Initialize environment with real fixed routes
        env = Mandl(num_flex_routes=1)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        num_fixed = len(state.routes.ids) - env.num_flex_routes

        # Create a full flexible route (no -1s)
        full_flex_route = jnp.arange(env.num_nodes)
        routes = state.routes._replace(
            nodes=jnp.concatenate(
                [
                    state.routes.nodes[:num_fixed],
                    full_flex_route[None, :],
                ]
            ),
        )

        _update_flexible_routes = self.variant(env._update_flexible_routes)

        # Try to add another node
        action = jnp.array([0])
        new_routes = _update_flexible_routes(routes, action, state.network.links)

        # Check that route remained unchanged
        assert jnp.array_equal(new_routes.nodes, routes.nodes)


if __name__ == "__main__":
    pytest.main([__file__])
