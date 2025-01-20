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

import chex
import jax.numpy as jnp
import pytest
from jax import jit
from jaxtyping import TypeCheckError

from jumanji.environments.routing.mandl.types import NetworkData, RouteBatch, RouteType


class TestNetworkData:
    @pytest.fixture
    def sample_network(self) -> NetworkData:
        node_coordinates = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        travel_times = jnp.array(
            [
                [0.0, 1.0, jnp.inf],
                [1.0, 0.0, 2.0],
                [jnp.inf, 2.0, 0.0],
            ]
        )
        is_terminal = jnp.array([True, False, False])
        return NetworkData(
            node_coordinates=node_coordinates, travel_times=travel_times, is_terminal=is_terminal
        )

    def test_network_data_initialization(self, sample_network: NetworkData) -> None:
        assert sample_network.node_coordinates.shape == (3, 2)
        assert sample_network.travel_times.shape == (3, 3)
        assert sample_network.is_terminal.shape == (3,)

    def test_network_data_invalid_shapes(self) -> None:
        node_coordinates = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        travel_times = jnp.array([[0.0, 1.0], [1.0, 0.0], [1.0, 2.0]])
        is_terminal = jnp.array([True, False])
        # shape is checked via jaxtyping see https://docs.kidger.site/jaxtyping/api/runtime-type-checking/
        with pytest.raises(TypeCheckError):
            NetworkData(
                node_coordinates=node_coordinates,
                travel_times=travel_times,
                is_terminal=is_terminal,
            )

    def test_network_data_invalid_values(self) -> None:
        node_coordinates = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        travel_times = jnp.array([[0.0, -1.0], [-1.0, 0.0]])
        is_terminal = jnp.array([True, False])
        with pytest.raises(ValueError):
            NetworkData(
                node_coordinates=node_coordinates,
                travel_times=travel_times,
                is_terminal=is_terminal,
            )

    @pytest.mark.skip
    def test_empty_network_initialization(self) -> None:
        pass

    def test_is_connected(self, sample_network: NetworkData) -> None:
        is_connected = jit(sample_network.is_connected)
        chex.assert_equal(is_connected(0, 1), True)
        chex.assert_equal(is_connected(0, 2), False)
        chex.assert_equal(is_connected(1, 2), True)

    @pytest.mark.skip
    def test_is_connected_invalid_indices(self, sample_network: NetworkData) -> None:
        # not sure what to do about it see
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        pass

    def test_get_travel_time(self, sample_network: NetworkData) -> None:
        get_travel_time = jit(sample_network.get_travel_time)
        chex.assert_equal(get_travel_time(0, 1), 1.0)
        chex.assert_equal(get_travel_time(0, 2), jnp.inf)
        chex.assert_equal(get_travel_time(1, 2), 2.0)
        chex.assert_equal(get_travel_time(0, 0), 0.0)

    @pytest.mark.skip
    def test_get_travel_time_invalid_indices(self, sample_network: NetworkData) -> None:
        # not sure what to do about it see
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        pass

    def test_self_connection(self, sample_network: NetworkData) -> None:
        is_connected = jit(sample_network.is_connected)
        get_travel_time = jit(sample_network.get_travel_time)
        chex.assert_equal(is_connected(0, 0), True)
        chex.assert_equal(get_travel_time(0, 0), 0.0)

    def test_non_connectivity(self, sample_network: NetworkData) -> None:
        get_travel_time = jit(sample_network.get_travel_time)
        is_connected = jit(sample_network.is_connected)
        chex.assert_equal(get_travel_time(0, 2), jnp.inf)
        chex.assert_equal(is_connected(0, 2), False)

    def test_single_node_network(self) -> None:
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array([[0.0]], dtype=jnp.float32),
            is_terminal=jnp.array([True], dtype=bool),
        )

        from_node = jnp.array(0)
        to_node = jnp.array(0)
        is_connected = jit(network.is_connected)
        get_travel_time = jit(network.get_travel_time)
        chex.assert_equal(is_connected(from_node=from_node, to_node=to_node), True)
        chex.assert_equal(get_travel_time(from_node=from_node, to_node=to_node), 0.0)


class TestRouteBatch:
    @pytest.fixture
    def all_on_demand_routes(self) -> RouteBatch:
        num_routes = 5
        return RouteBatch(
            ids=jnp.arange(num_routes),
            types=jnp.full(num_routes, RouteType.FLEXIBLE.value),
            stops=jnp.zeros((num_routes, 10), dtype=int),  # assuming max_route_length=10
            frequencies=jnp.ones(num_routes),
            on_demand=jnp.ones(num_routes, dtype=bool),
            num_flex_routes=num_routes,
            num_fix_routes=0,
        )

    @pytest.fixture
    def no_on_demand_routes(self) -> RouteBatch:
        num_routes = 5
        return RouteBatch(
            ids=jnp.arange(num_routes),
            types=jnp.full(num_routes, RouteType.FIXED.value),
            stops=jnp.zeros((num_routes, 10), dtype=int),
            frequencies=jnp.ones(num_routes),
            on_demand=jnp.zeros(num_routes, dtype=bool),
            num_flex_routes=0,
            num_fix_routes=5,
        )

    @pytest.fixture
    def mixed_routes(self) -> RouteBatch:
        num_routes = 5
        return RouteBatch(
            ids=jnp.arange(num_routes),
            types=jnp.array(
                [
                    RouteType.FIXED.value,
                    RouteType.FLEXIBLE.value,
                    RouteType.FIXED.value,
                    RouteType.FLEXIBLE.value,
                    RouteType.FLEXIBLE.value,
                ]
            ),
            stops=jnp.zeros((num_routes, 10), dtype=int),
            frequencies=jnp.ones(num_routes),
            on_demand=jnp.array([False, True, False, True, True]),
            num_flex_routes=3,
            num_fix_routes=2,
        )

    def test_get_valid_stops_all_valid(self, mixed_routes: RouteBatch) -> None:
        # Assuming stops are valid if within node indices [0, 100)
        mixed_routes.stops = jnp.array(
            [
                [1, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                [10, 20, 30, -1, -1, -1, -1, -1, -1, -1],
                [5, 6, 7, -1, -1, -1, -1, -1, -1, -1],
                [15, 25, 35, -1, -1, -1, -1, -1, -1, -1],
                [50, 60, 70, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        get_valid_stops = jit(mixed_routes.get_valid_stops)
        valid_mask = get_valid_stops()
        expected_mask = jnp.array(
            [
                [True, True, True, False, False, False, False, False, False, False],
                [True, True, True, False, False, False, False, False, False, False],
                [True, True, True, False, False, False, False, False, False, False],
                [True, True, True, False, False, False, False, False, False, False],
                [True, True, True, False, False, False, False, False, False, False],
            ]
        )
        assert (valid_mask == expected_mask).all()

    @pytest.mark.skip
    def test_get_valid_stops_with_invalid_indices(self, mixed_routes: RouteBatch) -> None:
        pass

    def test_update_routes_valid_actions(self, mixed_routes: RouteBatch) -> None:
        # Assume actions are new stop indices to replace the last valid stop
        actions = jnp.array([100, 200, 300])  # For the 3 on-demand routes
        update_routes = jit(mixed_routes.update_routes)
        updated_batch = update_routes(actions)
        chex.assert_trees_all_equal(updated_batch.stops[[1, 3, 4], 0], jnp.array([100, 200, 300]))

    @pytest.mark.skip
    def test_update_routes_empty_actions(self, mixed_routes: RouteBatch) -> None:
        pass

    @pytest.mark.skip
    def test_update_routes_mismatched_actions_length(self, mixed_routes: RouteBatch) -> None:
        pass

    @pytest.mark.skip
    def test_update_routes_invalid_action_values(self, mixed_routes: RouteBatch) -> None:
        pass

    def test_update_routes_all_fixed(self, no_on_demand_routes: RouteBatch) -> None:
        actions = jnp.array([], jnp.int32)  # No on-demand routes to update
        update_routes = jit(no_on_demand_routes.update_routes)
        updated_batch = update_routes(actions)
        chex.assert_trees_all_equal(updated_batch, no_on_demand_routes)
