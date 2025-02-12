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
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.mandl import Mandl
from jumanji.environments.routing.mandl.types import (
    Fleet,
    NetworkData,
    Passengers,
    RouteBatch,
    RouteType,
    State,
)

# jax.config.update("jax_disable_jit", True)


class TestActionMasking:
    @pytest.fixture
    def env(self) -> Mandl:
        """Create a Mandl environment with 2 flex routes."""
        return Mandl(num_flex_routes=2)

    @pytest.fixture
    def simple_network(self) -> NetworkData:
        """Create a simple network with 3 nodes in a line: 0--1--2"""
        return NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            travel_times=jnp.array([[0.0, 1.0, jnp.inf], [1.0, 0.0, 1.0], [jnp.inf, 1.0, 0.0]]),
            is_terminal=jnp.array([True, False, True]),
        )

    @pytest.fixture
    def empty_routes(self, simple_network: NetworkData) -> RouteBatch:
        """Create a route batch with empty flexible routes"""
        return RouteBatch(
            types=jnp.array([RouteType.FLEXIBLE, RouteType.FLEXIBLE]),
            stops=jnp.array([[-1, -1, -1], [-1, -1, -1]]),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(2),
            num_fix_routes=jnp.array(0),
        )

    def test_empty_route_mask(
        self, env: Mandl, simple_network: NetworkData, empty_routes: RouteBatch
    ) -> None:
        """Test that all actions are allowed for empty routes."""
        state = State(
            network=simple_network,
            fleet=Fleet(
                route_ids=jnp.array([0, 1]),
                current_edges=jnp.array([[0, 0], [0, 0]]),
                times_on_edge=jnp.zeros(2),
                passengers=jnp.array([[-1], [-1]]),
                directions=jnp.zeros(2, dtype=int),
            ),
            passengers=Passengers(
                origins=jnp.array([], dtype=int),
                destinations=jnp.array([], dtype=int),
                desired_departure_times=jnp.array([], dtype=float),
                time_waiting=jnp.array([], dtype=float),
                time_in_vehicle=jnp.array([], dtype=float),
                statuses=jnp.array([], dtype=int),
                has_transferred=jnp.array([], dtype=bool),
                transfer_nodes=jnp.array([], dtype=int),
            ),
            routes=empty_routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        get_action_mask = jax.jit(env.get_action_mask)
        mask = get_action_mask(state)
        expected_mask = jnp.ones((2, 4), dtype=bool)  # 3 nodes + 1 no-op action
        assert jnp.array_equal(mask, expected_mask)
