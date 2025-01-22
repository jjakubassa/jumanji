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

from jumanji.environments.routing.mandl.types import Fleet, NetworkData, RouteBatch, RouteType


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


class TestFleet:
    @pytest.fixture
    def fully_occupied_fleet(self) -> Fleet:
        return Fleet(
            route_ids=jnp.array([0], dtype=int),
            current_edges=jnp.array([[0, 1]], dtype=int),
            current_edges_indices=jnp.array([0], dtype=int),
            times_on_edge=jnp.array([15.0], dtype=float),
            passengers=jnp.array([[1, 2, 3, 4, 5]], dtype=int),  # max_capacity=5
            directions=jnp.array([0], dtype=int),
        )

    @pytest.fixture
    def partially_occupied_fleet(self) -> Fleet:
        return Fleet(
            route_ids=jnp.array([0, 1], dtype=int),
            current_edges=jnp.array([[0, 1], [1, 2]], dtype=int),
            current_edges_indices=jnp.array([0, 1], dtype=int),
            times_on_edge=jnp.array([10.0, 20.0], dtype=float),
            passengers=jnp.array(
                [
                    [1, -1, -1, -1, -1],  # 1 passenger
                    [2, 3, -1, -1, -1],  # 2 passengers
                ],
                dtype=int,
            ),
            directions=jnp.array([0, 1], dtype=int),
        )

    @pytest.fixture
    def mixed_fleet(self) -> Fleet:
        return Fleet(
            route_ids=jnp.array([0, 1, 2], dtype=int),
            current_edges=jnp.array([[0, 1], [1, 2], [2, 3]], dtype=int),
            current_edges_indices=jnp.array([0, 1, 2], dtype=int),
            times_on_edge=jnp.array([10.0, 20.0, 30.0], dtype=float),
            passengers=jnp.array(
                [
                    [1, 2, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [3, 4, 5, 6, 7],
                ],
                dtype=int,
            ),
            directions=jnp.array([0, 1, 0], dtype=int),
        )

    def test_capacities_left_fully_occupied(self, fully_occupied_fleet: Fleet) -> None:
        capacities_left = fully_occupied_fleet.capacities_left
        chex.assert_trees_all_equal(capacities_left, jnp.array([0], dtype=int))

    def test_capacities_left_partially_occupied(self, partially_occupied_fleet: Fleet) -> None:
        capacities_left = partially_occupied_fleet.capacities_left
        chex.assert_trees_all_equal(capacities_left, jnp.array([4, 3], dtype=int))

    def test_capacities_left_mixed_fleet(self, mixed_fleet: Fleet) -> None:
        capacities_left = mixed_fleet.capacities_left
        chex.assert_trees_all_equal(capacities_left, jnp.array([3, 5, 0], dtype=int))

    def test_seat_is_available_fully_occupied(self, fully_occupied_fleet: Fleet) -> None:
        seat_available = fully_occupied_fleet.seat_is_available
        chex.assert_trees_all_equal(seat_available, jnp.array([False], dtype=int))

    def test_seat_is_available_partially_occupied(self, partially_occupied_fleet: Fleet) -> None:
        seat_available = partially_occupied_fleet.seat_is_available
        chex.assert_trees_all_equal(seat_available, jnp.array([True, True], dtype=int))

    def test_seat_is_available_mixed_fleet(self, mixed_fleet: Fleet) -> None:
        seat_available = mixed_fleet.seat_is_available
        chex.assert_trees_all_equal(seat_available, jnp.array([True, True, False], dtype=int))

    @pytest.mark.skip
    def test_invalid_passenger_ids(self) -> None:
        raise NotImplementedError

    @pytest.mark.skip
    def test_fleet_with_no_passengers(self, no_passengers_fleet: Fleet) -> None:
        raise NotImplementedError

    def test_fleet_with_max_passengers(self, fully_occupied_fleet: Fleet) -> None:
        """Fleet where passenger slots are filled to maximum capacity."""
        capacities_left = fully_occupied_fleet.capacities_left
        chex.assert_trees_all_equal(capacities_left, jnp.array([0], dtype=int))

        seat_available = fully_occupied_fleet.seat_is_available
        chex.assert_trees_all_equal(seat_available, jnp.array([False], dtype=bool))

    def test_add_passenger_success(self, mixed_fleet: Fleet) -> None:
        """
        Test adding a passenger to a vehicle with available seats.
        """
        # Vehicle 0 has passengers [1, 2, -1, -1, -1], so first available seat is index 2
        add_passenger = jit(mixed_fleet.add_passenger)
        updated_fleet = add_passenger(vehicle_id=0, passenger_id=10)
        expected_passengers = jnp.array(
            [
                [1, 2, 10, -1, -1],
                [-1, -1, -1, -1, -1],
                [3, 4, 5, 6, 7],
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_fleet.passengers, expected_passengers)

    @pytest.mark.skip
    def test_add_passenger_to_full_vehicle(self, mixed_fleet: Fleet) -> None:
        """
        Test adding a passenger to a fully occupied vehicle should raise an error.
        """
        raise NotImplementedError

    @pytest.mark.skip
    def test_add_passenger_invalid_vehicle_id(self, mixed_fleet: Fleet) -> None:
        """
        Test adding a passenger to an invalid vehicle ID should raise an error.
        """
        raise NotImplementedError

    def test_remove_passenger_success(self, mixed_fleet: Fleet) -> None:
        """
        Test removing a passenger from a vehicle where the passenger exists.
        """
        # Remove passenger 4 from vehicle 2
        remove_passenger = jit(mixed_fleet.remove_passenger)
        updated_fleet = remove_passenger(vehicle_id=2, passenger_id=4)
        expected_passengers = jnp.array(
            [
                [1, 2, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [3, -1, 5, 6, 7],  # Passenger 4 removed
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_fleet.passengers, expected_passengers)

    @pytest.mark.skip
    def test_remove_passenger_not_found(self, mixed_fleet: Fleet) -> None:
        """
        Test removing a passenger who is not in the specified vehicle should raise an error.
        """
        raise NotImplementedError

    @pytest.mark.skip
    def test_remove_passenger_invalid_vehicle_id(self, mixed_fleet: Fleet) -> None:
        """
        Test removing a passenger from an invalid vehicle ID should raise an error.
        """
        raise NotImplementedError

    def test_add_and_remove_passenger_sequence(self, mixed_fleet: Fleet) -> None:
        """
        Test the sequence of adding a passenger and then removing them to ensure consistency.
        """
        # Add passenger 10 to vehicle 0
        add_passenger = jit(mixed_fleet.add_passenger)
        fleet_after_add = add_passenger(vehicle_id=0, passenger_id=10)
        expected_after_add = jnp.array(
            [
                [1, 2, 10, -1, -1],
                [-1, -1, -1, -1, -1],
                [3, 4, 5, 6, 7],
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(fleet_after_add.passengers, expected_after_add)

        # Remove passenger 10 from vehicle 0
        remove_passenger = jit(fleet_after_add.remove_passenger)
        fleet_after_remove = remove_passenger(vehicle_id=0, passenger_id=10)
        expected_after_remove = jnp.array(
            [
                [1, 2, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [3, 4, 5, 6, 7],
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(fleet_after_remove.passengers, expected_after_remove)

    @pytest.mark.skip
    def test_add_passenger_no_fleet(self, empty_fleet: Fleet) -> None:
        """
        Test adding a passenger when the fleet is empty should raise an error.
        """
        raise NotImplementedError

    @pytest.mark.skip
    def test_remove_passenger_no_fleet(self, empty_fleet: Fleet) -> None:
        """
        Test removing a passenger when the fleet is empty should raise an error.
        """
        raise NotImplementedError
