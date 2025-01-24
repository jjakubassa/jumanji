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
import jax.numpy as jnp
import pytest
from jax import jit
from jaxtyping import TypeCheckError

from jumanji.environments.routing.mandl.types import (
    Fleet,
    NetworkData,
    Passengers,
    PassengerStatus,
    RouteBatch,
    RouteType,
)


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
            types=jnp.full(num_routes, RouteType.FLEXIBLE),
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
            types=jnp.full(num_routes, RouteType.FIXED),
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
                    RouteType.FIXED,
                    RouteType.FLEXIBLE,
                    RouteType.FIXED,
                    RouteType.FLEXIBLE,
                    RouteType.FLEXIBLE,
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


class TestPassengers:
    @pytest.fixture
    def empty_passenger_state(self) -> Passengers:
        return Passengers(
            origins=jnp.array([], dtype=int),
            destinations=jnp.array([], dtype=int),
            desired_departure_times=jnp.array([], dtype=float),
            time_waiting=jnp.array([], dtype=float),
            time_in_vehicle=jnp.array([], dtype=float),
            statuses=jnp.array([], dtype=int),
        )

    @pytest.fixture
    def all_waiting_passengers(self) -> Passengers:
        return Passengers(
            origins=jnp.array([0, 1, 2, 3, 4], dtype=int),
            destinations=jnp.array([5, 6, 7, 8, 9], dtype=int),
            desired_departure_times=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            time_waiting=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            time_in_vehicle=jnp.array(
                [-1.0, -1.0, -1.0, -1.0, -1.0], dtype=float
            ),  # -1 indicates not in vehicle
            statuses=jnp.array(
                [
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                ],
                dtype=int,
            ),
        )

    @pytest.fixture
    def mixed_passenger_state(self) -> Passengers:
        return Passengers(
            origins=jnp.array([0, 1, 2, 3, 4, 5], dtype=int),
            destinations=jnp.array([5, 6, 7, 8, 9, 10], dtype=int),
            desired_departure_times=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
            time_waiting=jnp.array([10.0, 5.0, 0.0, 15.0, 3.0, 0.0], dtype=float),
            time_in_vehicle=jnp.array([-1.0, -1.0, 20.0, -1.0, 10.0, -1.0], dtype=float),
            statuses=jnp.array(
                [
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.COMPLETED,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.NOT_IN_SYSTEM,
                ],
                dtype=int,
            ),
        )

    @pytest.fixture
    def completed_passenger_state(self) -> Passengers:
        return Passengers(
            origins=jnp.array([0, 1, 2], dtype=int),
            destinations=jnp.array([3, 4, 5], dtype=int),
            desired_departure_times=jnp.array([0.0, 0.0, 0.0], dtype=float),
            time_waiting=jnp.array([0.0, 0.0, 0.0], dtype=float),
            time_in_vehicle=jnp.array([30.0, 40.0, 50.0], dtype=float),
            statuses=jnp.array(
                [
                    PassengerStatus.COMPLETED,
                    PassengerStatus.COMPLETED,
                    PassengerStatus.COMPLETED,
                ],
                dtype=int,
            ),
        )

    def test_initialization_empty(self, empty_passenger_state: Passengers) -> None:
        assert empty_passenger_state.origins.shape == (0,)
        assert empty_passenger_state.destinations.shape == (0,)
        assert empty_passenger_state.desired_departure_times.shape == (0,)
        assert empty_passenger_state.time_waiting.shape == (0,)
        assert empty_passenger_state.time_in_vehicle.shape == (0,)
        assert empty_passenger_state.statuses.shape == (0,)

    def test_initialization_all_waiting(self, all_waiting_passengers: Passengers) -> None:
        assert jnp.all(all_waiting_passengers.statuses == PassengerStatus.WAITING)
        assert jnp.all(all_waiting_passengers.time_in_vehicle == -1.0)

    def test_initialization_mixed_statuses(self, mixed_passenger_state: Passengers) -> None:
        expected_statuses = jnp.array(
            [
                PassengerStatus.WAITING,
                PassengerStatus.WAITING,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.COMPLETED,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.NOT_IN_SYSTEM,
            ],
            dtype=int,
        )
        assert (mixed_passenger_state.statuses == expected_statuses).all()

    def test_initialization_completed(self, completed_passenger_state: Passengers) -> None:
        assert jnp.all(completed_passenger_state.statuses == PassengerStatus.COMPLETED)
        assert jnp.all(completed_passenger_state.time_in_vehicle > 0.0)

    def test_increment_wait_times_all_waiting(self, all_waiting_passengers: Passengers) -> None:
        increment = jit(all_waiting_passengers.increment_wait_times)
        updated_state = increment()
        expected_time_waiting = (
            all_waiting_passengers.time_waiting + 1.0
        )  # Assuming increment by 1.0
        assert (updated_state.time_waiting == expected_time_waiting).all()

    def test_increment_wait_times_mixed(self, mixed_passenger_state: Passengers) -> None:
        increment = jit(mixed_passenger_state.increment_wait_times)
        updated_state = increment()
        # Only passengers with status WAITING should have their time_waiting incremented
        expected_time_waiting = mixed_passenger_state.time_waiting + jnp.where(
            mixed_passenger_state.statuses == PassengerStatus.WAITING, 1.0, 0.0
        )
        assert (updated_state.time_waiting == expected_time_waiting).all()

    def test_increment_in_vehicle_times_all_in_vehicle(
        self, mixed_passenger_state: Passengers
    ) -> None:
        # Modify mixed_passenger_state so all are IN_VEHICLE
        all_in_vehicle_state = replace(
            mixed_passenger_state,
            statuses=jnp.array(
                [
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.IN_VEHICLE,
                    PassengerStatus.IN_VEHICLE,
                ],
                dtype=int,
            ),
        )
        increment = jit(all_in_vehicle_state.increment_in_vehicle_times)
        updated_state = increment()
        expected_time_in_vehicle = (
            all_in_vehicle_state.time_in_vehicle + 1.0
        )  # Assuming increment by 1.0
        assert (updated_state.time_in_vehicle == expected_time_in_vehicle).all()

    def test_increment_in_vehicle_times_mixed(self, mixed_passenger_state: Passengers) -> None:
        increment = jit(mixed_passenger_state.increment_in_vehicle_times)
        updated_state = increment()
        # Only passengers with status IN_VEHICLE should have their time_in_vehicle incremented
        expected_time_in_vehicle = mixed_passenger_state.time_in_vehicle + jnp.where(
            mixed_passenger_state.statuses == PassengerStatus.IN_VEHICLE, 1.0, 0.0
        )
        assert (updated_state.time_in_vehicle == expected_time_in_vehicle).all()

    @pytest.fixture
    def varied_passenger_state(self) -> Passengers:
        """
        - Passenger 0: NOT_IN_SYSTEM, desired_departure_times=3.0
        - Passenger 1: NOT_IN_SYSTEM, desired_departure_times=10.0
        - Passenger 2: IN_VEHICLE
        - Passenger 3: WAITING
        - Passenger 4: COMPLETED
        """
        return Passengers(
            origins=jnp.array([0, 1, 2, 3, 4], dtype=int),
            destinations=jnp.array([5, 6, 7, 8, 9], dtype=int),
            desired_departure_times=jnp.array([3.0, 10.0, 1.0, 2.0, 2.0], dtype=float),
            time_waiting=jnp.array([0.0, 0.0, 1.0, 5.0, 10.0], dtype=float),
            time_in_vehicle=jnp.array([-1.0, -1.0, 15.0, -1.0, 20.0], dtype=float),
            statuses=jnp.array(
                [
                    PassengerStatus.NOT_IN_SYSTEM,  # index=0
                    PassengerStatus.NOT_IN_SYSTEM,  # index=1
                    PassengerStatus.IN_VEHICLE,  # index=2
                    PassengerStatus.WAITING,  # index=3
                    PassengerStatus.COMPLETED,  # index=4
                ],
                dtype=int,
            ),
        )

    def test_update_passengers_not_in_system_to_waiting(
        self, varied_passenger_state: Passengers
    ) -> None:
        """
        Only passenger 0 has desired_departure_times == 3.0, so at current_time=3.0,
        passenger 0 should move from NOT_IN_SYSTEM to WAITING.
        """
        current_time = jnp.array(3.0)
        # No one transitions to IN_VEHICLE/COMPLETED in this step:
        to_in_vehicle = jnp.array([False, False, False, False, False])
        to_completed = jnp.array([False, False, False, False, False])

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )
        expected_statuses = jnp.array(
            [
                PassengerStatus.WAITING,  # was NOT_IN_SYSTEM, now WAITING
                PassengerStatus.NOT_IN_SYSTEM,  # remains NOT_IN_SYSTEM
                PassengerStatus.IN_VEHICLE,  # unchanged
                PassengerStatus.WAITING,  # unchanged
                PassengerStatus.COMPLETED,  # unchanged
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_ps.statuses, expected_statuses)

    def test_update_passengers_waiting_to_in_vehicle(
        self, varied_passenger_state: Passengers
    ) -> None:
        """
        Mark passenger 3 for to_in_vehicle transition. Only passenger 3 moves from
        WAITING -> IN_VEHICLE.
        """
        current_time = jnp.array(2.0)  # does not match passenger 0's departure time=3.0
        to_in_vehicle = jnp.array([False, False, False, True, False])  # passenger 3 transitions
        to_completed = jnp.array([False, False, False, False, False])

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )

        # passenger 0 remains NOT_IN_SYSTEM, passenger 3 transitions to IN_VEHICLE
        expected_statuses = jnp.array(
            [
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.IN_VEHICLE,  # updated
                PassengerStatus.COMPLETED,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_ps.statuses, expected_statuses)

    def test_update_passengers_in_vehicle_to_completed(
        self, varied_passenger_state: Passengers
    ) -> None:
        """
        Mark passenger 2 for to_completed transition. Only passenger 2 moves from
        IN_VEHICLE -> COMPLETED.
        """
        current_time = jnp.array(0.0)  # won't trigger any new WAITING transitions
        to_in_vehicle = jnp.array([False, False, False, False, False])
        to_completed = jnp.array([False, False, True, False, False])  # passenger 2 transitions

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )
        expected_statuses = jnp.array(
            [
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.COMPLETED,  # updated from IN_VEHICLE
                PassengerStatus.WAITING,
                PassengerStatus.COMPLETED,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_ps.statuses, expected_statuses)

    def test_update_passengers_no_changes(self, varied_passenger_state: Passengers) -> None:
        """
        If current_time does not match any passenger's desired_departure_times and
        no transitions are flagged in the to_in_vehicle/to_completed arrays,
        then no status changes occur.
        """
        current_time = jnp.array(100.0)
        to_in_vehicle = jnp.zeros_like(varied_passenger_state.statuses, dtype=bool)
        to_completed = jnp.zeros_like(varied_passenger_state.statuses, dtype=bool)

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )
        # No changes
        chex.assert_trees_all_equal(updated_ps.statuses, varied_passenger_state.statuses)

    def test_update_passengers_conflicting_transitions(
        self, varied_passenger_state: Passengers
    ) -> None:
        """
        If a passenger is marked both to_in_vehicle and to_completed in the same step,
        the COMPLETED assignment overrides since it is applied last.
        """
        # Let's mark passenger 0 for both transitions. Even though passenger 0 is NOT_IN_SYSTEM,
        # the final 'where' for COMPLETED will override everything if to_completed is True.
        current_time = jnp.array(3.0)  # triggers passenger 0 to WAITING by time match first
        to_in_vehicle = jnp.array([True, False, False, False, False])  # passenger 0
        to_completed = jnp.array([True, False, False, False, False])  # passenger 0

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )

        # The final status for passenger 0 should be COMPLETED.
        expected_statuses = jnp.array(
            [
                PassengerStatus.COMPLETED,  # final override
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.WAITING,
                PassengerStatus.COMPLETED,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_ps.statuses, expected_statuses)

    def test_update_passengers_multiple_changes(self, varied_passenger_state: Passengers) -> None:
        """
        Example scenario:
        - passenger 0 has desired_departure_times=3.0 and current_time=3.0
        => NOT_IN_SYSTEM -> WAITING
        - passenger 3 transitions from WAITING -> IN_VEHICLE
        - passenger 2 transitions from IN_VEHICLE -> COMPLETED
        """
        current_time = jnp.array(3.0)
        to_in_vehicle = jnp.array([False, False, False, True, False])  # passenger 3
        to_completed = jnp.array([False, False, True, False, False])  # passenger 2

        update_passengers = jit(varied_passenger_state.update_passengers)
        updated_ps = update_passengers(
            current_time=current_time,
            to_in_vehicle=to_in_vehicle,
            to_completed=to_completed,
        )
        # passenger 0: NOT_IN_SYSTEM -> WAITING (due to matched time)
        # passenger 3: WAITING -> IN_VEHICLE
        # passenger 2: IN_VEHICLE -> COMPLETED
        expected_statuses = jnp.array(
            [
                PassengerStatus.WAITING,
                PassengerStatus.NOT_IN_SYSTEM,
                PassengerStatus.COMPLETED,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.COMPLETED,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_ps.statuses, expected_statuses)
