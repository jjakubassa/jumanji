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
import jax.numpy as jnp
import pytest
from jax import jit
from jaxtyping import TypeCheckError

from jumanji.environments.routing.mandl import (
    Fleet,
    NetworkData,
    Passengers,
    PassengerStatus,
    RouteBatch,
    RouteType,
    State,
    VehicleDirection,
    add_passenger,
    assign_passengers,
    calculate_journey_times,
    calculate_route_times,
    find_best_transfer_route,
    get_last_stops,
    get_travel_time,
    get_valid_stops,
    handle_completed_and_transferring_passengers,
    increment_in_vehicle_times,
    increment_wait_times,
    is_connected,
    move_vehicles,
    remove_passenger,
    update_passengers_to_waiting,
    update_routes,
)

# jax.config.update("jax_disable_jit", True)

is_connected = jit(is_connected)
get_travel_time = jit(get_travel_time)
get_valid_stops = jit(get_valid_stops)
update_routes = jit(update_routes)
get_last_stops = jit(get_last_stops)
add_passenger = jit(add_passenger)
remove_passenger = jit(remove_passenger)
add_passenger = jit(add_passenger)
increment_wait_times = jit(increment_wait_times)
increment_in_vehicle_times = jit(increment_in_vehicle_times)
update_passengers_to_waiting = jit(update_passengers_to_waiting)
calculate_route_times = jit(calculate_route_times)
move_vehicles = jit(move_vehicles)
calculate_journey_times = jit(calculate_journey_times)
assign_passengers = jit(assign_passengers)
handle_completed_and_transferring_passengers = jit(handle_completed_and_transferring_passengers)
find_best_transfer_route = jit(find_best_transfer_route)


class TestHandleCompletedPassengers:
    @pytest.fixture
    def simple_state(self) -> State:
        """Create a simple state with one vehicle and one passenger."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
            is_terminal=jnp.array([True, False], dtype=bool),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[1, 1]]),  # Vehicle at node 1
            times_on_edge=jnp.array([0.0]),  # Vehicle is at node (not moving)
            passengers=jnp.array([[0]]),  # Passenger 0 in vehicle
            directions=jnp.array([0]),
        )

        passengers = Passengers(
            origins=jnp.array([0]),
            destinations=jnp.array([1]),  # Destination matches vehicle's position
            desired_departure_times=jnp.array([0.0]),
            time_waiting=jnp.array([0.0]),
            time_in_vehicle=jnp.array([1.0]),
            statuses=jnp.array([PassengerStatus.IN_VEHICLE]),
            has_transferred=jnp.array([False]),
            transfer_nodes=jnp.array([-1]),  # -1 indicates no transfer
        )

        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED]),
            stops=jnp.array([[0, 1, -1]]),
            frequencies=jnp.ones(1),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(1),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    def test_multiple_vehicles_and_passengers(self) -> None:
        """Test handling multiple vehicles with multiple passengers.
        Setup:
        - 3 vehicles with 4 seats each
        - Vehicle 0: at node 1 (will drop off passenger 0)
        - Vehicle 1: at node 2 (will drop off passenger 2)
        - Vehicle 2: moving between nodes 0 and 1

        Passengers:
        - Passenger 0: in vehicle 0, wants node 1 (will complete)
        - Passenger 1: in vehicle 0, wants node 2
        - Passenger 2: in vehicle 1, wants node 2 (will complete)
        - Passenger 3: in vehicle 2, wants node 2
        - Passenger 4: in vehicle 2, wants node 0
        """
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array(
                [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=jnp.float32
            ),
            is_terminal=jnp.array([True, False, True], dtype=bool),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1, 2]),
            current_edges=jnp.array(
                [
                    [1, 2],  # Vehicle 0 at node 1, ready to move to node 2
                    [2, 1],  # Vehicle 1 at node 2, ready to move to node 1
                    [0, 1],  # Vehicle 2 moving from node 0 to 1
                ]
            ),
            times_on_edge=jnp.array(
                [
                    0.0,  # At node 1
                    0.0,  # At node 2
                    0.5,  # Moving between nodes
                ]
            ),
            passengers=jnp.array(
                [
                    [0, 1, -1, -1],  # Vehicle 0: passengers 0,1
                    [2, -1, -1, -1],  # Vehicle 1: passenger 2
                    [3, 4, -1, -1],  # Vehicle 2: passengers 3,4
                ]
            ),
            directions=jnp.array([0, 1, 0]),
        )

        passengers = Passengers(
            origins=jnp.array([0, 0, 0, 0, 0]),
            destinations=jnp.array([1, 2, 2, 2, 0]),  # Destinations for passengers 0-4
            desired_departure_times=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            time_waiting=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            time_in_vehicle=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            statuses=jnp.array([PassengerStatus.IN_VEHICLE] * 5),
            has_transferred=jnp.array([False] * 5),
            transfer_nodes=jnp.array([-1] * 5),  # No transfers needed
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED, RouteType.FIXED, RouteType.FIXED]),
                stops=jnp.array(
                    [
                        [0, 1, 2, -1],  # Route for vehicle 0
                        [2, 1, 0, -1],  # Route for vehicle 1
                        [0, 1, 2, -1],  # Route for vehicle 2
                    ]
                ),
                frequencies=jnp.ones(3),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(3),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check fleet updates:
        # - Vehicle 0: Passenger 0 should be removed (reached node 1)
        # - Vehicle 1: Passenger 2 should be removed (reached node 2)
        # - Vehicle 2: No change (vehicle is moving)
        expected_fleet_passengers = jnp.array(
            [
                [-1, 1, -1, -1],  # Passenger 0 removed
                [-1, -1, -1, -1],  # Passenger 2 removed
                [3, 4, -1, -1],  # No change
            ]
        )
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # Check passenger status updates:
        # - Passenger 0 should be COMPLETED (reached node 1)
        # - Passenger 2 should be COMPLETED (reached node 2)
        # - All others should remain IN_VEHICLE
        expected_passenger_statuses = jnp.array(
            [
                PassengerStatus.COMPLETED,  # Passenger 0 (reached node 1)
                PassengerStatus.IN_VEHICLE,  # Passenger 1 (going to node 2)
                PassengerStatus.COMPLETED,  # Passenger 2 (reached node 2)
                PassengerStatus.IN_VEHICLE,  # Passenger 3 (going to node 2)
                PassengerStatus.IN_VEHICLE,  # Passenger 4 (going to node 0)
            ]
        )
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_statuses)

        # Verify that other vehicle/passenger properties remain unchanged
        chex.assert_trees_all_equal(updated_state.fleet.times_on_edge, state.fleet.times_on_edge)
        chex.assert_trees_all_equal(updated_state.fleet.current_edges, state.fleet.current_edges)
        chex.assert_trees_all_equal(
            updated_state.passengers.time_in_vehicle, state.passengers.time_in_vehicle
        )

    def test_no_completed_passengers(self, simple_state: State) -> None:
        """Test when no passengers have reached their destination."""
        # Modify simple_state so passenger's destination doesn't match vehicle position
        state = replace(
            simple_state,
            passengers=replace(
                simple_state.passengers,
                destinations=jnp.array([0]),  # Change destination to node 0
            ),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check that no changes were made
        chex.assert_trees_all_equal(updated_state.fleet.passengers, state.fleet.passengers)
        chex.assert_trees_all_equal(updated_state.passengers.statuses, state.passengers.statuses)

    def test_vehicle_moving(self, simple_state: State) -> None:
        """Test that passengers aren't marked as completed when vehicle is moving."""
        # Modify simple_state so vehicle is moving (times_on_edge > 0)
        state = replace(
            simple_state,
            fleet=replace(
                simple_state.fleet,
                times_on_edge=jnp.array([1.0]),
            ),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check that no changes were made
        chex.assert_trees_all_equal(updated_state.fleet.passengers, state.fleet.passengers)
        chex.assert_trees_all_equal(updated_state.passengers.statuses, state.passengers.statuses)

    def test_empty_vehicles(self) -> None:
        """Test handling of empty vehicles with passengers in the system.
        Setup:
        - Two empty vehicles at different nodes
        - Five passengers in the system with different statuses
        Should verify that empty vehicles don't affect passenger statuses.
        """
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf],
                    [1.0, 0.0, 1.0],
                    [jnp.inf, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            is_terminal=jnp.array([True, False, True], dtype=bool),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1]),
            current_edges=jnp.array(
                [
                    [0, 1],  # Vehicle 0 at node 0
                    [1, 2],  # Vehicle 1 at node 1
                ]
            ),
            times_on_edge=jnp.array([0.0, 0.0]),  # Both at nodes
            passengers=jnp.array(
                [
                    [-1, -1, -1],  # Empty vehicle
                    [-1, -1, -1],  # Empty vehicle
                ]
            ),
            directions=jnp.array([0, 0]),
        )

        passengers = Passengers(
            origins=jnp.array([0, 1, 0, 1, 2]),
            destinations=jnp.array([1, 2, 2, 0, 1]),
            desired_departure_times=jnp.array([0.0, 0.0, 1.0, 1.0, 2.0]),
            time_waiting=jnp.array([5.0, 3.0, 0.0, 0.0, 0.0]),
            time_in_vehicle=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            statuses=jnp.array(
                [
                    PassengerStatus.WAITING,  # Passenger 0: waiting at node 0
                    PassengerStatus.WAITING,  # Passenger 1: waiting at node 1
                    PassengerStatus.NOT_IN_SYSTEM,  # Passenger 2: not yet in system
                    PassengerStatus.NOT_IN_SYSTEM,  # Passenger 3: not yet in system
                    PassengerStatus.NOT_IN_SYSTEM,  # Passenger 4: not yet in system
                ]
            ),
            has_transferred=jnp.array([False, False, False, False, False]),
            transfer_nodes=jnp.array([-1, -1, -1, -1, -1]),
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED, RouteType.FIXED]),
                stops=jnp.array(
                    [
                        [0, 1, 2, -1],  # Route: 0->1->2
                        [1, 2, 0, -1],  # Route: 1->2->0
                    ]
                ),
                frequencies=jnp.ones(2),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(2),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check that fleet remains unchanged (was already empty)
        chex.assert_trees_all_equal(updated_state.fleet.passengers, state.fleet.passengers)

        # Check that passenger statuses remain unchanged
        chex.assert_trees_all_equal(updated_state.passengers.statuses, state.passengers.statuses)

        # Verify that other properties remain unchanged
        chex.assert_trees_all_equal(updated_state.fleet.times_on_edge, state.fleet.times_on_edge)
        chex.assert_trees_all_equal(updated_state.fleet.current_edges, state.fleet.current_edges)
        chex.assert_trees_all_equal(
            updated_state.passengers.time_waiting, state.passengers.time_waiting
        )
        chex.assert_trees_all_equal(
            updated_state.passengers.time_in_vehicle, state.passengers.time_in_vehicle
        )

    def test_handle_transfers(self) -> None:
        """
        Test that passengers are correctly marked as TRANSFERRING when reaching their transfer node.
        """
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array(
                [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=jnp.float32
            ),
            is_terminal=jnp.array([True, False, True], dtype=bool),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[1, 1]]),  # Vehicle at node 1
            times_on_edge=jnp.array([0.0]),  # Vehicle is at node (not moving)
            passengers=jnp.array([[0]]),  # Passenger 0 in vehicle
            directions=jnp.array([0]),
        )

        passengers = Passengers(
            origins=jnp.array([0]),
            destinations=jnp.array([2]),  # Final destination is node 2
            desired_departure_times=jnp.array([0.0]),
            time_waiting=jnp.array([0.0]),
            time_in_vehicle=jnp.array([1.0]),
            statuses=jnp.array([PassengerStatus.IN_VEHICLE]),
            has_transferred=jnp.array([False]),
            transfer_nodes=jnp.array([1]),  # Transfer at node 1
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED]),
                stops=jnp.array([[0, 1, 2, -1]]),
                frequencies=jnp.ones(1),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(1),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check that passenger was removed from vehicle
        expected_fleet_passengers = jnp.array([[-1]])
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # Check that passenger status was updated to TRANSFERRING
        expected_passenger_status = jnp.array([PassengerStatus.TRANSFERRING])
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_status)

        # Check that has_transferred was updated
        expected_has_transferred = jnp.array([True])
        chex.assert_trees_all_equal(
            updated_state.passengers.has_transferred, expected_has_transferred
        )

    def test_multiple_passengers_with_transfers(self) -> None:
        """Test handling multiple passengers, some with transfers, some completing their journey.
        Setup:
        - Network with nodes 0,1,2,3 where:
          * Direct connections: 0<->1, 1<->2, 2<->3
          * No direct connection between 0<->2 or 1<->3 (requires transfer)
        - Two routes:
          * Route 1: 0<->1<->2
          * Route 2: 1<->2<->3
        This forces passengers going from 0->3 to transfer at node 1 or 2
        """
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf, jnp.inf],
                    [1.0, 0.0, 1.0, jnp.inf],
                    [jnp.inf, 1.0, 0.0, 1.0],
                    [jnp.inf, jnp.inf, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, False, True]),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1]),
            current_edges=jnp.array([[1, 1], [2, 2]]),  # Vehicles at nodes 1 and 2
            times_on_edge=jnp.array([0.0, 0.0]),
            passengers=jnp.array([[1, -1], [0, 2]]),  # 2 passengers in first vehicle, 1 in second
            directions=jnp.array([0, 0]),
        )

        passengers = Passengers(
            origins=jnp.array([0, 0, 0]),
            destinations=jnp.array([3, 1, 3]),  # Pass 1 completes at 1, others need transfer for 3
            desired_departure_times=jnp.array([0.0, 0.0, 0.0]),
            time_waiting=jnp.array([0.0, 0.0, 0.0]),
            time_in_vehicle=jnp.array([1.0, 1.0, 1.0]),
            statuses=jnp.array([PassengerStatus.IN_VEHICLE] * 3),
            has_transferred=jnp.array([False, False, False]),
            transfer_nodes=jnp.array([2, -1, 2]),  # Pass 0 and 2 transfer at node 2
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED, RouteType.FIXED]),
                stops=jnp.array(
                    [
                        [0, 1, 2, -1],  # Route 1: 0-1-2
                        [1, 2, 3, -1],  # Route 2: 1-2-3
                    ]
                ),
                frequencies=jnp.ones(2),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(2),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Check passenger statuses:
        # - Passenger 0 should be TRANSFERRING (at transfer node 2)
        # - Passenger 1 should be COMPLETED (reached destination 1)
        # - Passenger 2 should be TRANSFERRING (at transfer node 2)
        expected_statuses = jnp.array(
            [
                PassengerStatus.TRANSFERRING,
                PassengerStatus.COMPLETED,
                PassengerStatus.TRANSFERRING,
            ]
        )
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_statuses)

        # Check that all passengers were removed from vehicles
        expected_fleet_passengers = jnp.array([[-1, -1], [-1, -1]])
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # Check transfer flags
        expected_has_transferred = jnp.array([True, False, True])
        chex.assert_trees_all_equal(
            updated_state.passengers.has_transferred, expected_has_transferred
        )

    def test_no_double_transfers(self) -> None:
        """Test that passengers who have already transferred once can't transfer again."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array(
                [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=jnp.float32
            ),
            is_terminal=jnp.array([True, False, True], dtype=bool),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[1, 1]]),  # Vehicle at node 1
            times_on_edge=jnp.array([0.0]),
            passengers=jnp.array([[0]]),  # One passenger who has already transferred
            directions=jnp.array([0]),
        )

        passengers = Passengers(
            origins=jnp.array([0]),
            destinations=jnp.array([2]),
            desired_departure_times=jnp.array([0.0]),
            time_waiting=jnp.array([0.0]),
            time_in_vehicle=jnp.array([1.0]),
            statuses=jnp.array([PassengerStatus.IN_VEHICLE]),
            has_transferred=jnp.array([True]),  # Already made one transfer
            transfer_nodes=jnp.array([1]),  # Another transfer node (should be ignored)
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED]),
                stops=jnp.array([[0, 1, 2, -1]]),
                frequencies=jnp.ones(1),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(1),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = handle_completed_and_transferring_passengers(state)

        # Passenger should stay IN_VEHICLE since they can't transfer again
        expected_status = jnp.array([PassengerStatus.IN_VEHICLE])
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_status)

        # Passenger should still be in vehicle
        expected_fleet_passengers = jnp.array([[0]])
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # has_transferred should remain True
        expected_has_transferred = jnp.array([True])
        chex.assert_trees_all_equal(
            updated_state.passengers.has_transferred, expected_has_transferred
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

    @pytest.mark.skip
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

    @pytest.mark.skip
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
        chex.assert_equal(is_connected(sample_network, jnp.array(0), jnp.array(1)), True)
        chex.assert_equal(is_connected(sample_network, jnp.array(0), jnp.array(2)), False)
        chex.assert_equal(is_connected(sample_network, jnp.array(1), jnp.array(2)), True)

    @pytest.mark.skip
    def test_is_connected_invalid_indices(self, sample_network: NetworkData) -> None:
        # not sure what to do about it see
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        pass

    def test_get_travel_time(self, sample_network: NetworkData) -> None:
        chex.assert_equal(get_travel_time(sample_network, jnp.array(0), jnp.array(1)), 1.0)
        chex.assert_equal(get_travel_time(sample_network, jnp.array(0), jnp.array(2)), jnp.inf)
        chex.assert_equal(get_travel_time(sample_network, jnp.array(1), jnp.array(2)), 2.0)
        chex.assert_equal(get_travel_time(sample_network, jnp.array(0), jnp.array(0)), 0.0)

    @pytest.mark.skip
    def test_get_travel_time_invalid_indices(self, sample_network: NetworkData) -> None:
        # not sure what to do about it see
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        pass

    def test_self_connection(self, sample_network: NetworkData) -> None:
        chex.assert_equal(is_connected(sample_network, jnp.array(0), jnp.array(0)), True)
        chex.assert_equal(get_travel_time(sample_network, jnp.array(0), jnp.array(0)), 0.0)

    def test_non_connectivity(self, sample_network: NetworkData) -> None:
        chex.assert_equal(get_travel_time(sample_network, jnp.array(0), jnp.array(2)), jnp.inf)
        chex.assert_equal(is_connected(sample_network, jnp.array(0), jnp.array(2)), False)

    def test_single_node_network(self) -> None:
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0]], dtype=jnp.float32),
            travel_times=jnp.array([[0.0]], dtype=jnp.float32),
            is_terminal=jnp.array([True], dtype=bool),
        )

        from_node = jnp.array(0)
        to_node = jnp.array(0)
        chex.assert_equal(is_connected(network, from_node=from_node, to_node=to_node), True)
        chex.assert_equal(get_travel_time(network, from_node=from_node, to_node=to_node), 0.0)


class TestRouteBatch:
    @pytest.fixture
    def mixed_routes(self) -> RouteBatch:
        """Create a mixed batch of fixed and flexible routes, with empty initial stops."""
        num_routes = 5
        max_stops = 10
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
            stops=jnp.full((num_routes, max_stops), -1, dtype=int),  # All routes start empty
            frequencies=jnp.ones(num_routes),
            num_flex_routes=jnp.array(3),
            num_fix_routes=jnp.array(2),
        )

    @pytest.fixture
    def num_nodes(self) -> int:
        return 6

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
        valid_mask = get_valid_stops(mixed_routes)
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

    def test_update_routes_valid_actions(self, mixed_routes: RouteBatch, num_nodes: int) -> None:
        """Test updating routes with valid actions for all routes."""
        # Create actions for all routes (5 routes total)
        # Actions should be valid node indices (0 to num_nodes-1)
        actions = jnp.array([0, 2, 1, 3, 4])

        # Update routes with new stops

        updated_batch = update_routes(mixed_routes, jnp.array(num_nodes), actions)

        # Check that the first stop of each route has been updated
        chex.assert_trees_all_equal(updated_batch.stops[:, 0], actions)

        # Check that other properties remain unchanged
        chex.assert_trees_all_equal(updated_batch.types, mixed_routes.types)
        chex.assert_trees_all_equal(updated_batch.frequencies, mixed_routes.frequencies)
        chex.assert_trees_all_equal(updated_batch.num_flex_routes, mixed_routes.num_flex_routes)
        chex.assert_trees_all_equal(updated_batch.num_fix_routes, mixed_routes.num_fix_routes)

        # Check that other stops remain -1 (empty)
        chex.assert_trees_all_equal(
            updated_batch.stops[:, 1:],
            jnp.full((mixed_routes.stops.shape[0], mixed_routes.stops.shape[1] - 1), -1),
        )

    def test_update_routes_mixed_full_and_empty(
        self, mixed_routes: RouteBatch, num_nodes: int
    ) -> None:
        """Test updating routes with a mix of full, partially filled, and empty routes."""
        # Create routes with full, partial, and empty routes
        mixed_full_empty = replace(
            mixed_routes,
            stops=jnp.array(
                [
                    [1, 2, 3, 4, 5, 0, 1, 2, 3, 4],  # Full route
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Empty route
                    [3, 4, 5, 0, -1, -1, -1, -1, -1, -1],  # Partially filled route
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Empty route
                    [5, 0, 1, 2, 3, 4, 5, 0, 1, 2],  # Full route
                ]
            ),
        )

        # Mix of valid actions and no-ops
        actions = jnp.array([0, 1, num_nodes, 3, num_nodes])

        # Update routes with new stops
        updated_batch = update_routes(mixed_full_empty, jnp.array(num_nodes), actions)

        # Expected results:
        # - Full routes should remain unchanged
        # - Empty routes should be updated if action is not no-op
        # - Partially filled route should remain unchanged if action is no-op
        expected_stops = jnp.array(
            [
                [1, 2, 3, 4, 5, 0, 1, 2, 3, 4],  # Full route - unchanged
                [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Empty route - updated with action 1
                [3, 4, 5, 0, -1, -1, -1, -1, -1, -1],  # Partially filled - unchanged due to no-op
                [3, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Empty route - updated with action 3
                [5, 0, 1, 2, 3, 4, 5, 0, 1, 2],  # Full route - unchanged
            ]
        )

        # Check that routes are updated correctly
        chex.assert_trees_all_equal(updated_batch.stops, expected_stops)

        # Check that other properties remain unchanged
        chex.assert_trees_all_equal(updated_batch.types, mixed_full_empty.types)
        chex.assert_trees_all_equal(updated_batch.frequencies, mixed_full_empty.frequencies)
        chex.assert_trees_all_equal(updated_batch.num_flex_routes, mixed_full_empty.num_flex_routes)
        chex.assert_trees_all_equal(updated_batch.num_fix_routes, mixed_full_empty.num_fix_routes)

    @pytest.mark.skip
    def test_update_routes_empty_actions(self, mixed_routes: RouteBatch) -> None:
        pass

    @pytest.mark.skip
    def test_update_routes_mismatched_actions_length(self, mixed_routes: RouteBatch) -> None:
        pass

    @pytest.mark.skip
    def test_update_routes_invalid_action_values(
        self, mixed_routes: RouteBatch, num_nodes: int
    ) -> None:
        pass


class TestFleet:
    @pytest.fixture
    def fully_occupied_fleet(self) -> Fleet:
        return Fleet(
            route_ids=jnp.array([0], dtype=int),
            current_edges=jnp.array([[0, 1]], dtype=int),
            times_on_edge=jnp.array([15.0], dtype=float),
            passengers=jnp.array([[1, 2, 3, 4, 5]], dtype=int),  # max_capacity=5
            directions=jnp.array([0], dtype=int),
        )

    @pytest.fixture
    def partially_occupied_fleet(self) -> Fleet:
        return Fleet(
            route_ids=jnp.array([0, 1], dtype=int),
            current_edges=jnp.array([[0, 1], [1, 2]], dtype=int),
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
        updated_fleet = add_passenger(
            mixed_fleet, vehicle_id=jnp.array(0), passenger_id=jnp.array(10)
        )
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
        updated_fleet = remove_passenger(
            mixed_fleet, vehicle_id=jnp.array(2), passenger_id=jnp.array(4)
        )
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
        fleet_after_add = add_passenger(
            mixed_fleet, vehicle_id=jnp.array(0), passenger_id=jnp.array(10)
        )
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
        fleet_after_remove = remove_passenger(
            fleet_after_add, vehicle_id=jnp.array(0), passenger_id=jnp.array(10)
        )
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
            has_transferred=jnp.array([], dtype=bool),
            transfer_nodes=jnp.array([], dtype=int),
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
            has_transferred=jnp.array([False, False, False, False, False], dtype=bool),
            transfer_nodes=jnp.array([-1, -1, -1, -1, -1], dtype=int),
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
            has_transferred=jnp.array([False, False, False, False, False, False], dtype=bool),
            transfer_nodes=jnp.array([-1, -1, -1, -1, -1, -1], dtype=int),
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
            has_transferred=jnp.array([False, False, False], dtype=bool),
            transfer_nodes=jnp.array([-1, -1, -1], dtype=int),
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
        updated_state = increment_wait_times(all_waiting_passengers)
        expected_time_waiting = (
            all_waiting_passengers.time_waiting + 1.0
        )  # Assuming increment by 1.0
        assert (updated_state.time_waiting == expected_time_waiting).all()

    def test_increment_wait_times_mixed(self, mixed_passenger_state: Passengers) -> None:
        updated_state = increment_wait_times(mixed_passenger_state)
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
        updated_state = increment_in_vehicle_times(all_in_vehicle_state)
        expected_time_in_vehicle = (
            all_in_vehicle_state.time_in_vehicle + 1.0
        )  # Assuming increment by 1.0
        assert (updated_state.time_in_vehicle == expected_time_in_vehicle).all()

    def test_increment_in_vehicle_times_mixed(self, mixed_passenger_state: Passengers) -> None:
        updated_state = increment_in_vehicle_times(mixed_passenger_state)
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
            has_transferred=jnp.array([False, False, False, False, False], dtype=bool),
            transfer_nodes=jnp.array([-1, -1, -1, -1, -1], dtype=int),
        )

    def test_update_passengers_not_in_system_to_waiting(
        self, varied_passenger_state: Passengers
    ) -> None:
        """
        Only passenger 0 has desired_departure_times == 3.0, so at current_time=3.0,
        passenger 0 should move from NOT_IN_SYSTEM to WAITING.
        """
        current_time = jnp.array(3.0)

        updated_ps = update_passengers_to_waiting(
            varied_passenger_state,
            current_time=current_time,
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


class TestState:
    @pytest.fixture
    def sample_state(self) -> State:
        # Create a simple network with 3 nodes
        network = NetworkData(
            node_coordinates=jnp.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 3.0],
                    [2.0, 3.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, False]),
        )

        # Create a simple fleet
        fleet = Fleet(
            route_ids=jnp.array([0], dtype=int),
            current_edges=jnp.array([[0, 1]], dtype=int),
            times_on_edge=jnp.array([0.0], dtype=float),
            passengers=jnp.array([[-1, -1]], dtype=int),
            directions=jnp.array([0], dtype=int),
        )

        # Create routes
        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FLEXIBLE]),
            stops=jnp.array(
                [
                    [0, 1, 2, -1],  # Route 1: 0->1->2
                    [2, 1, 0, -1],  # Route 2: 2->1->0
                ]
            ),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(1),
            num_fix_routes=jnp.array(1),
        )

        # Create empty passenger state
        passengers = Passengers(
            origins=jnp.array([], dtype=int),
            destinations=jnp.array([], dtype=int),
            desired_departure_times=jnp.array([], dtype=float),
            time_waiting=jnp.array([], dtype=float),
            time_in_vehicle=jnp.array([], dtype=float),
            statuses=jnp.array([], dtype=int),
            has_transferred=jnp.array([], dtype=bool),
            transfer_nodes=jnp.array([], dtype=int),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    def test_calculate_route_times(self, sample_state: State) -> None:
        route_times, _ = calculate_route_times(sample_state)

        # Check shape
        chex.assert_shape(route_times, (2, 3, 3))

        # Check first route (0->1->2)
        chex.assert_equal(route_times[0, 0, 1], jnp.array(1.0))  # Time 0->1
        chex.assert_equal(route_times[0, 0, 2], jnp.array(4.0))  # Time 0->2 (via 1)
        chex.assert_equal(route_times[0, 1, 2], jnp.array(3.0))  # Time 0->1

        # Check second route (2->1->0)
        chex.assert_equal(route_times[1, 2, 1], jnp.array(3.0))  # Time 2->1
        chex.assert_equal(route_times[1, 2, 0], jnp.array(4.0))  # Time 2->0 (via 1)
        chex.assert_equal(route_times[1, 1, 0], jnp.array(1.0))  # Time 1->0

        # Check diagonal elements (should be 0)
        chex.assert_trees_all_equal(jnp.diagonal(route_times[0]), jnp.zeros(3))
        chex.assert_trees_all_equal(jnp.diagonal(route_times[1]), jnp.zeros(3))

        # Check symmetry on fixed routes
        chex.assert_trees_all_equal(route_times[0, 0, 1], route_times[0, 1, 0])
        chex.assert_trees_all_equal(route_times[0, 0, 2], route_times[0, 2, 0])
        chex.assert_trees_all_equal(route_times[0, 1, 2], route_times[0, 2, 1])

        # TODO: check second return value

    def test_calculate_route_times_nodes_not_in_routes(self) -> None:
        """Test that route times to nodes that are connected in the network but not included in any
        route are infinity."""
        # Create network where all nodes are connected
        network = NetworkData(
            node_coordinates=jnp.array(
                [
                    [0.0, 0.0],  # node 0
                    [1.0, 0.0],  # node 1
                    [2.0, 0.0],  # node 2 (connected but not in routes)
                ]
            ),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, True]),
        )

        # Routes only use nodes 0 and 1, never visit node 2
        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FLEXIBLE]),
            stops=jnp.array(
                [
                    [0, 1, -1],  # Fixed route between nodes 0 and 1
                    [0, 1, -1],  # Flexible route between nodes 0 and 1
                ]
            ),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(1),
            num_fix_routes=jnp.array(1),
        )

        state = State(
            network=network,
            fleet=Fleet(
                route_ids=jnp.array([0, 1]),
                current_edges=jnp.array([[0, 1], [0, 1]]),
                times_on_edge=jnp.array([0.0, 0.0]),
                passengers=jnp.array([[-1, -1], [-1, -1]]),
                directions=jnp.array([0, 0]),
            ),
            passengers=Passengers(
                origins=jnp.array([], dtype=jnp.int32),
                destinations=jnp.array([], dtype=jnp.int32),
                desired_departure_times=jnp.array([], dtype=jnp.float32),
                time_waiting=jnp.array([], dtype=jnp.float32),
                time_in_vehicle=jnp.array([], dtype=jnp.float32),
                statuses=jnp.array([], dtype=jnp.int32),
                has_transferred=jnp.array([], dtype=bool),
                transfer_nodes=jnp.array([], dtype=jnp.int32),
            ),
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        # Calculate route times
        route_times, route_directions = calculate_route_times(state)

        # Fixed route (index 0): Vehicle can turn around
        expected_times_fixed = jnp.array(
            [
                [0.0, 1.0, jnp.inf],  # from 0: can reach 0->1 but not 2
                [1.0, 0.0, jnp.inf],  # from 1: can reach 1->0 (by turning around) but not 2
                [jnp.inf, jnp.inf, 0.0],  # from 2: can't reach 0 or 1, only 2->2
            ]
        )

        # Flexible route (index 1): Vehicle only goes forward
        expected_times_flexible = jnp.array(
            [
                [0.0, 1.0, jnp.inf],  # from 0: can reach 0->1 but not 2
                [jnp.inf, 0.0, jnp.inf],  # from 1: can't go back to 0, can't reach 2
                [jnp.inf, jnp.inf, 0.0],  # from 2: can't reach 0 or 1, only 2->2
            ]
        )

        # Check each route type has correct times
        chex.assert_trees_all_equal(route_times[0], expected_times_fixed)
        chex.assert_trees_all_equal(route_times[1], expected_times_flexible)


class TestFleetMovement:
    @pytest.fixture
    def basic_state(self) -> State:
        """Create a simple state with one vehicle on a fixed route with predictable travel times"""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            # First edge takes 2 time steps, second edge takes 1 time step
            travel_times=jnp.array(
                [
                    [0.0, 2.0, jnp.inf],
                    [2.0, 0.0, 1.0],
                    [jnp.inf, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, True]),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[0, 1]]),  # Vehicle starts on first edge (0->1)
            times_on_edge=jnp.array([0.0]),  # Just starting on the edge
            passengers=jnp.array([[-1]]),  # Empty vehicle
            directions=jnp.array([0]),  # Forward direction
        )

        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED]),
            stops=jnp.array([[0, 1, 2]]),  # Simple route: 0->1->2
            frequencies=jnp.ones(1),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(1),
        )

        passengers = Passengers(
            origins=jnp.array([], dtype=int),
            destinations=jnp.array([], dtype=int),
            desired_departure_times=jnp.array([], dtype=float),
            time_waiting=jnp.array([], dtype=float),
            time_in_vehicle=jnp.array([], dtype=float),
            statuses=jnp.array([], dtype=int),
            has_transferred=jnp.array([], dtype=bool),
            transfer_nodes=jnp.array([], dtype=int),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    def test_increment_times_on_edge(self, basic_state: State) -> None:
        """Test that times_on_edge increases by 1"""
        updated_state = move_vehicles(basic_state)
        chex.assert_trees_all_equal(updated_state.fleet.times_on_edge, jnp.array([1.0]))

    def test_vehicle_movement_on_long_edge(self, basic_state: State) -> None:
        """Test vehicle movement on edge with travel time of 2"""
        # First step
        state1 = move_vehicles(basic_state)
        chex.assert_trees_all_equal(state1.fleet.times_on_edge, jnp.array([1.0]))
        chex.assert_trees_all_equal(
            state1.fleet.current_edges, jnp.array([[0, 1]])
        )  # Still on first edge

        state2 = move_vehicles(state1)
        state3 = move_vehicles(state2)
        chex.assert_trees_all_equal(state3.fleet.times_on_edge, jnp.array([0.0]))
        chex.assert_trees_all_equal(
            state3.fleet.current_edges, jnp.array([[2, 1]])
        )  # Turn around at end
        chex.assert_trees_all_equal(state3.fleet.directions, jnp.array([1]))  # Direction reversed

    @pytest.fixture
    def flex_route_state(self, basic_state: State) -> State:
        """Create a state with one vehicle on a flexible route"""
        basic_state = replace(
            basic_state,
            routes=replace(
                basic_state.routes,
                types=jnp.array([RouteType.FLEXIBLE]),
                num_flex_routes=jnp.array(1),
                num_fix_routes=jnp.array(0),
            ),
        )
        return basic_state

    def test_flex_route_stops_at_end(self, flex_route_state: State) -> None:
        """Test that flexible route vehicle stops when reaching final node"""
        # Position vehicle at last edge and set time to complete it
        flex_route_state = replace(
            flex_route_state,
            fleet=replace(
                flex_route_state.fleet,
                current_edges=jnp.array([[1, 2]]),
                times_on_edge=jnp.array([1.0]),
            ),
        )
        updated_state = move_vehicles(flex_route_state)
        # Should stay at node 2
        chex.assert_trees_all_equal(updated_state.fleet.current_edges, jnp.array([[2, 2]]))
        chex.assert_trees_all_equal(updated_state.fleet.times_on_edge, jnp.array([0.0]))

    def test_fixed_route_turns_around(self, basic_state: State) -> None:
        """Test that fixed route vehicle turns around at the end"""
        # Position vehicle at last edge and set time to complete it
        basic_state = replace(
            basic_state,
            fleet=replace(
                basic_state.fleet,
                current_edges=jnp.array([[1, 2]]),
                times_on_edge=jnp.array([1.0]),
            ),
        )
        updated_state = move_vehicles(basic_state)
        # Should start moving back (2->1)
        chex.assert_trees_all_equal(updated_state.fleet.current_edges, jnp.array([[2, 1]]))
        chex.assert_trees_all_equal(updated_state.fleet.times_on_edge, jnp.array([0.0]))
        # Direction should be reversed
        chex.assert_trees_all_equal(updated_state.fleet.directions, jnp.array([1]))


class TestCalculateTotalTravelTimes:
    @pytest.fixture
    def linear_network_state(self) -> State:
        """Create a simple linear network (0--1--2) with fixed travel times."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf],
                    [1.0, 0.0, 1.0],
                    [jnp.inf, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, True]),
        )

        # Route: 0->1->2 (fixed route)
        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED]),
            stops=jnp.array([[0, 1, 2, -1]]),
            frequencies=jnp.ones(1),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(1),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[0, 1]]),  # Vehicle starts on edge 0->1
            times_on_edge=jnp.array([0.0]),
            passengers=jnp.array([[-1, -1]]),  # Empty vehicle with capacity 2
            directions=jnp.array([0]),  # Forward direction
        )

        passengers = Passengers(
            origins=jnp.array([], dtype=jnp.int32),
            destinations=jnp.array([], dtype=jnp.int32),
            desired_departure_times=jnp.array([], dtype=jnp.float32),
            time_waiting=jnp.array([], dtype=jnp.float32),
            time_in_vehicle=jnp.array([], dtype=jnp.float32),
            statuses=jnp.array([], dtype=jnp.int32),
            has_transferred=jnp.array([], dtype=bool),
            transfer_nodes=jnp.array([], dtype=jnp.int32),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    @pytest.fixture
    def circular_network_state(self) -> State:
        """Create a circular network (0->1->2->0) with repeated stops."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, True]),
        )

        # Route: 0->1->2->0->1 (fixed route with repeated stops)
        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED]),
            stops=jnp.array([[0, 1, 2, 0, 1, -1]]),
            frequencies=jnp.ones(1),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(1),
        )

        fleet = Fleet(
            route_ids=jnp.array([0]),
            current_edges=jnp.array([[0, 1]]),
            times_on_edge=jnp.array([0.0]),
            passengers=jnp.array([[-1, -1]]),
            directions=jnp.array([0]),
        )

        passengers = Passengers(
            origins=jnp.array([], dtype=jnp.int32),
            destinations=jnp.array([], dtype=jnp.int32),
            desired_departure_times=jnp.array([], dtype=jnp.float32),
            time_waiting=jnp.array([], dtype=jnp.float32),
            time_in_vehicle=jnp.array([], dtype=jnp.float32),
            statuses=jnp.array([], dtype=jnp.int32),
            has_transferred=jnp.array([], dtype=bool),
            transfer_nodes=jnp.array([], dtype=jnp.int32),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    def test_forward_journey(self, linear_network_state: State) -> None:
        """Test journey time calculation in forward direction."""
        route_times, route_directions = calculate_route_times(linear_network_state)

        journey_times = calculate_journey_times(
            linear_network_state,
            passenger_origin=jnp.array(1),
            passenger_dest=jnp.array(2),
            route_times=route_times,
            route_directions=route_directions,
        )
        # Vehicle at edge 0->1, needs 1 time unit to reach node 1,
        # then 1 time unit to reach node 2
        expected_times = jnp.array([2.0])
        chex.assert_trees_all_equal(journey_times, expected_times)

    def test_backward_journey(self, linear_network_state: State) -> None:
        """Test journey time calculation when vehicle needs to go backward."""
        # Put vehicle at node 2 going backward
        state = replace(
            linear_network_state,
            fleet=replace(
                linear_network_state.fleet,
                current_edges=jnp.array([[2, 1]]),
                directions=jnp.array([1]),  # Backward direction
            ),
        )

        route_times, route_directions = calculate_route_times(state)
        journey_times = calculate_journey_times(
            state,
            passenger_origin=jnp.array(0),
            passenger_dest=jnp.array(1),
            route_times=route_times,
            route_directions=route_directions,
        )
        # Vehicle needs to go 2->1->0->1
        expected_times = jnp.array([3.0])  # 1 unit to 1, 1 to 0, 1 back to 1
        chex.assert_trees_all_equal(journey_times, expected_times)

    @pytest.mark.skip
    def test_repeated_stops(self, circular_network_state: State) -> None:
        """Test journey times with repeated stops in route."""
        # Test from first occurrence of stop 1
        route_times, route_directions = calculate_route_times(circular_network_state)
        journey_times = calculate_journey_times(
            circular_network_state,
            passenger_origin=jnp.array(1),
            passenger_dest=jnp.array(2),
            route_times=route_times,
            route_directions=route_directions,
        )
        # Vehicle at 0->1, needs 1 time unit to 1, then 1 to 2
        expected_times = jnp.array([2.0])
        chex.assert_trees_all_equal(journey_times, expected_times)

        # Test from second occurrence of stop 1
        state = replace(
            circular_network_state,
            fleet=replace(
                circular_network_state.fleet,
                current_edges=jnp.array([[2, 0]]),
                directions=jnp.array([0]),
            ),
        )

        route_times, route_directions = calculate_route_times(state)
        journey_times = calculate_journey_times(
            state,
            passenger_origin=jnp.array(1),
            passenger_dest=jnp.array(0),
            route_times=route_times,
            route_directions=route_directions,
        )
        # Vehicle needs to go 2->0->1->0
        expected_times = jnp.array([4.0])
        chex.assert_trees_all_equal(journey_times, expected_times)

    def test_flexible_route(self, linear_network_state: State) -> None:
        """Test journey times for flexible routes."""
        # Convert route to flexible
        state = replace(
            linear_network_state,
            routes=replace(
                linear_network_state.routes,
                types=jnp.array([RouteType.FLEXIBLE]),
                num_flex_routes=jnp.array(1),
                num_fix_routes=jnp.array(0),
            ),
        )

        # Test forward journey
        route_times, route_directions = calculate_route_times(state)
        journey_times = calculate_journey_times(
            state,
            passenger_origin=jnp.array(1),
            passenger_dest=jnp.array(2),
            route_times=route_times,
            route_directions=route_directions,
        )
        expected_times = jnp.array([2.0])  # Same as fixed route
        chex.assert_trees_all_equal(journey_times, expected_times)

        # Test "backward" journey (flexible routes don't turn around)
        state = replace(
            state,
            fleet=replace(
                state.fleet,
                current_edges=jnp.array([[2, 1]]),
                directions=jnp.array([1]),
            ),
        )
        route_times, route_directions = calculate_route_times(state)
        journey_times = calculate_journey_times(
            state,
            passenger_origin=jnp.array(0),
            passenger_dest=jnp.array(1),
            route_times=route_times,
            route_directions=route_directions,
        )
        # Should return inf as flexible routes don't turn around
        expected_times = jnp.array([jnp.inf])
        chex.assert_trees_all_equal(journey_times, expected_times)

    def test_multiple_vehicles(self) -> None:
        """Test journey times calculation with multiple vehicles."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf],
                    [1.0, 0.0, 1.0],
                    [jnp.inf, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, True]),
        )

        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FLEXIBLE]),
            stops=jnp.array(
                [
                    [0, 1, 2, -1],  # Fixed route
                    [1, 2, -1, -1],  # Flexible route
                ]
            ),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(1),
            num_fix_routes=jnp.array(1),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1]),
            current_edges=jnp.array([[0, 1], [1, 2]]),
            times_on_edge=jnp.array([0.0, 0.0]),
            passengers=jnp.array([[-1, -1], [-1, -1]]),
            directions=jnp.array([0, 0]),
        )

        state = State(
            network=network,
            fleet=fleet,
            passengers=Passengers(
                origins=jnp.array([], dtype=jnp.int32),
                destinations=jnp.array([], dtype=jnp.int32),
                desired_departure_times=jnp.array([], dtype=jnp.float32),
                time_waiting=jnp.array([], dtype=jnp.float32),
                time_in_vehicle=jnp.array([], dtype=jnp.float32),
                statuses=jnp.array([], dtype=jnp.int32),
                has_transferred=jnp.array([], dtype=bool),
                transfer_nodes=jnp.array([], dtype=jnp.int32),
            ),
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        route_times, route_directions = calculate_route_times(state)
        journey_times = calculate_journey_times(
            state,
            passenger_origin=jnp.array(1),
            passenger_dest=jnp.array(2),
            route_times=route_times,
            route_directions=route_directions,
        )
        # First vehicle: 1 time unit to node 1, then 1 to node 2
        # Second vehicle: Already at node 1, 1 time unit to node 2
        expected_times = jnp.array([2.0, 1.0])
        chex.assert_trees_all_equal(journey_times, expected_times)


class TestAssignPassengers:
    @pytest.fixture
    def basic_state(self) -> State:
        """Create a simple state with two vehicles and some waiting passengers."""
        network = NetworkData(
            node_coordinates=jnp.array([[0, 0], [1, 0], [2, 0]], dtype=jnp.float32),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf],
                    [1.0, 0.0, 1.0],
                    [jnp.inf, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            is_terminal=jnp.array([True, False, True], dtype=bool),
        )

        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FLEXIBLE], dtype=int),
            stops=jnp.array(
                [
                    [0, 1, 2, -1, -1, -1],  # Fixed route: 0->1->2
                    [1, 2, -1, -1, -1, -1],  # Flexible route: 1->2
                ],
                dtype=int,
            ),
            frequencies=jnp.ones(2, dtype=jnp.float32),
            num_flex_routes=jnp.array(1),
            num_fix_routes=jnp.array(1),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1], dtype=int),
            current_edges=jnp.array(
                [[0, 1], [1, 2]], dtype=int
            ),  # Vehicle 0: 0->1, Vehicle 1: 1->2
            times_on_edge=jnp.array([0.0, 0.0], dtype=jnp.float32),
            passengers=jnp.array([[-1, -1], [-1, -1]], dtype=int),  # Both vehicles empty
            directions=jnp.array([VehicleDirection.FORWARD, VehicleDirection.FORWARD], dtype=int),
        )

        passengers = Passengers(
            origins=jnp.array([1, 1, 0], dtype=int),
            destinations=jnp.array([2, 2, 2], dtype=int),
            desired_departure_times=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
            time_waiting=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            time_in_vehicle=jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
            statuses=jnp.array(
                [
                    PassengerStatus.WAITING,
                    PassengerStatus.WAITING,
                    PassengerStatus.NOT_IN_SYSTEM,
                ],
                dtype=int,
            ),
            has_transferred=jnp.array([False, False, False], dtype=bool),
            transfer_nodes=jnp.array([-1, -1, -1], dtype=int),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=passengers,
            routes=routes,
            current_time=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(0),
        )

    def test_immediate_assignment(self, basic_state: State) -> None:
        """Test that passengers get assigned to vehicles at their location."""
        updated_state = assign_passengers(basic_state)

        # Expect Vehicle 1 (at node 1->2) to pick up passengers 0 and 1
        expected_fleet_passengers = jnp.array(
            [
                [-1, -1],  # Vehicle 0 remains empty
                [0, 1],  # Vehicle 1 picks up passengers 0 and 1
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # Passenger 2 remains NOT_IN_SYSTEM
        expected_passenger_statuses = jnp.array(
            [
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.IN_VEHICLE,
                PassengerStatus.NOT_IN_SYSTEM,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_statuses)

    def test_wait_for_better_option(self, basic_state: State) -> None:
        """Test that passengers wait if a better option is coming soon."""
        # Setup:
        # Network with two different paths between nodes:
        # - Direct fast route: 1->2 (time: 1)
        # - Slow route: 1->3->4->2 (time: 3)
        state = State(
            network=NetworkData(
                node_coordinates=jnp.array(
                    [
                        [0.0, 0.0],  # node 0
                        [1.0, 0.0],  # node 1
                        [2.0, 0.0],  # node 2
                        [1.0, 1.0],  # node 3
                        [2.0, 1.0],  # node 4
                    ]
                ),
                travel_times=jnp.array(
                    [
                        [0.0, 1.0, jnp.inf, jnp.inf, jnp.inf],
                        [1.0, 0.0, 1.0, 1.0, jnp.inf],
                        [jnp.inf, 1.0, 0.0, jnp.inf, jnp.inf],
                        [jnp.inf, 1.0, jnp.inf, 0.0, 1.0],
                        [jnp.inf, jnp.inf, 1.0, 1.0, 0.0],
                    ]
                ),
                is_terminal=jnp.array([True, False, True, False, False]),
            ),
            fleet=Fleet(
                route_ids=jnp.array([0, 1]),  # Two different routes
                current_edges=jnp.array(
                    [
                        [0, 1],  # Fast vehicle: at node 0, will reach 1 soon
                        [1, 3],  # Slow vehicle: at node 1, taking longer route
                    ]
                ),
                times_on_edge=jnp.array([0.0, 0.0]),  # Fast vehicle almost at node 1
                passengers=jnp.array([[-1, -1], [-1, -1]]),
                directions=jnp.array(
                    [
                        VehicleDirection.FORWARD,
                        VehicleDirection.FORWARD,
                    ]
                ),
            ),
            routes=RouteBatch(
                types=jnp.array([RouteType.FIXED, RouteType.FIXED]),
                stops=jnp.array(
                    [
                        [0, 1, 2, -1, -1, -1],  # Fast route
                        [0, 1, 3, 4, 2, -1],  # Slow route
                    ]
                ),
                frequencies=jnp.ones(2),
                num_flex_routes=jnp.array(0),
                num_fix_routes=jnp.array(2),
            ),
            passengers=Passengers(
                origins=jnp.array([1], dtype=jnp.int32),
                destinations=jnp.array([2], dtype=jnp.int32),
                desired_departure_times=jnp.array([0.0], dtype=jnp.float32),
                time_waiting=jnp.array([0.0], dtype=jnp.float32),
                time_in_vehicle=jnp.array([-1.0], dtype=jnp.float32),
                statuses=jnp.array([PassengerStatus.WAITING], dtype=jnp.int32),
                has_transferred=jnp.array([False], dtype=bool),
                transfer_nodes=jnp.array([-1], dtype=jnp.int32),
            ),
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        updated_state = assign_passengers(state)

        # Passenger should wait for fast vehicle rather than take slow vehicle
        expected_fleet_passengers = jnp.array(
            [
                [-1, -1],  # Fast vehicle (passenger will wait for this)
                [-1, -1],  # Slow vehicle (immediate but not optimal)
            ]
        )
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        expected_passenger_statuses = jnp.array([PassengerStatus.WAITING])
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_statuses)

    def test_capacity_constraints(self, basic_state: State) -> None:
        """Test that assignments respect vehicle capacity."""
        state = replace(
            basic_state,
            fleet=replace(
                basic_state.fleet,
                passengers=jnp.array(
                    [
                        [3, 4],  # First vehicle full with passengers 3,4
                        [5, -1],  # Second vehicle has one free seat, passenger 5 onboard
                    ]
                ),
            ),
            passengers=Passengers(
                origins=jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32),
                destinations=jnp.array([2, 2, 1, 1, 2], dtype=jnp.int32),
                desired_departure_times=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
                time_waiting=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
                time_in_vehicle=jnp.array([-1.0, -1.0, 20.0, 20.0, 10.0], dtype=jnp.float32),
                statuses=jnp.array(
                    [
                        PassengerStatus.WAITING,  # passenger 0: waiting at node 1
                        PassengerStatus.WAITING,  # passenger 1: waiting at node 1
                        PassengerStatus.IN_VEHICLE,  # passenger 3: in first vehicle
                        PassengerStatus.IN_VEHICLE,  # passenger 4: in first vehicle
                        PassengerStatus.IN_VEHICLE,  # passenger 5: in second vehicle
                    ],
                    dtype=jnp.int32,
                ),
                has_transferred=jnp.array([False, False, False, False, False], dtype=bool),
                transfer_nodes=jnp.array([-1, -1, -1, -1, -1], dtype=int),
            ),
        )

        updated_state = assign_passengers(state)

        # Second vehicle should pick up one passenger (only has one free seat)
        expected_fleet_passengers = jnp.array(
            [
                [3, 4],  # First vehicle still full
                [5, 0],  # Second vehicle picks up passenger 0 in its free seat
            ]
        )
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # One passenger gets picked up, one remains waiting
        expected_passenger_statuses = jnp.array(
            [
                PassengerStatus.IN_VEHICLE,  # got picked up
                PassengerStatus.WAITING,  # still waiting (no space)
                PassengerStatus.IN_VEHICLE,  # unchanged
                PassengerStatus.IN_VEHICLE,  # unchanged
                PassengerStatus.IN_VEHICLE,  # unchanged
            ]
        )
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_statuses)

    def test_no_valid_assignments(self, basic_state: State) -> None:
        """Test behavior when no valid assignments are possible."""
        # Create a network where node 2 is unreachable from nodes 0 and 1
        state = replace(
            basic_state,
            network=replace(
                basic_state.network,
                travel_times=jnp.array(
                    [
                        [0.0, 1.0, jnp.inf],
                        [1.0, 0.0, jnp.inf],
                        [jnp.inf, jnp.inf, 0.0],
                    ],
                    dtype=jnp.float32,
                ),
            ),
            # Set all passenger destinations to node 2 (which is unreachable)
            passengers=replace(
                basic_state.passengers,
                destinations=jnp.array([2, 2, 2], dtype=int),
            ),
        )

        updated_state = assign_passengers(state)

        # No passengers should be assigned since their destination is unreachable
        expected_fleet_passengers = jnp.array(
            [
                [-1, -1],
                [-1, -1],
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_state.fleet.passengers, expected_fleet_passengers)

        # All passengers should remain WAITING
        expected_passenger_statuses = jnp.array(
            [
                PassengerStatus.WAITING,
                PassengerStatus.WAITING,
                PassengerStatus.NOT_IN_SYSTEM,
            ],
            dtype=int,
        )
        chex.assert_trees_all_equal(updated_state.passengers.statuses, expected_passenger_statuses)


class TestFindBestTransferRoute:
    @pytest.fixture
    def basic_network_state(self) -> State:
        """Create a simple network with 4 nodes in a line: 0-1-2-3."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, jnp.inf, jnp.inf],
                    [1.0, 0.0, 1.0, jnp.inf],
                    [jnp.inf, 1.0, 0.0, 1.0],
                    [jnp.inf, jnp.inf, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, False, True]),
        )

        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FIXED]),
            stops=jnp.array(
                [
                    [0, 1, -1, -1],  # Route 0: 0-1
                    [1, 2, 3, -1],  # Route 1: 1-2-3
                ]
            ),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(2),
        )

        fleet = Fleet(
            route_ids=jnp.array([0, 1]),
            current_edges=jnp.array([[0, 1], [1, 2]]),
            times_on_edge=jnp.array([0.0, 0.0]),
            passengers=jnp.array([[-1, -1], [-1, -1]]),
            directions=jnp.array([0, 0]),
        )

        return State(
            network=network,
            fleet=fleet,
            passengers=Passengers(
                origins=jnp.array([], dtype=jnp.int32),
                destinations=jnp.array([], dtype=jnp.int32),
                desired_departure_times=jnp.array([], dtype=jnp.float32),
                time_waiting=jnp.array([], dtype=jnp.float32),
                time_in_vehicle=jnp.array([], dtype=jnp.float32),
                statuses=jnp.array([], dtype=jnp.int32),
                has_transferred=jnp.array([], dtype=bool),
                transfer_nodes=jnp.array([], dtype=jnp.int32),
            ),
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

    def test_basic_transfer(self, basic_network_state: State) -> None:
        """Test basic transfer between two fixed routes."""
        route_times, _ = calculate_route_times(basic_network_state)
        best_time, transfer_node, first_leg_route, second_leg_route = find_best_transfer_route(
            basic_network_state, jnp.array(0), jnp.array(3), route_times
        )

        # Expected: Transfer at node 1 from route 0 to route 1
        # Time: 1 (0->1) + 2 (1->3) = 3
        chex.assert_trees_all_equal(best_time, jnp.array(3.0))
        chex.assert_trees_all_equal(transfer_node, jnp.array(1))
        chex.assert_trees_all_equal(first_leg_route, jnp.array(0))  # Route 0 for first leg
        chex.assert_trees_all_equal(second_leg_route, jnp.array(1))  # Route 1 for second leg

    def test_with_flex_routes(self, basic_network_state: State) -> None:
        """Test that flex routes are ignored for transfer planning."""
        # Convert route 1 to flexible
        state = replace(
            basic_network_state,
            routes=replace(
                basic_network_state.routes,
                types=jnp.array([RouteType.FIXED, RouteType.FLEXIBLE]),
                num_flex_routes=jnp.array(1),
                num_fix_routes=jnp.array(1),
            ),
        )

        route_times, _ = calculate_route_times(state)
        best_time, transfer_node, first_leg_route, second_leg_route = find_best_transfer_route(
            state, jnp.array(0), jnp.array(3), route_times
        )

        # Should return inf since no valid transfer possible (flex route ignored)
        chex.assert_trees_all_equal(best_time, jnp.array(jnp.inf))
        chex.assert_trees_all_equal(transfer_node, jnp.array(-1))
        chex.assert_trees_all_equal(first_leg_route, jnp.array(-1))
        chex.assert_trees_all_equal(second_leg_route, jnp.array(-1))

    def test_multiple_transfer_options(self) -> None:
        """Test scenario with multiple possible transfer points."""
        network = NetworkData(
            node_coordinates=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            travel_times=jnp.array(
                [
                    [0.0, 1.0, 2.0, jnp.inf],
                    [1.0, 0.0, 1.0, 2.0],
                    [2.0, 1.0, 0.0, 1.0],
                    [jnp.inf, 2.0, 1.0, 0.0],
                ]
            ),
            is_terminal=jnp.array([True, False, False, True]),
        )

        # Two overlapping fixed routes
        routes = RouteBatch(
            types=jnp.array([RouteType.FIXED, RouteType.FIXED]),
            stops=jnp.array(
                [
                    [0, 1, 2, -1],  # Route 0: 0-1-2
                    [1, 2, 3, -1],  # Route 1: 1-2-3
                ]
            ),
            frequencies=jnp.ones(2),
            num_flex_routes=jnp.array(0),
            num_fix_routes=jnp.array(2),
        )

        state = State(
            network=network,
            fleet=Fleet(
                route_ids=jnp.array([0, 1]),
                current_edges=jnp.array([[0, 1], [1, 2]]),
                times_on_edge=jnp.array([0.0, 0.0]),
                passengers=jnp.array([[-1, -1], [-1, -1]]),
                directions=jnp.array([0, 0]),
            ),
            passengers=Passengers(
                origins=jnp.array([], dtype=jnp.int32),
                destinations=jnp.array([], dtype=jnp.int32),
                desired_departure_times=jnp.array([], dtype=jnp.float32),
                time_waiting=jnp.array([], dtype=jnp.float32),
                time_in_vehicle=jnp.array([], dtype=jnp.float32),
                statuses=jnp.array([], dtype=jnp.int32),
                has_transferred=jnp.array([], dtype=bool),
                transfer_nodes=jnp.array([], dtype=jnp.int32),
            ),
            routes=routes,
            current_time=jnp.array(0.0),
            key=jax.random.PRNGKey(0),
        )

        route_times, _ = calculate_route_times(state)
        best_time, transfer_node, first_leg_route, second_leg_route = find_best_transfer_route(
            state, jnp.array(0), jnp.array(3), route_times
        )

        # Could transfer at node 1 or 2, should choose node 1 (earlier transfer)
        # Time: 1 (0->1) + 2 (1->3) = 3
        chex.assert_trees_all_equal(best_time, jnp.array(3.0))
        chex.assert_trees_all_equal(transfer_node, jnp.array(1))  # Transfer at node 1
        chex.assert_trees_all_equal(first_leg_route, jnp.array(0))  # First leg on route 0
        chex.assert_trees_all_equal(second_leg_route, jnp.array(1))  # Second leg on route 1

    @pytest.mark.skip
    def test_no_valid_transfer(self, basic_network_state: State) -> None:
        """Test when no valid transfer route exists."""
        # Try to find route to an unreachable node
        route_times, _ = calculate_route_times(basic_network_state)
        best_time, transfer_node, first_leg_route, second_leg_route = find_best_transfer_route(
            basic_network_state, jnp.array(0), jnp.array(4), route_times
        )

        # Should return inf/-1 since destination is unreachable
        chex.assert_trees_all_equal(best_time, jnp.array(jnp.inf))
        chex.assert_trees_all_equal(transfer_node, jnp.array(-1))
        chex.assert_trees_all_equal(first_leg_route, jnp.array(-1))
        chex.assert_trees_all_equal(second_leg_route, jnp.array(-1))
