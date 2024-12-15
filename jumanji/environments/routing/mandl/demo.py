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

from typing import List

import jax
import jax.numpy as jnp

from jumanji.environments.routing.mandl.env import Mandl
from jumanji.environments.routing.mandl.types import PassengerBatch, RouteBatch, State


def init_demand(num_passengers: int = 10) -> PassengerBatch:
    """Initialize passenger demand for testing."""
    return PassengerBatch(
        ids=jnp.arange(num_passengers),
        origins=jnp.array([0] * num_passengers),  # All start at node 0
        destinations=jnp.array([14] * num_passengers),  # All want to go to node 14
        departure_times=jnp.zeros(num_passengers),  # All want to depart at t=0
        time_waiting=jnp.zeros(num_passengers),
        time_in_vehicle=jnp.zeros(num_passengers),
        statuses=jnp.zeros(num_passengers, dtype=int),  # All start as not in system
    )


def run_simulation() -> List[State]:
    """Run a basic simulation of the transit network."""
    # Initialize environment
    env = Mandl()
    key = jax.random.PRNGKey(0)

    # Get initial state
    network_data, demand = env._load_instance_data()

    # Initialize passengers with demand
    passengers = init_demand()

    # Initialize vehicles
    vehicles = env._init_vehicles(num_vehicles=10, max_route_length=15)

    # Initialize routes
    routes = RouteBatch(
        ids=jnp.arange(2),
        nodes=jnp.array(
            [
                [0, 1, 2, 3, 4, -1],  # Route 0: 0->1->2->3->4
                [0, 5, 10, 14, -1, -1],  # Route 1: 0->5->10->14
            ]
        ),
        frequencies=jnp.array([2, 2]),  # 2 vehicles per route
        on_demand=jnp.array([False, False]),
    )

    # Create initial state
    state = State(
        network=network_data,
        vehicles=vehicles,
        routes=routes,
        passengers=passengers,
        current_time=0,
        key=key,
    )

    # Store states for animation
    states = [state]
    simulation_time = 60  # minutes

    # Run simulation
    for _ in range(simulation_time):
        # No actions for now, just simulate passenger movement
        action = 0
        state, _ = env.step(state, action)
        states.append(state)

    return states


def main() -> None:
    """Run demo and visualize simulation."""
    # Run simulation
    states = run_simulation()

    # Initialize viewer and create animation
    env = Mandl()
    _ = env.animate(states, interval=200, save_path="mandl_simulation.gif")

    print(f"Simulation completed with {len(states)} timesteps")
    print("Animation saved to mandl_simulation.gif")


if __name__ == "__main__":
    main()
