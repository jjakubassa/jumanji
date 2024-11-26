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
from typing import Dict, List, Tuple

import chex
import jax.numpy as jnp
from jax import random

from jumanji.environments.routing.mandl.env import Mandl
from jumanji.environments.routing.mandl.types import (
    Passenger,
    PassengerBatch,
    Route,
    State,
    Vehicle,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer


# Passenger Generation Functions
def passengers_to_batch(passengers: List[Passenger]) -> PassengerBatch:
    """Convert list of passengers to batch format"""
    return PassengerBatch(
        ids=jnp.array([p.id for p in passengers]),
        origins=jnp.array([p.origin for p in passengers]),
        destinations=jnp.array([p.destination for p in passengers]),
        departure_times=jnp.array([p.departure_time for p in passengers]),
        statuses=jnp.array([p.status for p in passengers]),
    )


# Simulation Functions
def simulate_passenger_flow(
    key: chex.PRNGKey,
    mandl_env: Mandl,
    simulation_period: float = 60.0,
) -> Tuple[List[Passenger], PassengerBatch]:
    """Simulate passenger flow for the given period"""
    network_data = mandl_env._load_network()
    passengers = mandl_env._generate_passengers(key, network_data.demand, simulation_period)
    passenger_batch = passengers_to_batch(passengers)
    return passengers, passenger_batch


# Analysis Functions
def analyze_passenger_demand(passengers: List[Passenger]) -> Dict:
    """Analyze passenger demand patterns"""
    origin_counts: Dict[int, int] = {}
    dest_counts: Dict[int, int] = {}
    time_periods: Dict[int, int] = {}

    for p in passengers:
        origin_counts[p.origin] = origin_counts.get(p.origin, 0) + 1
        dest_counts[p.destination] = dest_counts.get(p.destination, 0) + 1
        period = int(p.departure_time / 10)
        time_periods[period] = time_periods.get(period, 0) + 1

    return {
        "origin_counts": origin_counts,
        "destination_counts": dest_counts,
        "time_period_counts": time_periods,
    }


def read_solution_file(file_path: str) -> Tuple[str, List[Route], List[int]]:
    """Read the solution file and extract routes and vehicle counts"""
    routes = []
    vehicle_counts = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        title = lines[0].strip()
        num_routes = int(lines[1].strip())
        for i in range(2, 2 + num_routes):
            # ruff: noqa: RUF001
            route_nodes = [int(node) - 1 for node in lines[i].strip().split("–")]
            routes.append(
                Route(nodes=jnp.array(route_nodes), frequency=0, capacity=40)
            )  # Frequency will be set later
        for i in range(2 + num_routes, len(lines)):
            vehicle_counts.append(int(lines[i].strip()))
    return title, routes, vehicle_counts


def calculate_round_trip_time(route: Route, travel_times: jnp.ndarray) -> float:
    """Calculate the round trip time for a given route"""
    round_trip_time = 0.0
    for i in range(len(route.nodes) - 1):
        round_trip_time += travel_times[route.nodes[i], route.nodes[i + 1]]
    round_trip_time *= 2  # Back and forth
    return round_trip_time / 60.0  # Convert to hours


def main() -> None:
    # Initialize the Mandl environment
    mandl_env = Mandl()
    viewer = MandlViewer(name="Mandl", render_mode="human")

    # Set random seed
    key = random.PRNGKey(42)

    # Generate passengers for 1-hour simulation
    passengers, passenger_batch = simulate_passenger_flow(key, mandl_env, simulation_period=60.0)

    # Read routes and vehicle counts from the solution file
    title, routes, vehicle_counts = read_solution_file(
        "jumanji/environments/routing/mandl/mandl1_solution.txt"
    )

    # Load network data
    network_data = mandl_env._load_network()

    # Create vehicles for each route
    example_vehicles = []
    vehicle_id = 1
    current_time = 0.0  # Initial time

    for route, count in zip(routes, vehicle_counts, strict=False):
        interval = 60 / count  # Interval in minutes between vehicle departures
        for i in range(count):  # Generate vehicles as per the count
            departure_time = round(i * interval)
            current_edge = (route.nodes[0], route.nodes[1])  # Start at the first edge
            time_on_edge = 0.0  # Initially, the vehicle has just started
            example_vehicles.append(
                Vehicle(
                    id=vehicle_id,
                    route=route,
                    current_edge=current_edge,
                    time_on_edge=time_on_edge,
                    passengers=jnp.array([]),  # Empty for simplicity
                    capacity=route.capacity,
                    next_departure=departure_time,  # Set departure time
                )
            )
            vehicle_id += 1

    # Create initial state
    state = State(
        network=network_data,
        vehicles=example_vehicles,
        passengers=passengers,
        current_time=current_time,
        key=key,
        save_path="network_state.svg",
    )

    # Simulate over a few time steps
    states = [state]
    for t in range(1, 11):  # Simulate for 10 time steps (1 minute each)
        print(f"Time step: {t}")
        new_vehicles = []
        for vehicle in states[-1].vehicles:  # Use the last state in the list
            # Calculate new position based on time step
            new_vehicle = mandl_env._move_vehicle(vehicle, states[-1].network.links)
            new_vehicles.append(new_vehicle)
        new_state = replace(
            states[-1], vehicles=new_vehicles, current_time=t * 1.0
        )  # Update time by 1 minute
        states.append(new_state)

    # Animate the states and save as a GIF
    viewer.animate(states, interval=200, save_path="network_animation.gif")


if __name__ == "__main__":
    main()
