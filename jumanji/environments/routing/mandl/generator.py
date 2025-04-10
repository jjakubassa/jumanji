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

import abc
from typing import Literal, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.mandl.types import (
    State,
)
from jumanji.environments.routing.mandl.utils import (
    create_initial_fleet,
    create_initial_passengers,
    create_initial_routes,
    load_demand_data,
    load_network_data,
)


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class for Mandl environment.
    A `Generator` is responsible for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        network_name: Literal["mandl1", "ceder1", "mumford0", "mumford1", "mumford2", "mumford3"],
        num_fix_routes: int,
        num_flex_routes: int,
        total_vehicles: int,
        vehicle_capacity: int,
        runtime: float,
        buffer_time_start: float,
        buffer_time_end: float,
        max_route_length: int,
        allow_actions_fixed_routes: bool,
        solution_name: Optional[str] = None,
    ):
        """Initialize the generator with problem parameters.

        Args:
            network_name: Name of the network to use
            num_fix_routes: Number of fixed routes
            num_flex_routes: Number of flexible routes
            total_vehicles: Total number of vehicles available
            vehicle_capacity: Capacity of each vehicle
            runtime: Total runtime of the episode
            buffer_time_start: Buffer time at start
            buffer_time_end: Buffer time at end
            max_route_length: Maximum length of routes
            solution_name: Optional name of solution file to load
        """
        self.network_name = network_name
        self.num_fix_routes = num_fix_routes
        self.num_flex_routes = num_flex_routes
        self.total_vehicles = total_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.runtime = runtime
        self.buffer_time_start = buffer_time_start
        self.buffer_time_end = buffer_time_end
        self.max_route_length = max_route_length
        self.allow_actions_fixed_routes = allow_actions_fixed_routes

        # Load static data
        self.network_data = load_network_data(network_name)
        self.demand_data = load_demand_data(network_name)

        # Load solution routes if specified
        self._routes: tuple[tuple[int]] = tuple()
        self.vehicles_per_solution_route: tuple[int] = (0,)
        if solution_name is not None:
            from jumanji.environments.routing.mandl.utils import load_solution_data

            self._routes, self.vehicles_per_solution_route = load_solution_data(
                network_name, solution_name
            )
            self.num_solution_routes = len(self._routes)
        else:
            self.num_solution_routes = 0

        self._vehicles_per_route: Optional[tuple[int, ...]] = None

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a new state.

        Args:
            key: Random key for stochastic generation

        Returns:
            Initial state for the Mandl environment
        """

    def get_current_vehicles_per_route(self) -> tuple[int, ...]:
        """Get the current vehicle allocation."""
        if self._vehicles_per_route is None:
            raise ValueError("Vehicle allocation not yet generated. Call __call__ first.")
        return self._vehicles_per_route


class DefaultGenerator(Generator):
    """Default generator for Mandl environment instances."""

    def __init__(
        self,
        network_name: Literal[
            "mandl1", "ceder1", "mumford0", "mumford1", "mumford2", "mumford3"
        ] = "mandl1",
        num_fix_routes: int = 1,
        num_flex_routes: int = 16,
        total_vehicles: int = 99,
        vehicle_capacity: int = 50,
        runtime: float = 150.0,
        buffer_time_end: float = 50.0,
        buffer_time_start: Optional[float] = None,
        max_route_length: int = 8,
        solution_name: Optional[str] = None,
        allow_actions_fixed_routes: bool = True,
        random_vehicle_allocation: bool = False,
        vehicles_per_additional_fixed_route: Optional[tuple[int, ...]] = None,
        passenger_init_mode: Literal[
            "evenly_spaced", "rush_hour", "uniform_random", "all_at_start"
        ] = "evenly_spaced",
    ):
        """Initialize the default generator.

        Args:
            network_name: Name of the network to use
            num_fix_routes: Number of fixed routes
            num_flex_routes: Number of flexible routes
            total_vehicles: Total number of vehicles available
            vehicle_capacity: Capacity of each vehicle
            runtime: Total runtime of the episode
            buffer_time_end: Buffer time at end
            buffer_time_start: Buffer time at start (defaults to max_route_length)
            max_route_length: Maximum length of routes
            solution_name: Optional name of solution file to load
            random_vehicle_allocation: Whether to randomly allocate vehicles to fixed routes
            vehicles_per_additional_fixed_route: Optional allocation for additional fixed routes
            allow_actions_fixed_routes: Whether to allow actions on fixed routes
            passenger_init_mode: Mode for initializing passengers
        """
        if buffer_time_start is None:
            buffer_time_start = max_route_length

        super().__init__(
            network_name=network_name,
            num_fix_routes=num_fix_routes,
            num_flex_routes=num_flex_routes,
            total_vehicles=total_vehicles,
            vehicle_capacity=vehicle_capacity,
            runtime=runtime,
            buffer_time_start=buffer_time_start,
            buffer_time_end=buffer_time_end,
            max_route_length=max_route_length,
            solution_name=solution_name,
            allow_actions_fixed_routes=allow_actions_fixed_routes,
        )

        self.random_vehicle_allocation = random_vehicle_allocation
        self.passenger_init_mode = passenger_init_mode
        self.vehicles_per_additional_fixed_route = vehicles_per_additional_fixed_route

        # Validate vehicle allocations if specified
        if vehicles_per_additional_fixed_route is not None:
            if len(vehicles_per_additional_fixed_route) != num_fix_routes:
                raise ValueError(
                    f"Expected {num_fix_routes} vehicle counts for additional fixed routes, "
                    f"got {len(vehicles_per_additional_fixed_route)}"
                )

            total_fixed_vehicles = sum(self.vehicles_per_solution_route) + sum(
                vehicles_per_additional_fixed_route
            )
            if total_fixed_vehicles > total_vehicles:
                raise ValueError(
                    f"Total vehicles in fixed routes ({total_fixed_vehicles}) "
                    f"exceeds total vehicles ({total_vehicles})"
                )

    def _generate_random_vehicle_allocation(
        self,
        key: chex.PRNGKey,
        num_routes: int,
        total_vehicles: int,
        min_vehicles_per_route: int = 1,
    ) -> Tuple[int, ...]:
        """Generate random vehicle allocations ensuring minimum vehicles per route."""
        if total_vehicles < num_routes * min_vehicles_per_route:
            raise ValueError(
                f"Not enough vehicles ({total_vehicles}) to ensure minimum "
                f"of {min_vehicles_per_route} vehicles for {num_routes} routes"
            )

        # First, allocate minimum vehicles to each route
        remaining_vehicles = total_vehicles - (num_routes * min_vehicles_per_route)

        # Generate random proportions using numpy (convert from JAX)
        props = jax.random.uniform(key, shape=(num_routes,)).numpy()
        props = props / props.sum()

        # Calculate additional vehicles using regular Python
        additional_vehicles = []
        remaining = remaining_vehicles
        for p in props[:-1]:  # Process all but the last proportion
            vehicles = int(p * remaining_vehicles)
            additional_vehicles.append(vehicles)
            remaining -= vehicles

        # Add remaining vehicles to the last route
        additional_vehicles.append(remaining)

        # Add minimum vehicles and create final allocation
        final_allocation = [v + min_vehicles_per_route for v in additional_vehicles]

        # Sort in descending order and return as tuple
        return tuple(sorted(final_allocation, reverse=True))

    def __call__(self, key: chex.PRNGKey) -> State:
        # Split keys for different random operations
        key, route_key, vehicle_key, passenger_key = jax.random.split(key, 4)

        # Calculate remaining vehicles after accounting for solution routes and flex routes
        remaining_vehicles = (
            self.total_vehicles
            - sum(self.vehicles_per_solution_route)
            - self.num_flex_routes  # Reserve 1 vehicle per flex route
        )

        # Handle vehicle allocation
        if self.vehicles_per_additional_fixed_route is not None:
            # Use specified allocation
            vehicles_per_additional_fixed_route = self.vehicles_per_additional_fixed_route
        elif self.random_vehicle_allocation and self.num_fix_routes > 0:
            vehicles_per_additional_fixed_route = self._generate_random_vehicle_allocation(
                vehicle_key,
                self.num_fix_routes,
                remaining_vehicles,
                min_vehicles_per_route=1,
            )
        else:
            # If no random allocation and no specific allocation,
            # distribute remaining vehicles evenly
            if self.num_fix_routes > 0:
                base_vehicles = remaining_vehicles // self.num_fix_routes
                extra_vehicles = remaining_vehicles % self.num_fix_routes

                # Create allocation with extra vehicles distributed to first routes
                vehicles_per_additional_fixed_route = tuple(
                    base_vehicles + 1 if i < extra_vehicles else base_vehicles
                    for i in range(self.num_fix_routes)
                )
                # Sort in descending order
                vehicles_per_additional_fixed_route = tuple(
                    sorted(vehicles_per_additional_fixed_route, reverse=True)
                )
            else:
                vehicles_per_additional_fixed_route = tuple()

        # Create initial fleet
        initial_fleet, vehicles_per_route = create_initial_fleet(
            num_routes=self.num_fix_routes + self.num_flex_routes,
            num_flex_routes=self.num_flex_routes,
            total_vehicles=self.total_vehicles,
            vehicles_per_solution_route=self.vehicles_per_solution_route,
            vehicles_per_additional_fixed_route=vehicles_per_additional_fixed_route,
            vehicle_capacity=self.vehicle_capacity,
        )
        self._vehicles_per_route = vehicles_per_route

        # Create initial routes
        route_batch = create_initial_routes(
            self._routes,
            num_fix_routes=self.num_fix_routes,
            num_flex_routes=self.num_flex_routes,
            network_data=self.network_data,
            max_stops=self.max_route_length,
            key=route_key,
        )

        # Create initial state
        state = State(
            network=self.network_data,
            fleet=initial_fleet,
            passengers=create_initial_passengers(
                self.demand_data,
                passenger_key,
                runtime=self.runtime,
                buffer_time_start=self.buffer_time_start,
                buffer_time_end=self.buffer_time_end,
                mode=self.passenger_init_mode,
            ),
            routes=route_batch,
            current_time=jnp.array(0.0),
            key=key,
        )

        return state
