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

# %% imports
import jax
import matplotlib.pyplot as plt
import tqdm

from jumanji.environments.routing.mandl import Mandl
from jumanji.environments.routing.mandl.types import PassengerStatus, State

# jax.config.update("jax_disable_jit", True)


# %%
def main() -> State:
    # Enable interactive mode for real-time plotting
    plt.ion()

    # Create environment
    n_steps = 24 * 60 + 200
    env = Mandl(
        network_name="mandl1",
        runtime=24 * 60,
        vehicle_capacity=50,
        num_flex_routes=0,
        max_route_length=0,
        passenger_init_mode="evenly_spaced",
    )

    # Reset environment and get initial state
    key = jax.random.PRNGKey(42)
    state, timestep = env.reset(key)

    print("\nInitial State Info:")
    print(f"Number of vehicles: {state.fleet.num_vehicles}")
    print(f"Number of passengers: {state.passengers.num_passengers}")
    print(f"Number of routes: {state.routes.num_routes}")
    print(f"Network nodes: {state.network.num_nodes}")

    # Print route information
    print("\nRoute Information:")
    for i in range(state.routes.num_routes):
        route = state.routes.stops[i]
        valid_stops = route[route != -1]
        route_type = "Fixed" if state.routes.types[i] == 0 else "Flexible"
        print(f"Route {i+1} ({route_type}): {' -> '.join(str(s+1) for s in valid_stops)}")

    # Print vehicle positions
    print("\nVehicle Positions:")
    for i in range(state.fleet.num_vehicles):
        edge = state.fleet.current_edges[i]
        direction = "Forward" if state.fleet.directions[i] == 0 else "Backward"
        time_on_edge = state.fleet.times_on_edge[i]
        print(
            f"Vehicle {i+1}: Edge {edge[0]+1}->{edge[1]+1}, Direction: {direction}, "
            + f"Time on edge: {time_on_edge:.1f}"
        )

    # Print node coordinates
    print("\nNode Coordinates:")
    for i in range(state.network.num_nodes):
        coords = state.network.node_coordinates[i]
        print(f"Node {i+1}: ({coords[0]:.3f}, {coords[1]:.3f})")

    # Render initial state
    print("\nRendering initial state...")
    # env.render(state, save_path="initial_state.png")

    # Simulate a few steps
    print("\nSimulating steps...")
    # states = [state]
    step = jax.jit(env.step)

    for i in tqdm.tqdm(range(n_steps)):
        # For now, just use dummy action (no-op for all flexible routes)
        action = jax.numpy.full(
            state.routes.num_routes,
            state.network.num_nodes,  # no-op action
            dtype=int,
        )

        state, timestep = step(state, action)
        # states.append(state)

        print(f"\nStep {i+1}:")
        print(f"Time: {state.current_time:.1f}")
        print(
            "Not in system passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM).sum()}"
        )
        print(
            "Waiting passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.WAITING).sum()}"
        )
        print(
            "In-vehicle passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.IN_VEHICLE).sum()}"
        )
        print(
            "Completed passengers:"
            + f"{(state.passengers.statuses == PassengerStatus.COMPLETED).sum()}"
        )

        print(f"Sum of waiting times: {state.passengers.time_waiting.sum():.2f}")
        print(f"Sum of in-vehicle times: {state.passengers.time_in_vehicle.sum():.2f}")
        print(
            "Total Travel time:"
            f"{state.passengers.time_in_vehicle.sum() + state.passengers.time_waiting.sum():.2f}"
        )

        # Render every few steps
        # if (i + 1) % 3 == 0:
        #     env.render(state)

    # Create animation
    # print("\nCreating animation...")
    # animation = env.animate(states, interval=500, save_path="mandl_simulation.gif")

    # Clean up
    env.close()
    plt.ioff()

    # print("\nDemo completed! Check 'mandl_simulation.gif' for the animation.")

    return state


state = main()

# %%
# Print timing statistics
print("\nPassenger Statistics:")
print(f"Number of passengers: {state.passengers.num_passengers}")
print(f"Average waiting time: {state.passengers.time_waiting.mean():.2f}")
print(f"Average in-vehicle time: {state.passengers.time_in_vehicle.mean():.2f}")
print(
    "Average total time:"
    + f"{(state.passengers.time_waiting + state.passengers.time_in_vehicle).mean():.2f}"
)
print(f"Maximum waiting time: {state.passengers.time_waiting.max():.2f}")
print(f"Maximum in-vehicle time: {state.passengers.time_in_vehicle.max():.2f}")
print(f"Sum of waiting times: {state.passengers.time_waiting.sum():.2f}")
print(f"Sum of in-vehicle times: {state.passengers.time_in_vehicle.sum():.2f}")
print(
    "Total Travel time: "
    + f"{state.passengers.time_in_vehicle.sum() + state.passengers.time_waiting.sum():.2f}"
)

# %%
# Create figure for histograms
plt.figure(figsize=(12, 5))

# Plot waiting time distribution
plt.subplot(1, 2, 1)
plt.hist(state.passengers.time_waiting, bins=30, alpha=0.75)
plt.title("Distribution of Waiting Times")
plt.xlabel("Waiting Time")
plt.ylabel("Frequency")

# Plot in-vehicle time distribution
plt.subplot(1, 2, 2)
plt.hist(state.passengers.time_in_vehicle, bins=30, alpha=0.75)
plt.title("Distribution of In-Vehicle Times")
plt.xlabel("In-Vehicle Time")
plt.ylabel("Frequency")

# Show and save the histogram figure
plt.tight_layout()
plt.savefig("passenger_time_distributions.png", dpi=300, bbox_inches="tight")
print("Saved passenger time distributions to passenger_time_distributions.png")
plt.show()

# %%
print("\nPassenger Transfer Statistics:")
print(f"Number of transfers: {state.passengers.has_transferred.sum()}")
print(f"Average transfers per passenger: {state.passengers.has_transferred.mean():.2f}")
