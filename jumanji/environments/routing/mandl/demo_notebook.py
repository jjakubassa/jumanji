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

import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Passenger Stats""")
    return


@app.cell
def _(passenger_df):
    passenger_df
    return


@app.cell
def _(passenger_df, pl):
    last_state_passenger_df = passenger_df.filter(
        pl.col("time") == passenger_df["time"].max()
    )
    last_state_passenger_df
    return (last_state_passenger_df,)


@app.cell
def _(alt, last_state_passenger_df):
    (
        alt.Chart(last_state_passenger_df)
        .mark_bar()
        .encode(
            x="status:N",
            y="count()",
            color="status",
        )
    )
    return


@app.cell
def _(alt, last_state_passenger_df, pl):
    # pre-aggregate data
    _od_counts = last_state_passenger_df.group_by(
        ["origin", "destination", "status"]
    ).agg(pl.count("passenger_id").alias("count"))

    (
        alt.Chart(_od_counts)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            row="status:N",
            color=alt.Color("count")
            .scale(scheme="greenblue")
            .title("Number of Passengers"),
            tooltip=["origin", "destination", "count"],
        )
    )
    return


@app.cell(hide_code=True)
def _(alt, last_state_passenger_df, mo):
    _chart = (
        alt.Chart(last_state_passenger_df)
        .mark_bar()
        .encode(
            alt.X("time_waiting", bin=True),
            y="count()",
        )
    )

    mo.ui.altair_chart(
        _chart,
        # disable automatic selection
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(alt, last_state_passenger_df, mo):
    _chart = (
        alt.Chart(last_state_passenger_df)
        .mark_bar()
        .encode(
            x=alt.X("time_waiting", bin=True),
            y="count()",
            color="has_transferred",
        )
    )

    mo.ui.altair_chart(
        _chart,
        # disable automatic selection
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(alt, last_state_passenger_df, mo):
    # plot time waiting vs time in vehicle using altair
    _chart = (
        alt.Chart(last_state_passenger_df)
        .mark_circle()
        .encode(
            x="time_waiting",
            y="time_in_vehicle",
            color="has_transferred",
            tooltip=["time_waiting", "time_in_vehicle", "has_transferred"],
        )
    )

    mo.ui.altair_chart(
        _chart,
        # disable automatic selection
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(alt, last_state_passenger_df, mo, passenger_df, pl):
    # pre-aggregate data
    _od_counts = last_state_passenger_df.group_by(["origin", "destination"]).agg(
        pl.count("passenger_id").alias("count")
    )
    _od_time_waiting = last_state_passenger_df.group_by(
        ["origin", "destination"]
    ).agg(pl.mean("time_waiting").alias("mean_time_waiting"))

    _od_time_in_vehicle = passenger_df.group_by(["origin", "destination"]).agg(
        pl.mean("time_in_vehicle").alias("mean_time_in_vehicle")
    )

    _p_counts = (
        alt.Chart(_od_counts)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("count")
            .scale(scheme="greenblue")
            .title("Number of Passengers"),
            tooltip=["origin", "destination", "count"],
        )
    )

    _p_time_waiting = (
        alt.Chart(_od_time_waiting)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("mean_time_waiting")
            .scale(scheme="greenblue")
            .title("Average Waiting Time"),
            tooltip=["origin", "destination", "mean_time_waiting"],
        )
    )

    _p_time_in_vehcile = (
        alt.Chart(_od_time_in_vehicle)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("mean_time_in_vehicle")
            .scale(scheme="greenblue")
            .title("Average In-Vehicle Time"),
            tooltip=["origin", "destination", "mean_time_in_vehicle"],
        )
    )

    mo.hstack([_p_counts, _p_time_waiting, _p_time_in_vehcile])
    return


@app.cell(hide_code=True)
def _(alt, last_state_passenger_df, mo, passenger_df, pl):
    # pre-aggregate data
    _od_counts = last_state_passenger_df.group_by(
        ["origin", "destination", "has_transferred"]
    ).agg(pl.count("passenger_id").alias("count"))
    _od_time_waiting = last_state_passenger_df.group_by(
        ["origin", "destination", "has_transferred"]
    ).agg(pl.mean("time_waiting").alias("mean_time_waiting"))

    _od_time_in_vehicle = passenger_df.group_by(
        ["origin", "destination", "has_transferred"]
    ).agg(pl.mean("time_in_vehicle").alias("mean_time_in_vehicle"))

    _p_counts = (
        alt.Chart(_od_counts)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            row="has_transferred:N",
            color=alt.Color("count")
            .scale(scheme="greenblue")
            .title("Number of Passengers"),
            tooltip=["origin", "destination", "count"],
        )
    )

    _p_time_waiting = (
        alt.Chart(_od_time_waiting)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            row="has_transferred:N",
            color=alt.Color("mean_time_waiting")
            .scale(scheme="greenblue")
            .title("Average Waiting Time"),
            tooltip=["origin", "destination", "mean_time_waiting"],
        )
    )

    _p_time_in_vehcile = (
        alt.Chart(_od_time_in_vehicle)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            row="has_transferred:N",
            color=alt.Color("mean_time_in_vehicle")
            .scale(scheme="greenblue")
            .title("Average In-Vehicle Time"),
            tooltip=["origin", "destination", "mean_time_in_vehicle"],
        )
    )

    mo.hstack([_p_counts, _p_time_waiting, _p_time_in_vehcile])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vehicle Stats""")
    return


@app.cell
def _(fleet_df):
    fleet_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""When was which vehicle full? Are they of the same routes?""")
    return


@app.cell(hide_code=True)
def _(env, fleet_df, pl):
    try:
        _num_passengers_over_time = fleet_df.select(
            ["time", "num_passengers", "vehicle_id", "route_id"]
        ).filter(pl.col("num_passengers") == env.vehicle_capacity)
        _num_passengers_over_time.hvplot.scatter(
            x="time",
            y="vehicle_id",
            by="vehicle_id",
            groupby="route_id",
            title="Vehicle is full",
            hover_tooltips=["time", "vehicle_id"],
        )
    except:
        print("No full vehicles found")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Which passengers could not be served? The ones with the high waiting times at origin=0?""")
    return


@app.cell(hide_code=True)
def _(alt, env, fleet_df, pl):
    _num_passengers_matrix = (
        fleet_df.select(["time", "num_passengers", "current_from", "current_to"])
        .filter(pl.col("num_passengers") == env.vehicle_capacity)
        .group_by(["current_from", "current_to"])
        .agg(pl.count("time").alias("count"))
    )

    (
        alt.Chart(_num_passengers_matrix)
        .mark_rect()
        .encode(
            x="current_from:N",
            y="current_to:N",
            color=alt.Color("count")
            .scale(scheme="greenblue")
            .title("Timesteps with full vehicles"),
            tooltip=["current_from", "current_to", "count"],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Routes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How efficient are the routes compared to shortest path?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How efficient is the actual travel time (in vehicle) compared to the shortest path?""")
    return


@app.cell(hide_code=True)
def _(alt, jnp, last_state_passenger_df, mo, np, pl, states):
    # find shortest path for each od pair with floyd warshall algorithm
    def shortest_path_floyd_warshall(
        travel_times: np.ndarray,
    ) -> jnp.ndarray:
        """Compute shortest path distances between all pairs of nodes using Floyd-Warshall algorithm.

        Args:
            travel_times: Array of travel times between nodes

        Returns:
            Array of shortest path distances between all pairs of nodes
        """
        n = travel_times.shape[0]
        dist = travel_times.copy()

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist = dist.at[i, j].set(dist[i, k] + dist[k, j])

        return dist


    shortest_paths = shortest_path_floyd_warshall(states[-1].network.travel_times)
    shortest_paths = np.array(shortest_paths)

    # Create DataFrame for shortest path travel times
    _df_shortest_path = pl.DataFrame(
        {
            "origin": np.repeat(
                np.arange(shortest_paths.shape[0]), shortest_paths.shape[1]
            ),
            "destination": np.tile(
                np.arange(shortest_paths.shape[1]), shortest_paths.shape[0]
            ),
            "travel_time": shortest_paths.flatten(),
        }
    )

    # Plot shortest path travel times
    _p_shortest_path = (
        alt.Chart(_df_shortest_path)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color(
                "travel_time", scale=alt.Scale(scheme="greenblue"), title="min"
            ),
            tooltip=["origin", "destination", "travel_time"],
        )
        .properties(title="Shortest Path Travel Times")
    )

    _od_time_in_vehicle = last_state_passenger_df.group_by(
        ["origin", "destination"]
    ).agg(pl.mean("time_in_vehicle").alias("mean_time_in_vehicle"))

    _od_min_time_in_vehicle = last_state_passenger_df.group_by(
        ["origin", "destination"]
    ).agg(pl.min("time_in_vehicle").alias("min_time_in_vehicle"))

    _p_time_in_vehcile = (
        alt.Chart(_od_time_in_vehicle)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("mean_time_in_vehicle")
            .scale(scheme="greenblue")
            .title("min"),
            tooltip=["origin", "destination", "mean_time_in_vehicle"],
        )
    ).properties(title="Average In-Vehicle Time")


    combined_df = _od_time_in_vehicle.join(
        _df_shortest_path, on=["origin", "destination"], how="inner"
    ).select(
        "origin",
        "destination",
        "mean_time_in_vehicle",
        "travel_time",
        (pl.col("mean_time_in_vehicle") - pl.col("travel_time")).alias("diff"),
    )

    combined_df = combined_df.join(
        _od_min_time_in_vehicle, on=["origin", "destination"], how="inner"
    ).select(
        "origin",
        "destination",
        "mean_time_in_vehicle",
        "travel_time",
        "diff",
        "min_time_in_vehicle",
        (pl.col("min_time_in_vehicle") - pl.col("travel_time")).alias("min_diff"),
    )

    _p_diff = (
        alt.Chart(combined_df)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("diff").scale(scheme="greenblue").title("min"),
            tooltip=["origin", "destination", "diff"],
        )
    ).properties(title="Avg. Travel time - Shortest Path")

    _p_min_diff = (
        alt.Chart(combined_df)
        .mark_rect()
        .encode(
            x="origin:N",
            y="destination:N",
            color=alt.Color("min_diff").scale(scheme="greenblue").title("min"),
            tooltip=["origin", "destination", "min_diff"],
        )
    ).properties(title="Minimal Travel time - Shortest Path")


    mo.hstack([_p_shortest_path, _p_time_in_vehcile, _p_diff, _p_min_diff])
    return combined_df, shortest_path_floyd_warshall, shortest_paths


@app.cell
def _(passenger_df):
    passenger_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analysis of inefficient journeys""")
    return


@app.cell(hide_code=True)
def _(
    PassengerStatus,
    State,
    VehicleDirection,
    calculate_route_times,
    calculate_waiting_times,
    debug_assign_single_passenger,
    find_best_transfer_route,
    handle_completed_and_transferring_passengers,
    increment_in_vehicle_times,
    increment_wait_times,
    jnp,
    move_vehicles,
    np,
    replace,
    states,
    update_passengers_to_waiting,
    update_routes,
):
    def analyze_inefficient_journey(
        states: list[State], threshold_ratio: float = 2.0
    ) -> None:
        """Find and analyze a passenger with inefficient in-vehicle time compared to shortest path.

        Args:
            states: List of environment states
            threshold_ratio: Minimum ratio of actual to optimal time to be considered inefficient
        """
        final_state = states[-1]

        # 1. Find shortest paths using Floyd-Warshall
        travel_times = np.array(final_state.network.travel_times)
        n_nodes = len(travel_times)
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if (
                        travel_times[i, j]
                        > travel_times[i, k] + travel_times[k, j]
                    ):
                        travel_times[i, j] = (
                            travel_times[i, k] + travel_times[k, j]
                        )

        # 2. Find inefficient journeys
        inefficient_journeys = []
        for p_idx in range(final_state.passengers.num_passengers):
            if final_state.passengers.statuses[p_idx] == PassengerStatus.COMPLETED:
                origin = final_state.passengers.origins[p_idx]
                dest = final_state.passengers.destinations[p_idx]
                actual_time = final_state.passengers.time_in_vehicle[p_idx]
                shortest_time = travel_times[origin, dest]

                ratio = actual_time / shortest_time
                if ratio >= threshold_ratio:
                    inefficient_journeys.append(
                        (p_idx, ratio, actual_time, shortest_time)
                    )

        if not inefficient_journeys:
            print("No inefficient journeys found.")
            return

        # 3. Select worst case for detailed analysis
        inefficient_journeys.sort(key=lambda x: x[1], reverse=True)
        p_idx, ratio, actual_time, shortest_time = inefficient_journeys[0]

        print(f"\nAnalyzing most inefficient journey (Passenger {p_idx}):")
        print(f"Actual time: {actual_time:.1f}")
        print(f"Optimal time: {shortest_time:.1f}")
        print(f"Ratio: {ratio:.2f}x optimal")

        # 4. Track complete journey with detailed vehicle movements
        origin = final_state.passengers.origins[p_idx]
        dest = final_state.passengers.destinations[p_idx]
        print(f"\nJourney: Node {origin + 1} -> Node {dest + 1}")

        current_vehicle = None
        current_route = None
        journey_events = []

        print("\nDetailed journey timeline:")
        print("-" * 60)
        print("Time | Status      | Location | Vehicle | Route | Direction")
        print("-" * 60)

        for t, state in enumerate(states):
            status = state.passengers.statuses[p_idx]

            if status == PassengerStatus.NOT_IN_SYSTEM:
                if t == 0:  # Only print initial state
                    print(
                        f"{t:4d} | NOT_IN_SYS  | Node {origin + 1:2d}  |    -    |   -   |    -"
                    )

            elif status == PassengerStatus.WAITING:
                if current_vehicle is None:  # Only print when status changes
                    print(
                        f"{t:4d} | WAITING     | Node {origin + 1:2d}  |    -    |   -   |    -"
                    )

            elif status == PassengerStatus.IN_VEHICLE:
                # Find which vehicle the passenger is in
                for v_idx in range(state.fleet.num_vehicles):
                    if p_idx in state.fleet.passengers[v_idx]:
                        route_id = state.fleet.route_ids[v_idx]
                        pos_from = state.fleet.current_edges[v_idx, 0]
                        pos_to = state.fleet.current_edges[v_idx, 1]
                        direction = state.fleet.directions[v_idx]
                        dir_str = (
                            "FWD"
                            if direction == VehicleDirection.FORWARD
                            else "BWD"
                        )
                        time_on_edge = state.fleet.times_on_edge[v_idx]

                        # Print if vehicle/position/direction changes or first boarding
                        if (
                            v_idx != current_vehicle
                            or len(journey_events) == 0
                            or journey_events[-1][1:]
                            != (v_idx, pos_from, pos_to, direction)
                        ):
                            print(
                                f"{t:4d} | IN_VEHICLE  | {pos_from + 1:2d}->{pos_to + 1:2d}  | Bus {v_idx:2d}  | {route_id + 1:3d}  | {dir_str:5s}"
                            )
                            journey_events.append(
                                (t, v_idx, pos_from, pos_to, direction)
                            )

                        current_vehicle = v_idx
                        current_route = route_id
                        break

            elif status == PassengerStatus.TRANSFERRING:
                transfer_node = state.passengers.transfer_nodes[p_idx]
                if current_vehicle is not None:  # Just got off a vehicle
                    print(
                        f"{t:4d} | TRANSFER    | Node {transfer_node + 1:2d}  |    -    |   -   |    -"
                    )
                    current_vehicle = None
                    current_route = None

            elif status == PassengerStatus.COMPLETED:
                if current_vehicle is not None:  # Just completed
                    print(
                        f"{t:4d} | COMPLETED   | Node {dest + 1:2d}  |    -    |   -   |    -"
                    )
                    current_vehicle = None
                    break

        print("-" * 60)

        # 5. Analyze journey efficiency
        unique_vehicles = len(set(event[1] for event in journey_events))
        route_segments = len(journey_events)

        print(f"\nJourney Statistics:")
        print(f"Total vehicles used: {unique_vehicles}")
        print(f"Total route segments: {route_segments}")

        # 6. Calculate optimal path
        route_times, _ = calculate_route_times(final_state)
        best_time, transfer_node, first_leg_route, second_leg_route = (
            find_best_transfer_route(final_state, origin, dest, route_times)
        )

        print("\nOptimal path would be:")
        if jnp.isfinite(best_time):
            if transfer_node != -1:
                print(
                    f"Route {first_leg_route + 1}: Node {origin + 1} -> Node {transfer_node + 1}"
                )
                print(
                    f"Route {second_leg_route + 1}: Node {transfer_node + 1} -> Node {dest + 1}"
                )
                print(f"Total optimal time: {best_time:.1f}")
            else:
                print(
                    f"Direct route {first_leg_route + 1}: Node {origin + 1} -> Node {dest + 1}"
                )
                print(f"Total optimal time: {best_time:.1f}")
        else:
            print("No optimal path found!")

        print("\nANALYZING CRITICAL BOARDING DECISIONS:")
        # Find first boarding decision for this passenger
        for t in range(len(states) - 1):
            state = states[t]
            next_state = states[t + 1]

            # Check if passenger boarded at this step
            if (
                state.passengers.statuses[p_idx] in [PassengerStatus.NOT_IN_SYSTEM]
                and next_state.passengers.statuses[p_idx]
                == PassengerStatus.IN_VEHICLE
            ):
                print(f"\nFound boarding at t={t}")

                # Replicate all state updates before assignment
                print("\nReplicating state updates:")

                # 1. Update routes (using no-op action since we're not changing routes)
                action = jnp.full(
                    state.routes.num_routes, state.network.num_nodes, dtype=int
                )
                updated_state = replace(
                    state,
                    routes=update_routes(
                        state.routes, state.network.num_nodes, action
                    ),
                )
                print("1. Routes updated")

                # 2. Move vehicles
                updated_state = move_vehicles(updated_state)
                print("2. Vehicles moved")

                # 3. Increase times
                new_passengers = increment_wait_times(updated_state.passengers)
                new_passengers = increment_in_vehicle_times(new_passengers)
                updated_state = replace(updated_state, passengers=new_passengers)
                print("3. Times incremented")

                # 4. Handle completed/transferring
                updated_state = handle_completed_and_transferring_passengers(
                    updated_state
                )
                print("4. Completed/transferring handled")

                # 5. Update to waiting
                new_passengers = update_passengers_to_waiting(
                    updated_state.passengers,
                    updated_state.current_time,
                )
                updated_state = replace(updated_state, passengers=new_passengers)
                print("5. Updated to waiting")
                print(
                    f"Passenger status: {PassengerStatus(updated_state.passengers.statuses[p_idx]).name}"
                )

                print("\nAnalyzing assignment decision:")
                # Calculate necessary inputs for assignment analysis
                route_times, route_directions = calculate_route_times(
                    updated_state
                )
                waiting_times = calculate_waiting_times(
                    updated_state, route_times, route_directions
                )

                # Run detailed assignment analysis on the properly updated state
                debug_assign_single_passenger(
                    updated_state,
                    p_idx,
                    route_times,
                    route_directions,
                    waiting_times,
                )
                break

        return p_idx  # Return passenger ID for further analysis if needed


    analyze_inefficient_journey(states)
    return (analyze_inefficient_journey,)


@app.cell(hide_code=True)
def _(
    PassengerStatus,
    State,
    VehicleDirection,
    calculate_route_times,
    calculate_waiting_times,
    find_best_transfer_route,
    get_direction_if_connected,
    jnp,
    states,
):
    def analyze_stuck_passenger(
        states: list[State], min_wait_time: float = 100.0
    ) -> None:
        """Analyze a passenger who has been waiting for a long time.

        Args:
            states: List of environment states
            min_wait_time: Minimum waiting time to consider a passenger stuck
        """
        final_state = states[-1]

        # Find stuck passengers (long waiting times)
        waiting_mask = final_state.passengers.statuses == PassengerStatus.WAITING
        long_wait_mask = final_state.passengers.time_waiting > min_wait_time
        stuck_passengers = jnp.where(waiting_mask & long_wait_mask)[0]

        if len(stuck_passengers) == 0:
            print("No stuck passengers found.")
            return

        # Take first stuck passenger for analysis
        p_idx = int(stuck_passengers[0])

        print("\n" + "=" * 80)
        print(f"ANALYZING STUCK PASSENGER {p_idx}")
        print("=" * 80)

        print("\nBASIC INFO:")
        origin = final_state.passengers.origins[p_idx]
        dest = final_state.passengers.destinations[p_idx]
        print(f"Origin: Node {origin}")
        print(f"Destination: Node {dest}")
        print(
            f"Desired departure time: {final_state.passengers.desired_departure_times[p_idx]}"
        )
        print(
            f"Current waiting time: {final_state.passengers.time_waiting[p_idx]}"
        )
        print(f"Current time: {final_state.current_time}")
        print("\nTRANSFER INFO:")
        print(
            f"Has already transferred: {final_state.passengers.has_transferred[p_idx]}"
        )
        print(f"Transfer node: {final_state.passengers.transfer_nodes[p_idx]}")
        if final_state.passengers.statuses[p_idx] == PassengerStatus.TRANSFERRING:
            print("Currently in transfer!")
            print(
                f"Time waiting at transfer stop: {final_state.passengers.time_waiting[p_idx]}"
            )

        # Calculate route times and analyze transfer options
        route_times, route_directions = calculate_route_times(final_state)

        print("\nDIRECT ROUTE ANALYSIS:")
        # Check for direct routes first
        direct_times = route_times[:, origin, dest]
        direct_routes = jnp.where(jnp.isfinite(direct_times))[0]
        if len(direct_routes) > 0:
            print("Direct routes available:")
            for r in direct_routes:
                route = final_state.routes.stops[r]
                valid_stops = route[route != -1]
                print(
                    f"  Route {r} ({' -> '.join(str(s) for s in valid_stops)}): {direct_times[r]:.1f}"
                )
        else:
            print("No direct routes available")

        print("\nTRANSFER OPTIONS ANALYSIS:")
        best_time, transfer_node, first_leg_route, second_leg_route = (
            find_best_transfer_route(final_state, origin, dest, route_times)
        )

        if jnp.isfinite(best_time):
            print(f"Best transfer option:")
            print(f"  First leg: Route {first_leg_route} to node {transfer_node}")
            print(f"  Second leg: Route {second_leg_route} to destination")
            print(f"  Total expected time: {best_time:.1f}")

            # Show detailed path
            route1 = final_state.routes.stops[first_leg_route]
            route2 = final_state.routes.stops[second_leg_route]
            valid_stops1 = route1[route1 != -1]
            valid_stops2 = route2[route2 != -1]
            print(f"\nDetailed transfer path:")
            print(
                f"  First leg route: {' -> '.join(str(s) for s in valid_stops1)}"
            )
            print(
                f"  Second leg route: {' -> '.join(str(s) for s in valid_stops2)}"
            )
        else:
            print("No valid transfer path found!")

        print("\nRELEVANT VEHICLES ANALYSIS:")
        relevant_vehicles = []
        for v_idx in range(final_state.fleet.num_vehicles):
            route_id = final_state.fleet.route_ids[v_idx]
            pos = final_state.fleet.current_edges[v_idx, 0]
            next_pos = final_state.fleet.current_edges[v_idx, 1]
            direction = (
                "FWD"
                if final_state.fleet.directions[v_idx] == VehicleDirection.FORWARD
                else "BWD"
            )
            has_space = final_state.fleet.seat_is_available[v_idx]

            # Check if vehicle's route contains either origin or destination
            vehicle_route = final_state.routes.stops[route_id]
            valid_stops = vehicle_route[vehicle_route != -1]

            if origin in valid_stops or dest in valid_stops:
                relevant_vehicles.append(v_idx)
                print(f"\nVehicle {v_idx} (Route {route_id}):")
                print(f"  Current position: Node {pos} -> Node {next_pos}")
                print(f"  Direction: {direction}")
                print(f"  Has space: {has_space}")
                print(
                    f"  Time on edge: {final_state.fleet.times_on_edge[v_idx]:.1f}"
                )
                print(f"  Route: {' -> '.join(str(s) for s in valid_stops)}")
                print(
                    f"  Current occupancy: {final_state.fleet.num_passengers[v_idx]}/{len(final_state.fleet.passengers[v_idx])}"
                )

                # Check if origin/dest are in route
                if origin in valid_stops:
                    origin_idx = jnp.where(valid_stops == origin)[0][0]
                    print(f"  Origin is stop #{origin_idx} in route")
                if dest in valid_stops:
                    dest_idx = jnp.where(valid_stops == dest)[0][0]
                    print(f"  Destination is stop #{dest_idx} in route")

        print("\nBOARDING DECISION ANALYSIS:")
        # Get the state when passenger first became WAITING
        start_time = final_state.passengers.desired_departure_times[p_idx]
        waiting_start_idx = int(start_time)
        if waiting_start_idx < len(states):
            initial_state = states[waiting_start_idx]
            print(
                f"\nAnalyzing initial boarding opportunity at t={waiting_start_idx}:"
            )

            # Calculate waiting times for that state
            route_times, route_directions = calculate_route_times(initial_state)
            waiting_times = calculate_waiting_times(
                initial_state, route_times, route_directions
            )

            # For each relevant vehicle, show why boarding wasn't possible
            for v_idx in relevant_vehicles:
                route_id = initial_state.fleet.route_ids[v_idx]
                direction = initial_state.fleet.directions[v_idx]
                pos = initial_state.fleet.current_edges[v_idx, 0]

                print(f"\nVehicle {v_idx} (Route {route_id}):")
                is_at_stop = initial_state.fleet.is_at_node[v_idx]
                at_origin = pos == origin
                has_space = initial_state.fleet.seat_is_available[v_idx]

                required_direction = get_direction_if_connected(
                    initial_state, origin, dest
                )[route_id]
                moves_in_best_direction = (required_direction != -1) & (
                    direction == required_direction
                )

                print(f"  At correct stop: {is_at_stop and at_origin}")
                print(f"  Has space: {has_space}")
                print(f"  Current direction: {direction}")
                print(f"  Required direction: {required_direction}")
                print(f"  Moving in best direction: {moves_in_best_direction}")

                if (
                    is_at_stop
                    and at_origin
                    and has_space
                    and moves_in_best_direction
                ):
                    print("  Should have been able to board!")
                else:
                    print("  Boarding not possible due to:")
                    if not is_at_stop:
                        print("   - Vehicle not at a stop")
                    if not at_origin:
                        print("   - Vehicle not at passenger origin")
                    if not has_space:
                        print("   - No space available")
                    if not moves_in_best_direction:
                        print("   - Wrong direction")

        return p_idx  # Return passenger index for further analysis if needed


    analyze_stuck_passenger(states, min_wait_time=100.0)
    return (analyze_stuck_passenger,)


@app.cell(hide_code=True)
def _(
    Array,
    Bool,
    Float,
    Int,
    PassengerStatus,
    State,
    VehicleDirection,
    add_passenger,
    calculate_route_times,
    calculate_waiting_times,
    find_best_transfer_route,
    get_direction_if_connected,
    jax,
    jnp,
    np,
    replace,
):
    def debug_assign_single_passenger(
        state: State,
        passenger_idx: Int[Array, ""],
        route_times: Float[Array, "num_routes num_nodes num_nodes"],
        route_directions: Bool[Array, "num_routes num_nodes num_nodes"],
        waiting_times: Float[Array, "num_vehicles num_nodes 2"],
        discount_factor_future: float = 0.95,
    ) -> State:
        """Debug version with print statements of the actual assignment implementation."""
        # Calculate route times once at the start
        route_times, route_directions = calculate_route_times(state)
        all_waiting_times = calculate_waiting_times(
            state, route_times, route_directions
        )

        print("\n" + "=" * 80)
        print(f"ANALYZING PASSENGER ASSIGNMENT FOR PASSENGER {passenger_idx}")
        print("=" * 80)

        # Update origin in case passenger is transferring
        is_transferring = (
            state.passengers.statuses[passenger_idx]
            == PassengerStatus.TRANSFERRING
        )
        effective_origin = jnp.where(
            is_transferring,
            state.passengers.transfer_nodes[passenger_idx],
            state.passengers.origins[passenger_idx],
        )
        dest = state.passengers.destinations[passenger_idx]

        print(f"\nPASSENGER INFO:")
        print(
            f"Status: {PassengerStatus(state.passengers.statuses[passenger_idx]).name}"
        )
        print(f"Current location: Node {effective_origin + 1}")
        print(f"Destination: Node {dest + 1}")

        # Try direct route first
        in_vehicle_times = route_times[:, effective_origin, dest]
        has_direct_route = jnp.any(jnp.isfinite(in_vehicle_times))
        print(f"\nDirect routes available: {has_direct_route}")

        # If no direct route, find best transfer
        _, transfer_node, _, _ = jax.lax.cond(
            has_direct_route,
            lambda: (jnp.inf, jnp.array(-1), jnp.array(-1), jnp.array(-1)),
            lambda: find_best_transfer_route(
                state, effective_origin, dest, route_times
            ),
        )
        in_vehicle_times = jax.lax.cond(
            has_direct_route,
            lambda: in_vehicle_times,
            lambda: route_times[:, effective_origin, transfer_node],
        )

        # Calculate best direction for each vehicle
        required_directions = get_direction_if_connected(
            state, effective_origin, dest
        )
        best_direction = required_directions[
            state.fleet.route_ids
        ]  # routes -> vehicles

        # Calculate overall time
        route_ids = state.fleet.route_ids
        in_vehicle_times = in_vehicle_times[
            route_ids
        ]  # num of routes -> num vehicles
        wait_times = all_waiting_times[:, :, best_direction]
        journey_times = wait_times + in_vehicle_times

        print("\nVEHICLE ANALYSIS:")
        print("-" * 100)
        print(
            "Veh | Rt  | Pos | Dir | Wait | In-Veh | Total | At Stop | Has Space | Best Dir | Valid"
        )
        print("-" * 100)

        is_at_correct_stop = state.fleet.is_at_node & (
            state.fleet.current_edges[:, 0] == effective_origin
        )
        immediate_boarding_possible = (
            is_at_correct_stop & state.fleet.seat_is_available
        )

        # Print detailed vehicle info
        for v_idx in range(state.fleet.num_vehicles):
            route_id = route_ids[v_idx]
            pos = state.fleet.current_edges[v_idx, 0]
            dir_str = (
                "FWD"
                if state.fleet.directions[v_idx] == VehicleDirection.FORWARD
                else "BWD"
            )
            wait = wait_times[v_idx]
            in_veh = in_vehicle_times[v_idx]
            total = journey_times[v_idx]
            at_stop = is_at_correct_stop[v_idx]
            has_space = state.fleet.seat_is_available[v_idx]
            best_dir = (
                "FWD"
                if best_direction[v_idx] == VehicleDirection.FORWARD
                else "BWD"
            )
            valid = immediate_boarding_possible[v_idx]

            print(
                f"{v_idx:3d} | {route_id + 1:3d} | {pos + 1:3d} | {dir_str} | {wait:4.1f} | "
                f"{in_veh:6.1f} | {total:5.1f} | {at_stop:8} | {has_space:9} | {best_dir:8} | {valid}"
            )

        # Check route validity and calculate options
        connects_target = jnp.isfinite(journey_times)
        immediate_times = jnp.where(
            immediate_boarding_possible & connects_target, journey_times, jnp.inf
        )
        future_times = jnp.where(
            connects_target & state.fleet.seat_is_available, journey_times, jnp.inf
        )

        # Board if immediate option exists and no significantly better future option exists
        best_future_time = jnp.min(future_times)
        should_board = jnp.isfinite(immediate_times) & (
            best_future_time >= immediate_times * discount_factor_future
        )

        # Among valid options, pick one with shortest travel time
        valid_times = jnp.where(should_board, immediate_times, jnp.inf)
        best_time = jnp.min(valid_times)
        has_best_time = valid_times == best_time

        # Among those with best time, pick one with highest capacity
        best_vehicle = jnp.argmax(has_best_time * state.fleet.capacities_left)

        print("\nDECISION ANALYSIS:")
        print(f"Best immediate time: {best_time:.1f}")
        print(f"Best future time: {best_future_time:.1f}")
        print(f"Vehicles that should board: {np.where(should_board)[0]}")
        print(f"Vehicles with best time: {np.where(has_best_time)[0]}")
        print(f"Selected vehicle: {best_vehicle}")

        def update_state(state: State) -> State:
            new_fleet = add_passenger(state.fleet, best_vehicle, passenger_idx)
            new_statuses = state.passengers.statuses.at[passenger_idx].set(
                PassengerStatus.IN_VEHICLE
            )
            new_transfer_nodes = state.passengers.transfer_nodes.at[
                passenger_idx
            ].set(transfer_node)
            new_passengers = replace(
                state.passengers,
                statuses=new_statuses,
                transfer_nodes=new_transfer_nodes,
            )
            return replace(state, fleet=new_fleet, passengers=new_passengers)

        print("\nKey values at assignment time:")
        print(f"best_vehicle: {best_vehicle}")
        print(f"should_board[best_vehicle]: {should_board[best_vehicle]}")
        print(f"immediate_times: {immediate_times}")
        print(f"valid_times: {valid_times}")
        print(f"has_best_time: {has_best_time}")
        print(f"state.fleet.capacities_left: {state.fleet.capacities_left}")

        print("\nState just before boarding decision:")
        print(
            f"Passenger status: {PassengerStatus(state.passengers.statuses[passenger_idx]).name}"
        )
        print(f"Best vehicle selected: {best_vehicle}")

        new_state = jax.lax.cond(
            should_board[best_vehicle],
            update_state,
            lambda s: s,
            state,
        )
        print("\nState after update:")
        print(
            f"Passenger status: {PassengerStatus(new_state.passengers.statuses[passenger_idx]).name}"
        )
        print("Vehicle assignments:")
        for v_idx in range(new_state.fleet.num_vehicles):
            if passenger_idx in new_state.fleet.passengers[v_idx]:
                print(f"Passenger {passenger_idx} found in vehicle {v_idx}")

        print(
            f"\nFinal decision: {'Board vehicle ' + str(best_vehicle) if should_board[best_vehicle] else 'Wait'}"
        )

        return new_state
    return (debug_assign_single_passenger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run simulation""")
    return


@app.cell
def _(Mandl, PassengerStatus, jax, jnp):
    # Create environment
    n_steps = (24 * 60) + 1000
    key = jax.random.PRNGKey(42)
    env = Mandl(
        network_name="mandl1",
        runtime=n_steps - 1000,
        vehicle_capacity=50,
        num_flex_routes=0,
        max_route_length=0,
    )

    # Reset environment and get initial state
    state, timestep = env.reset(key)

    print("\nInitial State Info:")
    print(f"Number of vehicles: {state.fleet.num_vehicles}")
    print(f"Number of passengers: {state.passengers.num_passengers}")
    print(f"Number of routes: {state.routes.num_routes}")
    print(f"Network nodes: {state.network.num_nodes}")

    # Simulate a few steps
    print("\nSimulating steps...")
    states = [state]
    step = jax.jit(env.step)


    # rewrite loop as while loop that stops when all passengers are delivered
    # show progress bar over num_passengers completed
    n = 0

    while jnp.any(state.passengers.statuses != PassengerStatus.COMPLETED):
        print(
            f"Number of passengers NOT_IN_SYSTEM: {jnp.sum(state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM)}"
        )
        print(
            f"Number of passengers WAITING: {jnp.sum(state.passengers.statuses == PassengerStatus.WAITING)}"
        )
        print(
            f"Number of passengers TRANSFERRING: {jnp.sum(state.passengers.statuses == PassengerStatus.TRANSFERRING)}"
        )
        print(
            f"Number of passengers IN_VEHICLE: {jnp.sum(state.passengers.statuses == PassengerStatus.IN_VEHICLE)}"
        )
        print(
            f"Number of passengers COMPLETED: {jnp.sum(state.passengers.statuses == PassengerStatus.COMPLETED)}"
        )
        print(f"step: {n}\n")
        n += 1

        # For now, just use dummy action (no-op for all flexible routes)
        action = jax.numpy.full(
            state.routes.num_routes,
            state.network.num_nodes,  # no-op action
            dtype=int,
        )

        state, timestep = step(state, action)
        states.append(state)

        if n > 2_000:
            break

    print(
        f"Number of passengers NOT_IN_SYSTEM: {jnp.sum(state.passengers.statuses == PassengerStatus.NOT_IN_SYSTEM)}"
    )
    print(
        f"Number of passengers WAITING: {jnp.sum(state.passengers.statuses == PassengerStatus.WAITING)}"
    )
    print(
        f"Number of passengers TRANSFERRING: {jnp.sum(state.passengers.statuses == PassengerStatus.TRANSFERRING)}"
    )
    print(
        f"Number of passengers IN_VEHICLE: {jnp.sum(state.passengers.statuses == PassengerStatus.IN_VEHICLE)}"
    )
    print(
        f"Number of passengers COMPLETED: {jnp.sum(state.passengers.statuses == PassengerStatus.COMPLETED)}"
    )
    print(f"step: {n}\n")
    final_state = state
    return (
        action,
        env,
        final_state,
        key,
        n,
        n_steps,
        state,
        states,
        step,
        timestep,
    )


@app.cell
def _(env, final_state, plt):
    env.render(final_state)
    plt.show()
    return


@app.cell(hide_code=True)
def _(state):
    # Print timing statistics
    print("\nPassenger Statistics:")
    print(f"Number of passengers: {state.passengers.num_passengers}")
    print(f"Average waiting time: {state.passengers.time_waiting.mean():.2f}")
    print(
        f"Average in-vehicle time: {state.passengers.time_in_vehicle.mean():.2f}"
    )
    print(
        f"Average total time: {(state.passengers.time_waiting + state.passengers.time_in_vehicle).mean():.2f}"
    )
    print(f"Maximum waiting time: {state.passengers.time_waiting.max():.2f}")
    print(f"Maximum in-vehicle time: {state.passengers.time_in_vehicle.max():.2f}")
    print(f"Sum of waiting times: {state.passengers.time_waiting.sum():.2f}")
    print(f"Sum of in-vehicle times: {state.passengers.time_in_vehicle.sum():.2f}")
    print(
        f"Total Travel time: {state.passengers.time_in_vehicle.sum() + state.passengers.time_waiting.sum():.2f}"
    )
    return


@app.cell
def _(plt, state):
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

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(env, plt, states):
    env.animate(states[:10], interval=800, save_path="mandl_simulation.mp4")
    plt.show()
    return


@app.cell
def _():
    # busse einfärben in linien farbe

    # Andere visualisierung:
    # farbe der busse zeigt auslastung
    # pfeile zwischen OD zeigt demand
    # pfeile zwischen OD zeigt total travel time
    # pfeile zwischen OD zeigt kürzeste route
    # pfeile zwischen OD zeigt vehältnis von actual total travel time zu optimaler route
    return


@app.cell(hide_code=True)
def _(State, np, plt, rgb, state, svgwrite):
    def create_network_svg(
        state: State, width: int = 600, height: int = 600, margin: int = 50
    ) -> svgwrite.Drawing:
        """Create an SVG visualization of the network."""
        dwg = svgwrite.Drawing(size=(width, height))

        # Add white background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

        # Get node coordinates
        nodes = np.array(state.network.node_coordinates)
        x_min, y_min = nodes.min(axis=0)
        x_max, y_max = nodes.max(axis=0)

        def scale_for_svg(points: np.ndarray) -> np.ndarray:
            """Scale array of points to SVG coordinates"""
            scaled = np.zeros_like(points, dtype=float)
            scaled[:, 0] = margin + (points[:, 0] - x_min) / (x_max - x_min) * (
                width - 2 * margin
            )
            scaled[:, 1] = margin + (points[:, 1] - y_min) / (y_max - y_min) * (
                height - 2 * margin
            )
            return scaled

        scaled_nodes = scale_for_svg(nodes)

        # Create symbols definitions
        defs = dwg.defs

        # Define bus symbol
        bus_symbol = dwg.symbol(id="bus", viewBox="0 0 100 100")
        bus_symbol.add(dwg.rect((6, 3), (6, 3), rx=0, ry=0))
        defs.add(bus_symbol)

        # Draw base edges first
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if state.network.travel_times[i, j] < float("inf"):
                    dwg.add(
                        dwg.line(
                            start=tuple(scaled_nodes[i]),
                            end=tuple(scaled_nodes[j]),
                            stroke="black",
                            stroke_width=2,
                            opacity=1,
                        )
                    )

        # Create route segment symbols
        ROUTE_OFFSET = 3  # pixels
        routes_list = []
        for i in range(state.routes.stops.shape[0]):
            route = state.routes.stops[i]
            valid_stops = route[route != -1]
            routes_list.append(tuple(valid_stops.tolist()))

        unique_routes = list(set(routes_list))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_routes)))
        route_color_dict = dict(zip(unique_routes, colors, strict=False))

        # Assign offsets to unique routes
        route_offsets = {
            route: ((-1) ** i) * (1 + i // 2) * ROUTE_OFFSET
            for i, route in enumerate(unique_routes)
        }

        for route_idx, route in enumerate(routes_list):
            color = route_color_dict[route]
            svg_color = rgb(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            offset = route_offsets[route]

            # Create route symbol
            route_id = f"route_{route_idx}"
            route_symbol = dwg.symbol(id=route_id)

            # Add each segment
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]

                from_pos = scaled_nodes[from_node]
                to_pos = scaled_nodes[to_node]

                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                length = np.sqrt(dx**2 + dy**2)
                angle = np.degrees(np.arctan2(dy, dx))

                # Calculate perpendicular offset
                if length > 0:
                    perpx = -dy / length * offset
                    perpy = dx / length * offset

                    # Apply offset to positions
                    from_pos_offset = from_pos + np.array([perpx, perpy])
                    to_pos_offset = to_pos + np.array([perpx, perpy])

                    # Draw the route segment directly instead of using edge symbol
                    route_symbol.add(
                        dwg.line(
                            start=tuple(from_pos_offset),
                            end=tuple(to_pos_offset),
                            stroke=svg_color,
                            stroke_width=2,
                        )
                    )

            defs.add(route_symbol)

            # Add route instance to main drawing
            dwg.add(dwg.use(f"#route_{route_idx}"))

        # Draw nodes
        node_radius = 20
        for i, (x, y) in enumerate(scaled_nodes):
            dwg.add(
                dwg.circle(
                    center=(x, y),
                    r=node_radius,
                    fill="white",
                    stroke="black",
                    stroke_width=2,
                )
            )
            dwg.add(
                dwg.text(
                    str(i + 1),
                    insert=(x, y + 6),
                    text_anchor="middle",
                    font_size=14,
                )
            )

        return dwg


    create_network_svg(state)
    return (create_network_svg,)


@app.cell(hide_code=True)
def _(State, np, pl, states):
    def collect_raw_state_data(states: list[State]) -> dict[str, pl.DataFrame]:
        """Collect raw data from a list of Mandl environment states without aggregation.

        Args:
            states: List of State objects from the Mandl environment

        Returns:
            Dictionary of DataFrames with raw data over time
        """
        # Initialize empty lists to store time series data
        fleet_data = []
        passenger_data = []
        route_data = []
        system_data = []

        for t, state in enumerate(states):
            # 1. Fleet data - one row per vehicle at each timestep
            fleet_df = pl.DataFrame(
                {
                    "time": np.full(state.fleet.num_vehicles, t),
                    "vehicle_id": np.arange(state.fleet.num_vehicles),
                    "route_id": np.array(state.fleet.route_ids),
                    "current_from": np.array(state.fleet.current_edges)[:, 0],
                    "current_to": np.array(state.fleet.current_edges)[:, 1],
                    "time_on_edge": np.array(state.fleet.times_on_edge),
                    "direction": np.array(state.fleet.directions),
                    "at_node": np.array(state.fleet.is_at_node),
                    "num_passengers": np.array(state.fleet.num_passengers),
                }
            )

            # Add passenger data for each vehicle
            for seat_idx in range(state.fleet.passengers.shape[1]):
                fleet_df = fleet_df.with_columns(
                    pl.lit(np.array(state.fleet.passengers)[:, seat_idx]).alias(
                        f"passenger_{seat_idx}"
                    )
                )

            fleet_data.append(fleet_df)

            # 2. Passenger data - one row per passenger at each timestep
            passenger_df = pl.DataFrame(
                {
                    "time": np.full(state.passengers.num_passengers, t),
                    "passenger_id": np.arange(state.passengers.num_passengers),
                    "origin": np.array(state.passengers.origins),
                    "destination": np.array(state.passengers.destinations),
                    "desired_departure_time": np.array(
                        state.passengers.desired_departure_times
                    ),
                    "time_waiting": np.array(state.passengers.time_waiting),
                    "time_in_vehicle": np.array(state.passengers.time_in_vehicle),
                    "status": np.array(state.passengers.statuses),
                    "has_transferred": np.array(state.passengers.has_transferred),
                    "transfer_node": np.array(state.passengers.transfer_nodes),
                }
            )
            passenger_data.append(passenger_df)

            # 3. Route data - one row per route at each timestep
            num_routes = state.routes.num_routes
            route_df = pl.DataFrame(
                {
                    "time": np.full(num_routes, t),
                    "route_id": np.arange(num_routes),
                    "route_type": np.array(state.routes.types),
                    "frequency": np.array(state.routes.frequencies),
                }
            )

            # Add stops data for each route
            for stop_idx in range(state.routes.max_route_length):
                route_df = route_df.with_columns(
                    pl.lit(np.array(state.routes.stops)[:, stop_idx]).alias(
                        f"stop_{stop_idx}"
                    )
                )

            route_data.append(route_df)

        # Combine all time steps into single DataFrames
        combined_fleet_data = pl.concat(fleet_data)
        combined_passenger_data = pl.concat(passenger_data)
        combined_route_data = pl.concat(route_data)

        return {
            "fleet": combined_fleet_data,
            "passengers": combined_passenger_data,
            "routes": combined_route_data,
        }


    stats_over_time = collect_raw_state_data(states)
    fleet_df = stats_over_time["fleet"]
    passenger_df = stats_over_time["passengers"]
    routes_df = stats_over_time["routes"]
    return (
        collect_raw_state_data,
        fleet_df,
        passenger_df,
        routes_df,
        stats_over_time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import tqdm
    import polars as pl
    import numpy as np
    import altair as alt
    import hvplot.polars

    alt.data_transformers.enable("vegafusion")
    alt.renderers.enable("default")

    from jumanji.environments.routing.mandl import Mandl
    from jumanji.environments.routing.mandl.types import PassengerStatus, State

    from typing import Optional

    import jax.numpy as jnp
    import matplotlib.colors as mcolors
    import svgwrite
    from svgwrite import rgb

    from jumanji.environments.routing.mandl.types import (
        VehicleDirection,
        calculate_waiting_times,
        calculate_route_times,
        find_best_transfer_route,
        PassengerStatus,
        State,
        add_passenger,
        update_passengers_to_waiting,
        update_routes,
        move_vehicles,
        increment_wait_times,
        increment_in_vehicle_times,
        handle_completed_and_transferring_passengers,
        get_position_in_route,
        get_direction_if_connected,
    )
    from jaxtyping import Int, Float, Bool, Array
    from dataclasses import replace
    return (
        Array,
        Bool,
        Float,
        Int,
        Mandl,
        Optional,
        PassengerStatus,
        State,
        VehicleDirection,
        add_passenger,
        alt,
        calculate_route_times,
        calculate_waiting_times,
        find_best_transfer_route,
        get_direction_if_connected,
        get_position_in_route,
        handle_completed_and_transferring_passengers,
        hvplot,
        increment_in_vehicle_times,
        increment_wait_times,
        jax,
        jnp,
        mcolors,
        mo,
        move_vehicles,
        np,
        pl,
        plt,
        replace,
        rgb,
        svgwrite,
        tqdm,
        update_passengers_to_waiting,
        update_routes,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
