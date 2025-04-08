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

__generated_with = "0.12.4"
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
            column="status:N",
            color=alt.Color("count")
            .scale(scheme="greenblue")
            .title("Number of Passengers"),
            tooltip=["origin", "destination", "count"],
        )
    )
    return


@app.cell
def _(last_state_passenger_df, pl):
    _od_counts = last_state_passenger_df.group_by(
        ["origin", "destination", "status"]
    ).agg(pl.count("passenger_id").alias("count"))


    _od_counts = _od_counts.to_pandas()

    from plotnine import (
        ggplot,
        geom_tile,
        facet_grid,
        scale_fill_gradient,
        theme_minimal,
        theme,
        element_text,
        labs,
        aes,
        geom_text,
        element_blank,
        element_rect,
        scale_x_discrete,
        scale_y_discrete,
        scale_color_continuous,
        scale_fill_cmap,
    )

    origins = sorted(_od_counts["origin"].unique())
    destinations = sorted(_od_counts["destination"].unique())

    _p = (
        ggplot(_od_counts, aes(x="origin", y="destination", fill="count"))
        + geom_tile(aes(width=1.00, height=1.00))
        # + geom_text(aes(label='count'), show_legend=False)
        + scale_fill_cmap(cmap_name="GnBu", name="Number of Passengers")
        + scale_x_discrete(limits=origins)
        + scale_y_discrete(limits=destinations)
        + theme(
            panel_background=element_rect(fill="white"),
            # axis_ticks=element_blank(),
            panel_grid=element_blank(),  # remove grid lines
            # panel_border=element_blank(),  # remove border
            axis_line=element_blank(),  # remove axis lines
            figure_size=(5.78, 4),
        )
    )

    _p.draw()
    return (
        aes,
        destinations,
        element_blank,
        element_rect,
        element_text,
        facet_grid,
        geom_text,
        geom_tile,
        ggplot,
        labs,
        origins,
        scale_color_continuous,
        scale_fill_cmap,
        scale_fill_gradient,
        scale_x_discrete,
        scale_y_discrete,
        theme,
        theme_minimal,
    )


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


@app.cell(disabled=True, hide_code=True)
def _(mo):
    mo.md(r"""Which passengers could not be served? The ones with the high waiting times at origin=0?""")
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
            y="origin:N",
            x="destination:N",
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
            y="origin:N",
            x="destination:N",
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
            y="origin:N",
            x="destination:N",
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


@app.cell(disabled=True, hide_code=True)
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
def _(PassengerStatus, State, get_vehicles_position_and_dest_node, jnp, state):
    def analyze_passenger_handling(state: State) -> None:
        """Analyze the passenger handling logic for debugging purposes."""

        # 1. First analyze vehicles at stops
        is_at_stop = state.fleet.is_at_node
        print("\nVEHICLE STOP ANALYSIS:")
        print(f"Total vehicles at stops: {is_at_stop.sum()} / {len(is_at_stop)}")

        # 2. Analyze passenger statuses and locations
        current_from, current_to = get_vehicles_position_and_dest_node(state)

        print("\nPASSENGER STATUS COUNTS:")
        for status in PassengerStatus:
            count = (state.passengers.statuses == status).sum()
            print(f"{status.name}: {count}")

        # 3. Analyze passengers in vehicles
        print("\nVEHICLE PASSENGER ANALYSIS:")
        for v_idx in range(state.fleet.num_vehicles):
            if state.fleet.is_at_node[v_idx]:
                route_id = state.fleet.route_ids[v_idx]
                pos = current_from[v_idx]
                passengers_in_vehicle = state.fleet.passengers[v_idx]
                valid_passengers = passengers_in_vehicle[
                    passengers_in_vehicle != -1
                ]

                if len(valid_passengers) > 0:
                    print(f"\nVehicle {v_idx} at node {pos} (Route {route_id}):")
                    print(f"Passengers: {valid_passengers}")

                    # Check each passenger's destination
                    for p_idx in valid_passengers:
                        dest = state.passengers.destinations[p_idx]
                        transfer_node = state.passengers.transfer_nodes[p_idx]
                        has_transferred = state.passengers.has_transferred[p_idx]

                        print(f"  Passenger {p_idx}:")
                        print(f"    Destination: {dest}")
                        print(f"    At destination: {pos == dest}")
                        print(f"    Transfer node: {transfer_node}")
                        print(f"    Has transferred: {has_transferred}")
                        print(f"    At transfer node: {pos == transfer_node}")
                        print(
                            f"    Should exit: {pos == dest or (pos == transfer_node and not has_transferred)}"
                        )

        # 4. Analyze transfer opportunities
        print("\nTRANSFER ANALYSIS:")
        transferring_mask = (
            state.passengers.statuses == PassengerStatus.TRANSFERRING
        )
        transferring_passengers = jnp.where(transferring_mask)[0]

        if len(transferring_passengers) > 0:
            print(
                f"\nFound {len(transferring_passengers)} transferring passengers:"
            )
            for p_idx in transferring_passengers:
                transfer_node = state.passengers.transfer_nodes[p_idx]
                dest = state.passengers.destinations[p_idx]
                wait_time = state.passengers.time_waiting[p_idx]

                print(f"\nPassenger {p_idx}:")
                print(f"  Transfer node: {transfer_node}")
                print(f"  Final destination: {dest}")
                print(f"  Waiting time: {wait_time}")

                # Check available vehicles at transfer node
                vehicles_at_node = (
                    current_from == transfer_node
                ) & state.fleet.is_at_node
                if vehicles_at_node.any():
                    print("  Available vehicles at transfer node:")
                    for v_idx in jnp.where(vehicles_at_node)[0]:
                        route_id = state.fleet.route_ids[v_idx]
                        has_space = state.fleet.seat_is_available[v_idx]
                        print(
                            f"    Vehicle {v_idx} (Route {route_id}, Has space: {has_space})"
                        )
                else:
                    print("  No vehicles currently at transfer node")

        return None


    # Call this function at key points in your simulation
    analyze_passenger_handling(state)
    return (analyze_passenger_handling,)


@app.cell
def _(State, get_vehicles_position_and_dest_node, jnp, states):
    def analyze_vehicle_positions(state: State) -> None:
        """Detailed analysis of vehicle positions and stop detection."""
        print("\nDETAILED VEHICLE POSITION ANALYSIS:")

        # Get vehicle positions
        current_from, current_to = get_vehicles_position_and_dest_node(state)

        for v_idx in range(state.fleet.num_vehicles):
            route_id = state.fleet.route_ids[v_idx]
            time_on_edge = state.fleet.times_on_edge[v_idx]
            direction = state.fleet.directions[v_idx]
            is_at_node = state.fleet.is_at_node[v_idx]

            # Get the route
            route = state.routes.stops[route_id]
            valid_stops = route[route != -1]

            print(f"\nVehicle {v_idx} (Route {route_id}):")
            print(
                f"  Current position: Node {current_from[v_idx]} -> Node {current_to[v_idx]}"
            )
            print(f"  Time on edge: {time_on_edge:.6f}")
            print(f"  Direction: {'Forward' if direction == 0 else 'Backward'}")
            print(f"  Is at node (according to is_at_node): {is_at_node}")
            print(f"  Route: {' -> '.join(str(s) for s in valid_stops)}")

            # Check if vehicle should be at a node
            if jnp.isclose(time_on_edge, 0.0, rtol=1e-5, atol=1e-8):
                print("  Should be at node (time_on_edge ≈ 0)")
                if not is_at_node:
                    print(
                        "  WARNING: Vehicle should be at node but is_at_node is False!"
                    )

            # Get travel time for current edge
            travel_time = state.network.travel_times[
                current_from[v_idx], current_to[v_idx]
            ]
            print(f"  Edge travel time: {travel_time:.1f}")
            print(f"  Progress on edge: {(time_on_edge / travel_time) * 100:.1f}%")

        # Check is_at_node computation
        print("\nIS_AT_NODE COMPUTATION CHECK:")
        print(f"times_on_edge: {state.fleet.times_on_edge}")
        print(
            f"is_close_to_zero: {jnp.isclose(state.fleet.times_on_edge, 0.0, rtol=1e-5, atol=1e-8)}"
        )
        print(f"is_at_node: {state.fleet.is_at_node}")


    analyze_vehicle_positions(states[-2])
    return (analyze_vehicle_positions,)


@app.cell(disabled=True)
def _(
    PassengerStatus,
    State,
    get_vehicles_position_and_dest_node,
    jnp,
    states,
):
    def analyze_stuck_in_vehicle_passengers(state: State) -> None:
        """Analyze passengers who should exit vehicles but don't."""
        print("\nANALYZING PASSENGERS THAT SHOULD EXIT VEHICLES:")

        # Get vehicle positions
        current_from, current_to = get_vehicles_position_and_dest_node(state)

        for v_idx in range(state.fleet.num_vehicles):
            if state.fleet.is_at_node[v_idx]:
                current_node = current_from[v_idx]
                route_id = state.fleet.route_ids[v_idx]
                passengers = state.fleet.passengers[v_idx]
                valid_passengers = passengers[passengers != -1]

                if len(valid_passengers) > 0:
                    print(
                        f"\nVehicle {v_idx} at node {current_node} (Route {route_id}):"
                    )

                    for p_idx in valid_passengers:
                        dest = state.passengers.destinations[p_idx]
                        transfer_node = state.passengers.transfer_nodes[p_idx]
                        has_transferred = state.passengers.has_transferred[p_idx]
                        status = state.passengers.statuses[p_idx]
                        time_in_vehicle = state.passengers.time_in_vehicle[p_idx]

                        should_exit_dest = current_node == dest
                        should_exit_transfer = (
                            current_node == transfer_node
                        ) & ~has_transferred
                        should_exit = should_exit_dest | should_exit_transfer

                        if should_exit:
                            print(f"\n  Passenger {p_idx} should exit:")
                            print(f"    Current node: {current_node}")
                            print(f"    Destination: {dest}")
                            print(f"    Transfer node: {transfer_node}")
                            print(f"    Has transferred: {has_transferred}")
                            print(
                                f"    Current status: {PassengerStatus(status).name}"
                            )
                            print(f"    Time in vehicle: {time_in_vehicle}")
                            print(
                                f"    Should exit due to: {'Destination' if should_exit_dest else 'Transfer'}"
                            )

                            # Analyze route coverage
                            route = state.routes.stops[route_id]
                            valid_stops = route[route != -1]
                            print(
                                f"    Current route: {' -> '.join(str(s) for s in valid_stops)}"
                            )

                            if should_exit_transfer:
                                # Check if any route can complete the journey
                                for r_idx in range(state.routes.num_routes):
                                    route = state.routes.stops[r_idx]
                                    valid_stops = route[route != -1]
                                    if (
                                        transfer_node in valid_stops
                                        and dest in valid_stops
                                    ):
                                        transfer_idx = jnp.where(
                                            valid_stops == transfer_node
                                        )[0][0]
                                        dest_idx = jnp.where(valid_stops == dest)[
                                            0
                                        ][0]
                                        if transfer_idx < dest_idx:
                                            print(
                                                f"    Available connecting route {r_idx}: {' -> '.join(str(s) for s in valid_stops)}"
                                            )

        # Also check for any IN_VEHICLE passengers at their destinations
        in_vehicle_mask = state.passengers.statuses == PassengerStatus.IN_VEHICLE
        if in_vehicle_mask.any():
            print("\nChecking all IN_VEHICLE passengers:")
            in_vehicle_passengers = jnp.where(in_vehicle_mask)[0]

            for p_idx in in_vehicle_passengers:
                # Find which vehicle the passenger is in
                for v_idx in range(state.fleet.num_vehicles):
                    if p_idx in state.fleet.passengers[v_idx]:
                        current_node = current_from[v_idx]
                        dest = state.passengers.destinations[p_idx]
                        transfer_node = state.passengers.transfer_nodes[p_idx]
                        has_transferred = state.passengers.has_transferred[p_idx]
                        time_in_vehicle = state.passengers.time_in_vehicle[p_idx]

                        if (current_node == dest) or (
                            current_node == transfer_node and not has_transferred
                        ):
                            print(
                                f"\nPassenger {p_idx} in vehicle {v_idx} should exit at node {current_node}:"
                            )
                            print(f"  Destination: {dest}")
                            print(f"  Transfer node: {transfer_node}")
                            print(f"  Has transferred: {has_transferred}")
                            print(f"  Time in vehicle: {time_in_vehicle}")
                            print(
                                f"  Vehicle is_at_node: {state.fleet.is_at_node[v_idx]}"
                            )
                            print(
                                f"  Vehicle time_on_edge: {state.fleet.times_on_edge[v_idx]}"
                            )


    analyze_stuck_in_vehicle_passengers(states[50])
    return (analyze_stuck_in_vehicle_passengers,)


@app.cell
def _(PassengerStatus, State, get_vehicles_position_and_dest_node, states):
    def analyze_all_vehicle_passengers(state: State) -> None:
        """Show detailed info for all passengers in vehicles."""
        print("\nDETAILED VEHICLE PASSENGER ANALYSIS:")

        current_from, current_to = get_vehicles_position_and_dest_node(state)

        for v_idx in range(state.fleet.num_vehicles):
            passengers = state.fleet.passengers[v_idx]
            valid_passengers = passengers[passengers != -1]

            if len(valid_passengers) > 0:
                current_node = current_from[v_idx]
                route_id = state.fleet.route_ids[v_idx]
                route = state.routes.stops[route_id]
                valid_stops = route[route != -1]

                print(
                    f"\nVehicle {v_idx} at/between node(s) {current_from[v_idx]} -> {current_to[v_idx]}:"
                )
                print(
                    f"  Route {route_id}: {' -> '.join(str(s) for s in valid_stops)}"
                )
                print(f"  Is at node: {state.fleet.is_at_node[v_idx]}")
                print(f"  Time on edge: {state.fleet.times_on_edge[v_idx]}")
                print(
                    f"  Direction: {'Forward' if state.fleet.directions[v_idx] == 0 else 'Backward'}"
                )
                print(f"  Passengers ({len(valid_passengers)}):")

                for p_idx in valid_passengers:
                    dest = state.passengers.destinations[p_idx]
                    transfer_node = state.passengers.transfer_nodes[p_idx]
                    has_transferred = state.passengers.has_transferred[p_idx]
                    status = state.passengers.statuses[p_idx]
                    time_in_vehicle = state.passengers.time_in_vehicle[p_idx]

                    print(f"\n    Passenger {p_idx}:")
                    print(f"      Status: {PassengerStatus(status).name}")
                    print(
                        f"      Current location: {current_from[v_idx]} -> {current_to[v_idx]}"
                    )
                    print(f"      Destination: {dest}")
                    print(f"      Transfer node: {transfer_node}")
                    print(f"      Has transferred: {has_transferred}")
                    print(f"      Time in vehicle: {time_in_vehicle}")

                    # Check if should exit
                    should_exit_dest = (
                        current_node == dest and state.fleet.is_at_node[v_idx]
                    )
                    should_exit_transfer = (
                        current_node == transfer_node
                        and not has_transferred
                        and state.fleet.is_at_node[v_idx]
                    )

                    if should_exit_dest:
                        print(f"      *** Should exit - at destination! ***")
                    elif should_exit_transfer:
                        print(f"      *** Should exit - at transfer node! ***")

        print("\nSummary:")
        print(
            f"Total passengers in vehicles: {(state.fleet.passengers != -1).sum()}"
        )
        print(
            f"Total vehicles with passengers: {((state.fleet.passengers != -1).sum(axis=1) > 0).sum()}"
        )


    analyze_all_vehicle_passengers(states[-15])
    return (analyze_all_vehicle_passengers,)


@app.cell
def _(State, jnp, states):
    def analyze_passenger_assignment(state: State, passenger_idx: int) -> None:
        """Analyze why a passenger was assigned to a particular route."""
        print(f"\nAnalyzing assignment for passenger {passenger_idx}:")
        origin = state.passengers.origins[passenger_idx]
        dest = state.passengers.destinations[passenger_idx]
        print(f"Origin: {origin}, Destination: {dest}")

        # Check all routes
        for r_idx in range(state.routes.num_routes):
            route = state.routes.stops[r_idx]
            valid_stops = route[route != -1]
            print(f"\nRoute {r_idx}: {' -> '.join(str(s) for s in valid_stops)}")

            # Check if origin and destination are on route
            origin_on_route = origin in valid_stops
            dest_on_route = dest in valid_stops
            print(f"Origin on route: {origin_on_route}")
            print(f"Destination on route: {dest_on_route}")

            if origin_on_route and dest_on_route:
                origin_idx = jnp.where(valid_stops == origin)[0][0]
                dest_idx = jnp.where(valid_stops == dest)[0][0]
                print(f"Origin at position: {origin_idx}")
                print(f"Destination at position: {dest_idx}")
                print(
                    f"Correct direction: {'Forward' if dest_idx > origin_idx else 'Backward'}"
                )


    analyze_passenger_assignment(states[-15], 28)
    return (analyze_passenger_assignment,)


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


@app.cell
def _(
    PassengerStatus,
    State,
    calculate_route_times,
    calculate_travel_times,
    find_best_transfer_route,
    get_vehicles_position_and_dest_node,
    jnp,
    np,
    states,
):
    def identify_stranded_passengers(
        final_state: State, limit: int = 5
    ) -> list[int]:
        """Find passengers who never completed their journey."""
        waiting_mask = final_state.passengers.statuses == PassengerStatus.WAITING
        transferring_mask = (
            final_state.passengers.statuses == PassengerStatus.TRANSFERRING
        )
        stranded_mask = waiting_mask | transferring_mask

        stranded_indices = jnp.where(stranded_mask)[0]
        num_stranded = len(stranded_indices)
        total_passengers = final_state.passengers.num_passengers

        print(
            f"Found {num_stranded} stranded passengers out of {total_passengers} ({num_stranded / total_passengers * 100:.1f}%)"
        )

        # Get some additional stats about the stranded passengers
        if num_stranded > 0:
            waiting_time = final_state.passengers.time_waiting[stranded_indices]
            print(f"Average waiting time: {waiting_time.mean():.1f}")
            print(f"Max waiting time: {waiting_time.max():.1f}")

            # Show distribution of origins/destinations
            origins = final_state.passengers.origins[stranded_indices]
            destinations = final_state.passengers.destinations[stranded_indices]

            print("\nTop origin nodes for stranded passengers:")
            unique_origins, counts = np.unique(origins, return_counts=True)
            for idx in np.argsort(-counts)[:3]:  # Top 3
                print(f"  Node {unique_origins[idx]}: {counts[idx]} passengers")

            print("\nTop destination nodes for stranded passengers:")
            unique_dests, counts = np.unique(destinations, return_counts=True)
            for idx in np.argsort(-counts)[:3]:  # Top 3
                print(f"  Node {unique_dests[idx]}: {counts[idx]} passengers")

        # Return a sample of stranded passengers
        return stranded_indices[:limit].tolist()


    def analyze_boarding_opportunities(
        states: list[State], passenger_idx: int
    ) -> None:
        """Analyze specific instances where a passenger should have boarded a vehicle."""
        print(f"\n{'=' * 80}")
        print(f"ANALYZING BOARDING OPPORTUNITIES FOR PASSENGER {passenger_idx}")
        print(f"{'=' * 80}")

        # Get passenger info from final state
        final_state = states[-1]
        origin = final_state.passengers.origins[passenger_idx]
        destination = final_state.passengers.destinations[passenger_idx]

        print(f"Origin: Node {origin}")
        print(f"Destination: Node {destination}")

        # Find when passenger became WAITING
        became_waiting_at = None
        for t, state in enumerate(states):
            if state.passengers.statuses[passenger_idx] == PassengerStatus.WAITING:
                became_waiting_at = t
                break

        if became_waiting_at is None:
            print("Passenger never entered WAITING state!")
            return

        print(f"Passenger became WAITING at time {became_waiting_at}")

        # Find time points where vehicles were at the passenger's origin
        boarding_opportunities = []
        for t in range(became_waiting_at, len(states)):
            state = states[t]

            # Skip if passenger is no longer waiting
            if state.passengers.statuses[passenger_idx] != PassengerStatus.WAITING:
                continue

            # Check if any vehicles are at the passenger's origin
            current_from, _ = get_vehicles_position_and_dest_node(state)
            vehicles_at_origin = jnp.where(
                (current_from == origin) & state.fleet.is_at_node
            )[0]

            if len(vehicles_at_origin) > 0:
                boarding_opportunities.append((t, vehicles_at_origin))

        print(
            f"\nFound {len(boarding_opportunities)} potential boarding opportunities"
        )

        # Analyze each boarding opportunity in detail
        for i, (t, vehicles) in enumerate(
            boarding_opportunities[:5]
        ):  # Limit to first 5
            state = states[t]

            print(f"\nBOARDING OPPORTUNITY {i + 1} at time {t}:")
            print(f"  {len(vehicles)} vehicle(s) at passenger's origin")

            # Calculate travel times
            travel_times_per_vehicle = calculate_travel_times(state)

            # Run through the same logic as in assign_passengers
            for v_idx in vehicles:
                route_id = state.fleet.route_ids[v_idx]
                has_capacity = state.fleet.capacities_left[v_idx] > 0

                print(f"\n  Vehicle {v_idx} (Route {route_id}):")
                print(f"    Has capacity: {has_capacity}")

                if not has_capacity:
                    print("    ❌ Cannot board - vehicle is full")
                    continue

                # Check direct route
                direct_time = travel_times_per_vehicle[v_idx, origin, destination]
                has_direct_route = jnp.isfinite(direct_time)

                if has_direct_route:
                    print(
                        f"    Direct route available - travel time: {direct_time:.1f}"
                    )
                    travel_time = direct_time
                    effective_dest = destination
                else:
                    # Check for transfer option
                    best_time, transfer_node, first_leg, second_leg = (
                        find_best_transfer_route(
                            state, origin, destination, travel_times_per_vehicle
                        )
                    )

                    if transfer_node != -1:
                        print(
                            f"    Transfer route via node {transfer_node} - travel time: {best_time:.1f}"
                        )
                        travel_time = travel_times_per_vehicle[
                            v_idx, origin, transfer_node
                        ]
                        effective_dest = transfer_node
                    else:
                        print("    ❌ No viable route to destination")
                        continue

                # Calculate boarding decision factors
                route_times, _ = calculate_route_times(state)
                shortest_travel_time_overall = jnp.min(route_times, axis=0)
                best_future_time = shortest_travel_time_overall[
                    origin, effective_dest
                ]

                # Apply the boarding threshold
                max_travel_time_ratio = 1.2  # Default from assign_passengers
                should_board = (
                    travel_time <= best_future_time * max_travel_time_ratio
                )

                print(f"    Current travel time: {travel_time:.1f}")
                print(f"    Best future time: {best_future_time:.1f}")
                print(
                    f"    Threshold (best_future * {max_travel_time_ratio}): {best_future_time * max_travel_time_ratio:.1f}"
                )

                if should_board:
                    print(
                        f"    ✅ Should board! (ratio: {travel_time / best_future_time:.2f})"
                    )

                    # Double-check that all boarding conditions are satisfied
                    actual_cond = (
                        state.fleet.is_at_node[v_idx]
                        and (current_from[v_idx] == origin)
                        and (state.fleet.capacities_left[v_idx] > 0)
                        and jnp.isfinite(travel_time)
                        and (
                            travel_time <= best_future_time * max_travel_time_ratio
                        )
                    )

                    print(f"    All conditions met: {actual_cond}")
                    print(f"    is_at_node: {state.fleet.is_at_node[v_idx]}")
                    print(
                        f"    at_correct_location: {current_from[v_idx] == origin}"
                    )
                    print(
                        f"    has_capacity: {state.fleet.capacities_left[v_idx] > 0}"
                    )
                    print(f"    finite_travel_time: {jnp.isfinite(travel_time)}")
                    print(
                        f"    ratio_acceptable: {travel_time <= best_future_time * max_travel_time_ratio}"
                    )

                    # This passenger should have boarded but didn't, likely a bug!
                    print(
                        "\n    🔴 BUG DETECTED: Passenger should have boarded but didn't!"
                    )
                else:
                    print(
                        f"    ❌ Should not board (ratio: {travel_time / best_future_time if jnp.isfinite(best_future_time) else 'inf':.2f})"
                    )


    # Find stranded passengers
    stranded_passengers = identify_stranded_passengers(states[-1])

    # For each stranded passenger, analyze their boarding opportunities
    for passenger_idx in stranded_passengers:
        analyze_boarding_opportunities(states, passenger_idx)
    return (
        analyze_boarding_opportunities,
        identify_stranded_passengers,
        passenger_idx,
        stranded_passengers,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run simulation""")
    return


@app.cell(hide_code=True)
def _(Mandl, PassengerStatus, jax, jnp):
    # Create environment
    n_steps = 200
    buffer_steps = 50

    key = jax.random.PRNGKey(42)
    env = Mandl(
        network_name="mandl1",
        solution_name="yoo2023with8stops",
        runtime=n_steps,
        buffer_time_end=buffer_steps,
        vehicle_capacity=50,
        num_fix_routes=0,
        num_flex_routes=0,
        max_route_length=8,
        total_vehicles=99,
        passenger_init_mode="evenly_spaced",
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

        if n > 200:
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
        buffer_steps,
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


@app.cell(hide_code=True)
def _(plt, state):
    # Create figure for histograms
    plt.figure(figsize=(5.78, 2.5))

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


@app.cell(hide_code=True)
def _(State, np, plt, rgb, svgwrite):
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


    # create_network_svg(state)
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
                    # current_from": np.array(state.fleet.current_edges)[:, 0],
                    # "current_to": np.array(state.fleet.current_edges)[:, 1],
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


@app.cell
def _(states):
    states[-1].routes.stops
    return


@app.cell
def _(states):
    states[-1].fleet
    return


@app.cell
def _(n_steps, states):
    for _i in range(n_steps):
        print(states[_i].fleet.times_on_edge)
    return


@app.cell
def _(n_steps, states):
    for _i in range(n_steps):
        print(states[_i].fleet.current_edges)
    return


@app.cell
def _(n_steps, states):
    for _i in range(n_steps):
        print(states[_i].fleet.directions)
    return


@app.cell
def _(n_steps, states):
    def _():
        from collections import defaultdict

        # Dictionary to track (route, node) combinations
        route_node_stats = defaultdict(lambda: {"count": 0, "capacities": []})

        for _i in range(n_steps):
            _state = states[_i]

            # For each vehicle that's at a node
            for v_idx in range(_state.fleet.num_vehicles):
                if _state.fleet.is_at_node[v_idx]:
                    route_id = _state.fleet.route_ids[v_idx]
                    edge_idx = _state.fleet.current_edges[v_idx]

                    # Get the actual node from the route definition
                    # The vehicle is at the "from" node of the current edge
                    node = _state.routes.stops[route_id, edge_idx]

                    # Create key and record data
                    key = (int(route_id), int(node))
                    route_node_stats[key]["count"] += 1
                    route_node_stats[key]["capacities"].append(
                        int(_state.fleet.capacities_left[v_idx])
                    )

        # Print results grouped by route and node
        print("Route, Node: Total vehicle arrivals, [Capacities left each time]")
        print("--------------------------------------------------------")
        for (route, node), data in sorted(route_node_stats.items()):
            print(
                f"Route {route}, Node {node}: {data['count']} arrivals, capacities: {data['capacities']}"
            )

        # Summary statistics to show bunching
        print("\nBunching Statistics:")
        print("-------------------")
        for (route, node), data in sorted(route_node_stats.items()):
            # Count instances where multiple vehicles arrived simultaneously (same timestep)
            timestep_counts = defaultdict(int)
            for i in range(n_steps):
                vehicles_at_node = 0
                for v_idx in range(states[i].fleet.num_vehicles):
                    if (
                        states[i].fleet.is_at_node[v_idx]
                        and states[i].fleet.route_ids[v_idx] == route
                        and states[i].routes.stops[
                            route, states[i].fleet.current_edges[v_idx]
                        ]
                        == node
                    ):
                        vehicles_at_node += 1
                if vehicles_at_node > 0:
                    timestep_counts[vehicles_at_node] += 1

            # Print bunching statistics
            if len(timestep_counts) > 0:
                max_bunch = max(timestep_counts.keys())
                if max_bunch > 1:  # Only show nodes with bunching
                    print(
                        f"Route {route}, Node {node}: max vehicles at once = {max_bunch}"
                    )
                    for num, count in sorted(timestep_counts.items()):
                        print(f"  {num} vehicle(s) arrived {count} times")
        return print(f"  {num} vehicle(s) arrived {count} times")


    _()
    return


@app.cell
def _(n_steps, states):
    for _i in range(n_steps):
        _state = states[_i]
        _capacity = _state.fleet.capacities_left[_state.fleet.is_at_node]
        print(f"{_i} - capacity left: {_capacity}")
    return


@app.cell
def _(State, jnp, states):
    def analyze_route_edge_indices(state: State) -> None:
        """Analyze edge indices and reversal conditions for each route."""
        print("\nROUTE EDGE INDEX ANALYSIS:")

        for r_idx in range(state.routes.num_routes):
            route = state.routes.stops[r_idx]
            valid_stops = route[route != -1]
            num_valid_stops = len(valid_stops)
            num_edges = num_valid_stops - 1

            print(f"\nRoute {r_idx}:")
            print(f"  Stops: {' -> '.join(str(s) for s in valid_stops)}")
            print(f"  Number of valid stops: {num_valid_stops}")
            print(f"  Number of edges: {num_edges}")

            # Calculate last_valid_edge_idx using current method
            max_num_stops = state.routes.stops.shape[1]
            max_num_edges = max_num_stops - 1
            last_valid_edge_idx = max_num_edges - (route == -1).sum() - 1

            print(f"  Last valid edge index (current): {last_valid_edge_idx}")
            print(f"  Last valid edge index (should be): {num_edges - 1}")

            # Show vehicles on this route
            route_vehicles = jnp.where(state.fleet.route_ids == r_idx)[0]
            print("\n  Vehicles on route:")
            for v_idx in route_vehicles:
                print(f"    Vehicle {v_idx}:")
                print(f"      Current edge: {state.fleet.current_edges[v_idx]}")
                print(
                    f"      Direction: {'Forward' if state.fleet.directions[v_idx] == 0 else 'Backward'}"
                )
                print(f"      Time on edge: {state.fleet.times_on_edge[v_idx]}")

                # Check reversal conditions
                is_at_end = state.fleet.current_edges[v_idx] == last_valid_edge_idx
                is_at_start = state.fleet.current_edges[v_idx] == 0
                print(f"      Is at end: {is_at_end}")
                print(f"      Is at start: {is_at_start}")


    analyze_route_edge_indices(states[70])
    return (analyze_route_edge_indices,)


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
        calculate_shortest_route_times,
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
        get_vehicles_position_and_dest_node,
        calculate_invehicle_times,
        floyd_warshall,
    )
    from jaxtyping import Int, Float, Bool, Array
    from dataclasses import replace

    import scienceplots

    # plt.style.use(["science", "ieee"])
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
        calculate_invehicle_times,
        calculate_shortest_route_times,
        find_best_transfer_route,
        floyd_warshall,
        get_direction_if_connected,
        get_position_in_route,
        get_vehicles_position_and_dest_node,
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
        scienceplots,
        svgwrite,
        tqdm,
        update_passengers_to_waiting,
        update_routes,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
