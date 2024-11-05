import jax.numpy as jnp

# # fmt: off
# demand=jnp.array(
#     [
#         #   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
#         [    0, 400, 200,  60,  80, 150,  75,  75,  30, 160,  30,  25,  35,   0,   0], # 1
#         [  400,   0,  50, 120,  20, 180,  90,  90,  15, 130,  20,  10,  10,   5,   0], # 2
#         [  200,  50,   0,  40,  60, 180,  90,  90,  15,  45,  20,  10,  10,   5,   0], # 3
#         [   60, 120,  40,   0,  50, 100,  50,  50,  15, 240,  40,  25,  10,   5,   0], # 4
#         [   80,  20,  60,  50,   0,  50,  25,  25,  10, 120,  20,  15,   5,   0,   0], # 5
#         [  150, 180, 180, 100,  50,   0, 100, 100,  30, 880,  60,  15,  15,  10,   0], # 6
#         [   75,  90,  90,  50,  25, 100,   0,  50,  15, 440,  35,  10,  10,   5,   0], # 7
#         [   75,  90,  90,  50,  25, 100,  50,   0,  15, 440,  35,  10,  10,   5,   0], # 8
#         [   30,  15,  15,  15,  10,  30,  15,  15,   0, 140,  20,   5,   0,   0,   0], # 9
#         [  160, 130,  45, 240, 120, 880, 440, 440, 140,   0, 600, 250, 500, 200,   0], #10
#         [   30,  20,  20,  40,  20,  60,  35,  35,  20, 600,   0,  75,  95,  15,   0], #11
#         [   25,  10,  10,  25,  15,  15,  10,  10,   5, 250,  75,   0,  70,   0,   0], #12
#         [   35,  10,  10,  10,   5,  15,  10,  10,   0, 500,  95,  70,   0,  45,   0], #13
#         [    0,   5,   5,   5,   0,  10,   5,   5,   0, 200,  15,   0,  45,   0,   0], #14
#         [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], #15
#     ],
#     dtype=jnp.int32,
# )
# # fmt: on

# col_sums = jnp.sum(demand, axis=0)
# row_sums = jnp.sum(demand, axis=1)

# print("\nColumn sums:", col_sums)
# print("Row sums:", row_sums)

# # fmt: off
# network=jnp.array(  # not verified yet
#     [
#         # 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
#         [ 0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 1
#         [ 8,  0,  2,  3,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 2
#         [ 0,  2,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 3
#         [ 0,  3,  0,  0,  4,  4,  0,  0,  0,  0,  0, 10,  0,  0,  0], # 4
#         [ 0,  6,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 5
#         [ 0,  0,  3,  4,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  3], # 6
#         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0,  2], # 7
#         [ 0,  0,  0,  0,  0,  2,  0,  0,  0,  8,  0,  0,  0,  0,  2], # 8
#         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8], # 9
#         [ 0,  0,  0,  0,  0,  0,  7,  8,  0,  0,  5,  0, 10,  8,  0], # 10
#         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0, 10,  5,  0,  0], # 11
#         [ 0,  0,  0, 10,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0], # 12
#         [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 10,  5,  0,  0,  2,  0], # 13
#         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  2,  0,  0], # 14
#         [ 0,  0,  0,  0,  0,  3,  2,  2,  8,  0,  0,  0,  0,  0,  0], # 15
#     ],
#     dtype=jnp.int32,
# )
# # fmt: on

# is_symmetric = jnp.array_equal(network, network.T)
# print("\nNetwork is symmetric:", is_symmetric)


network = jnp.array(
    [
        [0, 1, 0, 0],
        [1, 0, 3, 1],
        [0, 3, 0, 1],
        [0, 1, 1, 0],
    ],
    jnp.int32,
)

print("\nNetwork:\n", network)

demand = jnp.array(
    [
        [0, 2, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 5],
        [0, 0, 5, 0],
    ],
    jnp.int32,
)

print("\nDemand:\n", demand)

routes = jnp.array([[0, 1, 3, 2], [2, 1, -1, -1]], jnp.int32)
# routes = jnp.array([[1, 3, 2, -1], [0, 1, -1, -1]], jnp.int32)
print("\nRoutes:\n", routes)


def get_routes_edges(routes):
    num_edges = routes.shape[1] - 1
    num_routes = routes.shape[0]
    routes_edges = jnp.full((num_routes, num_edges, 2), -1, jnp.int32)

    for route_idx, route in enumerate(routes):
        for edge_idx, (node, next_node) in enumerate(zip(route[:-1], route[1:])):
            if node != -1 and next_node != -1:
                routes_edges = routes_edges.at[route_idx, edge_idx, :].set(
                    jnp.array([node, next_node])
                )

    return routes_edges


routes_edges = get_routes_edges(routes)
print("\nRoutes edges:\n", routes_edges)


def are_valid_routes(routes, network, min_stops=2, max_stops=10, max_travel_time=120):
    # „1) The output network is a connected graph. This means that all stops are reachable from any other stop.
    # 2) Each and every stop in the graph must be present in at least one of the routes.
    # 3) Routes must be ultimately different from each other. Identical routes are considered to be a single route with their frequencies combined.
    # 4) No route can have a cycle.
    # 5) The number of output routes, |L|, is predefined by the operator.
    # 6) All buses in the operator's fleet have the same capacity.
    # 7) The number of stops in each route has to be within a predefined range. In addition, the estimated travel time to traverse a single route has to be bounded by a predefined value. All this is to account for feasibility of bus schedules and to avoid overtiring the bus drivers." (Darwish et al., 2020, p. 3)

    # 2) Each and every stop in the graph must be present in at least one of the routes.
    routes_edges = get_routes_edges(routes)
    node_occured = jnp.zeros(network.shape[0])
    for route in routes_edges:
        for edge in route:
            if edge[0] != -1:
                node_occured = node_occured.at[edge[0]].set(True)
            if edge[1] != -1:
                node_occured = node_occured.at[edge[1]].set(True)

    if not jnp.all(node_occured):
        return False

    # Check other constraints for each route
    for route in routes:
        # 7) Check if route has at least min_stops and at most max_stops
        num_stops = len([x for x in route if x != -1])
        if num_stops < min_stops or num_stops > max_stops:
            return False

        # 4) No route can have a cycle.
        visited = jnp.zeros(len(network), dtype=bool)
        for stop in route:
            if stop == -1:
                continue
            if visited[stop]:
                return False
            visited = visited.at[stop].set(True)

        # Check if all edges in route exist in network and calculate travel time
        travel_time = 0
        for curr_stop, next_stop in zip(route[:-1], route[1:]):
            if curr_stop == -1 or next_stop == -1:
                continue
            if network[curr_stop, next_stop] == 0:
                return False
            travel_time += network[curr_stop, next_stop]

        # 7) Check travel time constraint
        if travel_time > max_travel_time:
            return False

    return True


valid = are_valid_routes(routes, network)
print("\nRoutes are valid:", valid)


def find_routes_with_transfers(routes, origin, destination):
    direct_routes = []
    transfer_routes = []

    for i, route1 in enumerate(routes):
        # Check for direct routes
        if origin in route1 and destination in route1:
            # Get positions to check order
            origin_idx = jnp.where(route1 == origin)[0][0]
            dest_idx = jnp.where(route1 == destination)[0][0]
            if origin_idx < dest_idx:
                direct_routes.append(i)

        # Check for transfer routes
        for j, route2 in enumerate(routes):
            if i == j:
                continue

            # Find transfer points (common stops between routes)
            transfer_points = jnp.sort(
                jnp.array([stop for stop in route1 if stop != -1 and stop in route2])
            )

            for transfer in transfer_points:
                # Check if origin -> transfer -> destination possible
                if origin in route1 and transfer in route1:
                    origin_idx = jnp.where(route1 == origin)[0][0]
                    transfer_idx1 = jnp.where(route1 == transfer)[0][0]

                    if transfer in route2 and destination in route2:
                        transfer_idx2 = jnp.where(route2 == transfer)[0][0]
                        dest_idx = jnp.where(route2 == destination)[0][0]

                        if origin_idx < transfer_idx1 and transfer_idx2 < dest_idx:
                            transfer_routes.append((i, j, int(transfer)))

    return direct_routes, transfer_routes


def assign_passengers(network, demand, routes):
    unmet_demand = jnp.zeros_like(demand)
    zero_transfer_demand = jnp.zeros_like(demand)
    one_transfer_demand = jnp.zeros_like(demand)

    for i in range(demand.shape[0]):
        for j in range(demand.shape[1]):
            if demand[i, j] == 0:
                continue

            print(f"Demand from {i} to {j}: {demand[i, j]}")

    return unmet_demand, zero_transfer_demand, one_transfer_demand


unmet_demand, zero_transfer_demand, one_transfer_demand = assign_passengers(
    network, demand, routes
)
# print("\nUnmet demand:\n", unmet_demand)
# print("\nZero transfer demand:\n", zero_transfer_demand)
# print("\nOne transfer demand:\n", one_transfer_demand)


def modify_graph_with_transfers(adj_matrix, routes, transfer_penalty=5):
    """
    Modifies a public transport graph by splitting nodes and adding transfer penalties.
    Includes all possible routes with one transfer.

    Args:
        adj_matrix: jnp.ndarray of shape (n, n) representing travel times between nodes.
                    adj_matrix[i, j] = travel time between nodes i and j (0 if no edge exists).
        routes: jnp.ndarray of shape (num_lines, max_num_stops), where each row represents
                a route as a sequence of stops (nodes). -1 indicates unused slots.
        transfer_penalty: Penalty for transferring between different lines at the same node.

    Returns:
        A tuple:
        - Modified adjacency matrix as jnp.ndarray.
        - Expanded routes as jnp.ndarray in the same format as the input routes.
    """
    num_nodes = adj_matrix.shape[0]
    num_lines, max_num_stops = routes.shape

    # Each original node splits into two sub-nodes -> double the number of nodes
    new_num_nodes = num_nodes * 2
    new_adj_matrix = jnp.zeros((new_num_nodes, new_num_nodes), jnp.int32)

    # Helper function to map original node to sub-nodes
    def map_to_subnodes(node):
        return node * 2, node * 2 + 1  # Returns (a, b)

    # Add transfer edges (penalty) between sub-nodes
    for node in range(num_nodes):
        node_a, node_b = map_to_subnodes(node)
        new_adj_matrix = new_adj_matrix.at[node_a, node_b].set(transfer_penalty)
        new_adj_matrix = new_adj_matrix.at[node_b, node_a].set(transfer_penalty)

    # Add edges for each route (line edges)
    for line_idx in range(num_lines):
        route = routes[line_idx]
        for i in range(max_num_stops - 1):
            if route[i] == -1 or route[i + 1] == -1:
                break  # Skip if we reach the end of the route
            stop1, stop2 = route[i], route[i + 1]
            stop1_a, stop1_b = map_to_subnodes(stop1)
            stop2_a, stop2_b = map_to_subnodes(stop2)

            # Use `a` nodes for even lines, `b` nodes for odd lines
            if line_idx % 2 == 0:
                new_adj_matrix = new_adj_matrix.at[stop1_a, stop2_a].set(
                    adj_matrix[stop1, stop2]
                )
                new_adj_matrix = new_adj_matrix.at[stop2_a, stop1_a].set(
                    adj_matrix[stop2, stop1]
                )  # Undirected
            else:
                new_adj_matrix = new_adj_matrix.at[stop1_b, stop2_b].set(
                    adj_matrix[stop1, stop2]
                )
                new_adj_matrix = new_adj_matrix.at[stop2_b, stop1_b].set(
                    adj_matrix[stop2, stop1]
                )  # Undirected

    # Add transfer edges (transfer between lines at the same station)
    for i in range(num_nodes):
        neighbors = jnp.where(adj_matrix[i] > 0)[0]
        for neighbor in neighbors:
            i_a, i_b = map_to_subnodes(i)
            neighbor_a, neighbor_b = map_to_subnodes(neighbor)

            # Connect transfers between `a` and `b` sub-nodes for neighbors
            new_adj_matrix = new_adj_matrix.at[i_b, neighbor_a].set(
                adj_matrix[i, neighbor]
            )
            new_adj_matrix = new_adj_matrix.at[neighbor_a, i_b].set(
                adj_matrix[neighbor, i]
            )  # Undirected

    # Prepare expanded routes array
    max_route_length = 2 * max_num_stops + 1
    num_expanded_routes = (
        num_lines * max_num_stops * 3
    )  # Worst-case: direct + two transfers per stop pair
    expanded_routes = -jnp.ones(
        (num_expanded_routes, max_route_length), dtype=jnp.int32
    )
    route_idx = 0

    for line_idx in range(num_lines):
        original_route = []
        for stop in routes[line_idx]:
            if stop == -1:
                break  # Skip unused slots
            stop_a, stop_b = map_to_subnodes(stop)
            # Use `a` nodes for even lines, `b` nodes for odd lines
            if line_idx % 2 == 0:
                original_route.append(stop_a)
            else:
                original_route.append(stop_b)

        # Generate all possible routes with one transfer
        for i in range(len(original_route) - 1):
            current_stop = original_route[i]
            next_stop = original_route[i + 1]

            # Add direct route
            expanded_routes = expanded_routes.at[route_idx, :2].set(
                jnp.array([current_stop, next_stop])
            )
            route_idx += 1

            # Add routes with one transfer
            current_station = current_stop // 2
            next_station = next_stop // 2
            if current_station == next_station:
                continue  # Skip if the transfer doesn't make sense (same station)
            transfer_a, transfer_b = map_to_subnodes(current_station)
            expanded_routes = expanded_routes.at[route_idx, :3].set(
                jnp.array([current_stop, transfer_b, next_stop])
            )
            route_idx += 1
            expanded_routes = expanded_routes.at[route_idx, :3].set(
                jnp.array([current_stop, transfer_a, next_stop])
            )
            route_idx += 1

    # Remove unused rows
    expanded_routes = expanded_routes[:route_idx]

    return new_adj_matrix, expanded_routes


# Modify the graph
modified_adj_matrix, expanded_routes = modify_graph_with_transfers(
    network, routes, transfer_penalty=2
)

# Print the results
print("Modified Adjacency Matrix:")
print(modified_adj_matrix)

print("\nExpanded Routes:")
print(expanded_routes)
