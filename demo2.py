from typing import List, NamedTuple

import jax.numpy as jnp


class Network(NamedTuple):
    n_stops: int
    weights: jnp.ndarray


class Routes(NamedTuple):
    n_routes: int
    route_edges: jnp.ndarray


class DirectPath(NamedTuple):
    route: int
    cost: float
    start: int
    end: int


class TransferPath(NamedTuple):
    first_route: int
    second_route: int
    total_cost: float
    transfer_stop: int
    start: int
    end: int


def get_route_shortest_paths(network: Network, routes: Routes, route_idx: int):
    """Get shortest paths for a route by masking the network"""
    route = routes.route_edges[route_idx]
    dist = jnp.where(route == 1, network.weights, jnp.inf)

    # Simple Floyd-Warshall with JAX immutable arrays
    n_stops = len(dist)
    for intermediate in range(n_stops):
        for start in range(n_stops):
            for end in range(n_stops):
                dist_via_intermediate = (
                    dist[start, intermediate] + dist[intermediate, end]
                )
                dist = dist.at[start, end].set(
                    jnp.minimum(dist[start, end], dist_via_intermediate)
                )
    return dist


def find_direct_paths(
    route_paths: List[jnp.ndarray], start: int, end: int
) -> List[DirectPath]:
    direct_paths = []
    for r in range(len(route_paths)):
        cost = route_paths[r][start, end]
        if cost < jnp.inf:
            direct_paths.append(
                DirectPath(route=r, cost=float(cost), start=start, end=end)
            )
    return sorted(direct_paths, key=lambda x: x.cost)


def find_transfer_paths(
    route_paths: List[jnp.ndarray], start: int, end: int, transfer_penalty: float = 2.0
) -> List[TransferPath]:
    transfer_paths = []
    n_routes = len(route_paths)
    n_stops = len(route_paths[0])

    for r1 in range(n_routes):
        for r2 in range(n_routes):
            if r1 != r2:
                for transfer_stop in range(n_stops):
                    cost1 = route_paths[r1][start, transfer_stop]
                    cost2 = route_paths[r2][transfer_stop, end]

                    if cost1 < jnp.inf and cost2 < jnp.inf:
                        total_cost = cost1 + cost2 + transfer_penalty
                        transfer_paths.append(
                            TransferPath(
                                first_route=r1,
                                second_route=r2,
                                total_cost=float(total_cost),
                                transfer_stop=transfer_stop,
                                start=start,
                                end=end,
                            )
                        )
    return sorted(transfer_paths, key=lambda x: x.total_cost)


def find_paths(
    network: Network,
    routes: Routes,
    start: int,
    end: int,
    transfer_penalty: float = 2.0,
):
    """Find all valid paths between start and end, including transfers"""
    # Calculate shortest paths for each route
    route_paths = []
    for r in range(routes.n_routes):
        route_paths.append(get_route_shortest_paths(network, routes, r))

    # Always check direct paths first
    direct_paths = find_direct_paths(route_paths, start, end)

    # Only look for transfer paths if no direct paths exist
    transfer_paths = (
        []
        if direct_paths
        else find_transfer_paths(route_paths, start, end, transfer_penalty)
    )

    return direct_paths, transfer_paths


def print_paths(direct_paths: List[DirectPath], transfer_paths: List[TransferPath]):
    if not direct_paths and not transfer_paths:
        print("No valid paths found!")
        return

    if direct_paths:
        print("\nDirect paths:")
        for path in direct_paths:
            print(
                f"{path.start} -> {path.end}, via route {path.route}, cost: {path.cost:1f}"
            )

    if transfer_paths:
        print("\nTransfer paths:")
        for path in transfer_paths:
            print(
                f"{path.start} -> {path.end}, via route {path.first_route}, transfer at {path.transfer_stop} to route {path.second_route}, cost: {path.total_cost:1f}"
            )


def create_example_network():
    n_stops = 7  # A=0, B=1, C=2, D=3, E=4, F=5, G=6
    n_routes = 4

    # Initialize weights matrix with infinity
    weights = jnp.full((n_stops, n_stops), jnp.inf)
    weights = weights.at[jnp.diag_indices(n_stops)].set(0)

    # Define edges and their weights
    edges = [
        (0, 1, 10),  # A -> B
        (1, 2, 15),  # B -> C
        (2, 3, 20),  # C -> D
        (3, 4, 12),  # D -> E
        (2, 5, 10),  # C -> F
        (5, 4, 15),  # F -> E
        (0, 6, 8),  # A -> G
        (6, 4, 9),  # G -> E
        (0, 4, 50),  # A -> E (long direct path)
    ]

    # Make edges bidirectional
    for start, end, weight in edges:
        weights = weights.at[start, end].set(weight)
        weights = weights.at[end, start].set(weight)

    route_edges = jnp.zeros((n_routes, n_stops, n_stops))

    # Route 0: A -> B -> C
    route_edges = route_edges.at[0, 0, 1].set(1)
    route_edges = route_edges.at[0, 1, 0].set(1)
    route_edges = route_edges.at[0, 1, 2].set(1)
    route_edges = route_edges.at[0, 2, 1].set(1)

    # Route 1: C -> D -> E
    route_edges = route_edges.at[1, 2, 3].set(1)
    route_edges = route_edges.at[1, 3, 2].set(1)
    route_edges = route_edges.at[1, 3, 4].set(1)
    route_edges = route_edges.at[1, 4, 3].set(1)

    # Route 2: C -> F -> E
    route_edges = route_edges.at[2, 2, 5].set(1)
    route_edges = route_edges.at[2, 5, 2].set(1)
    route_edges = route_edges.at[2, 5, 4].set(1)
    route_edges = route_edges.at[2, 4, 5].set(1)

    # Route 3: A -> G -> E (direct route)
    route_edges = route_edges.at[3, 0, 6].set(1)
    route_edges = route_edges.at[3, 6, 0].set(1)
    route_edges = route_edges.at[3, 6, 4].set(1)
    route_edges = route_edges.at[3, 4, 6].set(1)

    return Network(n_stops=n_stops, weights=weights), Routes(
        n_routes=n_routes, route_edges=route_edges
    )


if __name__ == "__main__":
    network, routes = create_example_network()

    # Test different scenarios
    test_cases = [
        (0, 4),  # A to E (has both direct and transfer options)
        (0, 2),  # A to C (direct route only)
        (2, 4),  # C to E (direct route only)
        (0, 5),  # A to F (transfer required)
        (6, 2),  # G to C (transfer required)
    ]

    for start, end in test_cases:
        direct_paths, transfer_paths = find_paths(network, routes, start, end)
        print_paths(direct_paths, transfer_paths)
