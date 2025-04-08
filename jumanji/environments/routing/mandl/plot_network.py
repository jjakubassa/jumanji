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

import argparse

import jax
import matplotlib.pyplot as plt
import numpy as np

from jumanji.environments.routing.mandl import Mandl


def list_networks() -> list[str]:
    """Print all available networks."""
    networks = ["mandl1", "ceder1", "mumford0", "mumford1", "mumford2", "mumford3"]
    print("\nAvailable networks:")
    for net in networks:
        print(f"- {net}")
    return networks


def plot_network(network_name: str, show_labels: bool = True) -> None:
    """Plot a single network."""
    # Create environment with specified network
    env = Mandl(network_name=network_name)  # type: ignore

    # Reset to get initial state
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)

    # Create figure
    fig, ax = plt.subplots(figsize=(5.78, 5.78))

    # Get node coordinates and travel times
    nodes = state.network.node_coordinates
    travel_times = state.network.travel_times

    transformed_nodes = np.column_stack(
        [
            nodes[:, 1],  # y becomes x
            nodes[:, 0],  # x becomes y
        ]
    )

    # Plot network links
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if travel_times[i, j] < float("inf"):
                from_node = transformed_nodes[i]
                to_node = transformed_nodes[j]
                ax.plot(
                    [from_node[0], to_node[0]],
                    [from_node[1], to_node[1]],
                    "gray",
                    alpha=0.5,
                    zorder=1,
                )

    # Plot nodes with different sizes based on whether labels are shown
    node_size = 500 if show_labels else 50
    ax.scatter(
        transformed_nodes[:, 0],
        transformed_nodes[:, 1],
        c="white",
        s=node_size,
        edgecolors="black",
        linewidth=2 if show_labels else 1,
        zorder=3,
    )

    # Add node labels if requested
    if show_labels:
        for i in range(len(nodes)):
            ax.annotate(
                str(i + 1),  # 1-based indexing for display
                (transformed_nodes[i, 0], transformed_nodes[i, 1]),
                ha="center",
                va="center",
                zorder=4,
            )

    # Remove axes and borders
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set plot limits (adjusted for the transformation)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Add title
    # plt.title(f"{network_name} network")

    plt.tight_layout()

    # Save with network name and label info
    label_str = "with_labels" if show_labels else "no_labels"
    save_path = f"network_plot_{network_name}_{label_str}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    networks = list_networks()

    parser = argparse.ArgumentParser(description="Plot transit network structure")
    parser.add_argument(
        "--network",
        type=str,
        choices=[*networks, "all"],
        default="mandl1",
        help='Network to plot (use "all" to plot all networks)',
    )
    parser.add_argument("--hide-labels", action="store_true", help="Hide node labels in the plot")
    parser.add_argument("--list", action="store_true", help="List all available networks and exit")

    args = parser.parse_args()

    if args.list:
        # Already listed networks at start
        exit(0)

    if args.network == "all":
        print("\nGenerating plots for all networks...")
        for network in networks:
            print(f"\nPlotting {network}...")
            plot_network(network, not args.hide_labels)
            print(f"Plotting {network} without labels...")
            plot_network(network, False)
    else:
        plot_network(args.network, not args.hide_labels)
