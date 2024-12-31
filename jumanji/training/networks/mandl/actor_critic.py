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

from typing import Optional, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jumanji.environments.routing.mandl import Mandl, Observation
from jumanji.training.networks.actor_critic import ActorCriticNetworks, FeedForwardNetwork
from jumanji.training.networks.parametric_distribution import MultiCategoricalParametricDistribution
from jumanji.training.networks.transformer_block import TransformerBlock


class MandlTorso(hk.Module):
    def __init__(
        self,
        num_nodes: int,
        transformer_num_blocks: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.transformer_num_blocks = transformer_num_blocks
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.embedding_size = 64  # Fixed size for initial embeddings

    def __call__(self, obs: Observation) -> chex.Array:
        # 1. Network Structure Embedding
        network_features = self._embed_network_structure(
            obs.network, obs.travel_times
        )  # Shape: (batch_size, self.embedding_size)

        # 2. Route Embedding
        route_features = self._embed_routes(
            obs.routes, obs.frequencies, obs.on_demand
        )  # Shape: (batch_size, num_routes, self.embedding_size + 2)

        # 3. Passenger Embedding
        passenger_features = self._embed_passengers(
            obs.origins, obs.destinations, obs.time_waiting, obs.statuses
        )  # Shape: (batch_size, self.embedding_size)

        # Project all features to model_size
        network_projection = hk.Linear(self.model_size)
        route_projection = hk.Linear(self.model_size)
        passenger_projection = hk.Linear(self.model_size)

        network_features = network_projection(network_features)
        # For routes, we need to reshape to apply projection
        b, r, f = route_features.shape  # batch, routes, features
        route_features = route_projection(route_features.reshape(-1, f)).reshape(
            b, r, self.model_size
        )
        passenger_features = passenger_projection(passenger_features)

        # Add sequence dimension to features that don't have it
        network_features = network_features[:, None, :]  # (B, 1, model_size)
        passenger_features = passenger_features[:, None, :]  # (B, 1, model_size)

        # Concatenate along sequence dimension
        combined_features = jnp.concatenate(
            [
                network_features,  # (B, 1, model_size)
                route_features,  # (B, R, model_size)
                passenger_features,  # (B, 1, model_size)
            ],
            axis=1,
        )  # Result: (B, R+2, model_size)

        # Apply transformer blocks
        for _ in range(self.transformer_num_blocks):
            transformer = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2.0 / self.transformer_num_blocks,
                model_size=self.model_size,
            )
            combined_features = transformer(
                query=combined_features, key=combined_features, value=combined_features, mask=None
            )

        return combined_features

    def _embed_network_structure(
        self, network: jnp.ndarray, travel_times: jnp.ndarray
    ) -> jnp.ndarray:
        network_info = jnp.stack(
            [
                network.astype(float),
                jnp.where(travel_times == jnp.inf, 0.0, travel_times),
            ],
            axis=-1,
        )  # Shape: (batch_size, num_nodes, num_nodes, 2)

        network_mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(self.embedding_size),
                jnp.tanh,
                hk.Linear(self.embedding_size),
                jnp.tanh,
            ]
        )
        return network_mlp(network_info)  # Shape: (batch_size, embedding_size)

    def _embed_routes(
        self, routes: jnp.ndarray, frequencies: jnp.ndarray, on_demand: jnp.ndarray
    ) -> jnp.ndarray:
        # Convert routes to embeddings
        route_mlp = hk.Sequential(
            [
                hk.Linear(self.embedding_size),
                jnp.tanh,
                hk.Linear(self.embedding_size),
                jnp.tanh,
            ]
        )
        route_features = route_mlp(routes.astype(float))  # (batch_size, num_routes, embedding_size)

        # Add frequencies and on_demand as additional features
        route_info = jnp.concatenate(
            [
                route_features,  # (B, R, embedding_size)
                frequencies[..., None],  # (B, R, 1)
                on_demand[..., None].astype(float),  # (B, R, 1)
            ],
            axis=-1,
        )  # Result: (B, R, embedding_size + 2)

        return route_info

    def _embed_passengers(
        self,
        origins: jnp.ndarray,
        destinations: jnp.ndarray,
        time_waiting: jnp.ndarray,
        statuses: jnp.ndarray,
    ) -> jnp.ndarray:
        passenger_mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(self.embedding_size),
                jnp.tanh,
                hk.Linear(self.embedding_size),
                jnp.tanh,
            ]
        )

        # Create OD matrices
        batch_size = origins.shape[0]
        waiting_matrix = jnp.zeros((batch_size, self.num_nodes, self.num_nodes))
        in_vehicle_matrix = jnp.zeros((batch_size, self.num_nodes, self.num_nodes))

        # Process matrices through MLP
        passenger_features = passenger_mlp(jnp.stack([waiting_matrix, in_vehicle_matrix], axis=-1))
        return passenger_features  # Shape: (batch_size, embedding_size)


def make_actor_critic_networks_mandl(
    mandl: Mandl,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create actor-critic networks for Mandl environment."""
    num_values = jnp.full(
        shape=(mandl.num_flex_routes,),  # One per route
        fill_value=mandl.num_nodes + 1,  # Each route has same number of choices
        dtype=jnp.int32,
    )
    parametric_action_distribution = MultiCategoricalParametricDistribution(num_values=num_values)

    def actor_fn(obs: Observation) -> chex.Array:
        torso = MandlTorso(
            num_nodes=mandl.num_nodes,
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings = torso(obs)  # Shape: (B, R+2, model_size)

        # Generate logits for each route
        all_logits = hk.Linear(mandl.num_nodes + 1)(embeddings)  # Shape: (B, R+2, num_nodes+1)

        # Extract logits for all flexible routes
        flex_route_start = 1 + jnp.sum(~obs.on_demand)  # 1 for network embedding
        flex_routes_logits = jax.lax.dynamic_slice(
            all_logits,
            start_indices=(0, flex_route_start, 0),
            slice_sizes=(all_logits.shape[0], mandl.num_flex_routes, all_logits.shape[2]),
        )  # Shape: (B, num_flex_routes, num_nodes+1)

        # Scale logits (optional)
        scaled_logits = 10 * jnp.tanh(flex_routes_logits)  # Keep batch dimension!

        return scaled_logits  # Shape: (B, num_flex_routes, num_nodes+1)

    def critic_fn(obs: Observation) -> chex.Array:
        torso = MandlTorso(
            num_nodes=mandl.num_nodes,
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        embeddings = torso(obs)
        pooled_embeddings = jnp.mean(embeddings, axis=1)  # Average over sequence dimension
        value = hk.Linear(1)(pooled_embeddings).squeeze(-1)
        return value

    policy_network = FeedForwardNetwork(*hk.without_apply_rng(hk.transform(actor_fn)))
    value_network = FeedForwardNetwork(*hk.without_apply_rng(hk.transform(critic_fn)))

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
