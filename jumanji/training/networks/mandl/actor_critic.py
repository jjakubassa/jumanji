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
from jumanji.environments.routing.mandl.types import PassengerStatus, RouteType
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
            obs.network.node_coordinates, obs.network.travel_times
        )

        # 2. Route Embedding
        route_features = self._embed_routes(
            obs.routes.stops, obs.routes.frequencies, obs.routes.types == RouteType.FLEXIBLE
        )

        # 3. Passenger Embedding
        passenger_features = self._embed_passengers(
            obs.origins, obs.destinations, obs.passenger_statuses
        )

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

        # normalize input features
        network_features = hk.LayerNorm(axis=-1)(network_features)
        route_features = hk.LayerNorm(axis=-1)(route_features)
        passenger_features = hk.LayerNorm(axis=-1)(passenger_features)

        # Concatenate along sequence dimension
        combined_features = jnp.concatenate(
            [
                network_features,  # (B, 1, model_size)
                route_features,  # (B, R, model_size)
                passenger_features,  # (B, 1, model_size)
            ],
            axis=1,
        )  # Result: (B, R+2, model_size)

        # Create attention mask
        batch_size = combined_features.shape[0]
        seq_len = combined_features.shape[1]

        # Create base mask (batch_size, 1, seq_len, seq_len) for multi-head attention
        attention_mask = jnp.ones((batch_size, 1, seq_len, seq_len))

        # Prevent routes from attending to other routes
        route_start = 1  # After network features
        route_end = route_start + obs.routes.stops.shape[1]
        attention_mask = attention_mask.at[:, 0, route_start:route_end, route_start:route_end].set(
            0
        )

        # Allow all to attend to network and passenger features
        attention_mask = attention_mask.at[:, 0, :, [0, -1]].set(1)

        # Broadcast mask for all heads
        attention_mask = jnp.broadcast_to(
            attention_mask, (batch_size, self.transformer_num_heads, seq_len, seq_len)
        )

        # Apply transformer blocks with mask
        for _ in range(self.transformer_num_blocks):
            transformer = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2.0 / self.transformer_num_blocks,
                model_size=self.model_size,
            )

            combined_features = transformer(
                query=combined_features,
                key=combined_features,
                value=combined_features,
                mask=attention_mask,
            )

        combined_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            combined_features
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
        """Embed routes with position information and route characteristics.

        Args:
            routes: Route node sequences, shape (batch_size, num_routes, max_route_length)
            frequencies: Route frequencies, shape (batch_size, num_routes)
            on_demand: Boolean indicating if route is flexible, shape (batch_size, num_routes)

        Returns:
            Route embeddings with shape (batch_size, num_routes, embedding_size + 2)
        """
        # 1. Create route position embeddings
        num_routes = routes.shape[1]
        route_positions = jnp.arange(num_routes)[None, :, None]  # Shape: (1, num_routes, 1)
        route_position_embedding = hk.Linear(self.embedding_size)(
            route_positions.astype(float)
        )  # Shape: (1, num_routes, embedding_size)

        # 2. Create route sequence embeddings
        route_mlp = hk.Sequential(
            [
                hk.Linear(self.embedding_size),
                jnp.tanh,
                hk.Linear(self.embedding_size),
                jnp.tanh,
            ]
        )
        route_sequence_features = route_mlp(
            routes.astype(float)
        )  # (batch_size, num_routes, embedding_size)

        # 3. Create route type embeddings for fixed vs flexible routes
        route_type_embedding = hk.Linear(self.embedding_size)(
            on_demand[..., None].astype(float)
        )  # Shape: (batch_size, num_routes, embedding_size)

        # 4. Combine all embeddings
        combined_route_embeddings = (
            route_sequence_features  # Base sequence features
            + route_position_embedding  # Position information
            + route_type_embedding  # Route type (fixed/flexible)
        )

        # 5. Add frequencies and on_demand as additional features
        route_info = jnp.concatenate(
            [
                combined_route_embeddings,  # (batch_size, num_routes, embedding_size)
                frequencies[..., None],  # (batch_size, num_routes, 1)
                on_demand[..., None].astype(float),  # (batch_size, num_routes, 1)
            ],
            axis=-1,
        )  # Result: (batch_size, num_routes, embedding_size + 2)

        # 6. Final route-specific MLP
        final_mlp = hk.Sequential(
            [
                hk.Linear(self.embedding_size + 2),
                jnp.tanh,
            ]
        )

        # Reshape to apply final MLP
        b, r, f = route_info.shape
        route_info = final_mlp(route_info.reshape(-1, f)).reshape(b, r, -1)

        return route_info

    def _embed_passengers(
        self,
        origins: jnp.ndarray,
        destinations: jnp.ndarray,
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

        # Create OD matrices based on actual passenger data - vectorized version
        batch_size = origins.shape[0]
        waiting_mask = statuses == PassengerStatus.WAITING

        # Create indices for updating
        batch_idx = jnp.arange(batch_size)[:, None]
        origins_idx = origins[waiting_mask]
        destinations_idx = destinations[waiting_mask]

        # Create OD matrix using scatter_add
        od_matrix = jnp.zeros((batch_size, self.num_nodes, self.num_nodes))
        od_matrix = od_matrix.at[batch_idx, origins_idx, destinations_idx].add(1)

        passenger_features = passenger_mlp(od_matrix)
        return passenger_features


def make_actor_critic_networks_mandl(
    mandl: Mandl,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create actor-critic networks for Mandl environment."""
    num_values = jnp.full(
        shape=(mandl._route_batch.num_flex_routes,),
        fill_value=mandl._network_data.num_nodes + 1,
        dtype=jnp.int32,
    )
    parametric_action_distribution = MultiCategoricalParametricDistribution(num_values=num_values)

    def actor_fn(obs: Observation) -> chex.Array:
        torso = MandlTorso(
            num_nodes=mandl._network_data.num_nodes,
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings = torso(obs)  # Shape: (B, R+2, model_size)

        # Generate logits for each route
        all_logits = hk.Linear(mandl._network_data.num_nodes + 1)(
            embeddings
        )  # Shape: (B, R+2, num_nodes+1)

        # Extract logits for all flexible routes
        flex_route_start = 1 + jnp.sum(
            obs.routes.types != RouteType.FLEXIBLE
        )  # 1 for network embedding
        flex_routes_logits = jax.lax.dynamic_slice(
            all_logits,
            start_indices=(0, flex_route_start, 0),
            slice_sizes=(all_logits.shape[0], mandl.num_flex_routes, all_logits.shape[2]),
        )  # Shape: (B, num_flex_routes, num_nodes+1)

        # Apply action mask before scaling
        masked_logits = jnp.where(
            obs.action_mask,  # Shape: (B, num_flex_routes, num_nodes+1)
            flex_routes_logits,
            -1e9,  # Large negative value for invalid actions
        )

        # Scale masked logits
        scaled_logits = 10 * jnp.tanh(masked_logits)

        return scaled_logits

    def critic_fn(obs: Observation) -> chex.Array:
        torso = MandlTorso(
            num_nodes=mandl._network_data.num_nodes,
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
