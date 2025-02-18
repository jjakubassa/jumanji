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
            obs.origins, obs.destinations, obs.passenger_statuses, obs.desired_departure_times
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
        network_features = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="network_norm"
        )(network_features)

        route_features = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="route_norm"
        )(route_features)

        passenger_features = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="passenger_norm"
        )(passenger_features)

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
        self, node_coordinates: jnp.ndarray, travel_times: jnp.ndarray
    ) -> jnp.ndarray:
        # Create features from travel times matrix
        edge_features = jnp.where(
            travel_times == jnp.inf, 0.0, travel_times
        )  # Shape: (B, num_nodes, num_nodes)

        # Flatten edge features - need to handle batch dimension
        batch_size = edge_features.shape[0]
        flat_edge_features = edge_features.reshape(
            batch_size, -1
        )  # Shape: (B, num_nodes * num_nodes)

        # Calculate mean features and reshape to match dimensions
        mean_outgoing = jnp.mean(edge_features, axis=2)  # Shape: (B, num_nodes)
        mean_incoming = jnp.mean(edge_features, axis=1)  # Shape: (B, num_nodes)

        # Concatenate all features
        network_features = jnp.concatenate(
            [
                flat_edge_features,
                mean_outgoing,
                mean_incoming,
            ],
            axis=1,
        )  # Shape: (B, num_nodes * num_nodes + 2 * num_nodes)

        network_mlp = hk.Sequential(
            [
                hk.Linear(self.embedding_size),
                jnp.tanh,
                hk.Linear(self.embedding_size),
                jnp.tanh,
            ]
        )

        return network_mlp(network_features)  # Shape: (B, embedding_size)

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
        origins: jnp.ndarray,  # Shape: (batch_size, num_passengers)
        destinations: jnp.ndarray,  # Shape: (batch_size, num_passengers)
        statuses: jnp.ndarray,  # Shape: (batch_size, num_passengers)
        desired_departure_times: jnp.ndarray,  # Shape: (batch_size, num_passengers)
    ) -> jnp.ndarray:
        # Create feature vector: for each passenger combine
        # (origin, destination, desired_time, is_waiting)
        is_waiting = (statuses == PassengerStatus.WAITING).astype(jnp.float32)
        passenger_features = jnp.stack(
            [
                origins.astype(jnp.float32),
                destinations.astype(jnp.float32),
                desired_departure_times,
                is_waiting,
            ],
            axis=-1,
        )  # Shape: (batch_size, num_passengers, 4)

        # First MLP to process each passenger independently
        passenger_mlp = hk.Sequential(
            [
                hk.Linear(32),
                jnp.tanh,
                hk.Linear(32),
                jnp.tanh,
            ]
        )

        # Process each passenger independently
        batch_size, num_passengers, feature_dim = passenger_features.shape
        reshaped_features = passenger_features.reshape(-1, feature_dim)
        processed_features = passenger_mlp(reshaped_features)
        processed_features = processed_features.reshape(batch_size, num_passengers, -1)

        # Pool across passengers
        pooled_features = jnp.mean(processed_features, axis=1)  # Shape: (batch_size, 32)

        # Final projection to embedding size
        final_projection = hk.Linear(self.embedding_size)
        return final_projection(pooled_features)  # Shape: (batch_size, embedding_size)


def make_actor_critic_networks_mandl(
    mandl: Mandl,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create actor-critic networks for Mandl environment."""
    # Change this to use total number of routes instead of flex routes
    num_values = jnp.full(
        shape=(mandl._route_batch.num_routes,),  # Use total routes instead of flex routes
        fill_value=mandl._network_data.num_nodes + 1,
        dtype=jnp.int32,
    )
    parametric_action_distribution = MultiCategoricalParametricDistribution(num_values=num_values)

    def actor_fn(obs: Observation) -> chex.Array:
        # Get numbers without batch dimension
        num_nodes = obs.network.travel_times.shape[1]  # Use shape[1] instead of shape[0]
        num_routes = obs.routes.stops.shape[1]  # Use shape[1] instead of shape[0]

        torso = MandlTorso(
            num_nodes=num_nodes,
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings = torso(obs)  # Shape: (B, R+2, model_size)

        # Extract route embeddings (skip network and passenger embeddings)
        route_embeddings = embeddings[:, 1 : 1 + num_routes]  # Shape: (B, R, model_size)

        # Project each route embedding to its action logits
        logits = hk.Linear(num_nodes + 1)(route_embeddings)  # Shape: (B, R, num_nodes+1)

        # Add batch dimension to mask if needed
        action_mask = obs.action_mask
        if action_mask.ndim < logits.ndim:
            action_mask = action_mask[None, ...]

        # Apply mask with broadcasting
        masked_logits = jnp.where(action_mask, logits, jnp.full_like(logits, -1e9))

        # Scale masked logits
        scaled_logits = 10 * jnp.tanh(masked_logits)

        # Reshape to have a leading 1 dimension that can be squeezed
        # and ensure the result will be (num_routes,)
        # scaled_logits = scaled_logits.reshape(1, num_routes, -1)  # Shape: (1, R, num_nodes+1)

        return scaled_logits

    def critic_fn(obs: Observation) -> chex.Array:
        num_nodes = obs.network.is_terminal.shape[0]
        torso = MandlTorso(
            num_nodes=num_nodes,
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
