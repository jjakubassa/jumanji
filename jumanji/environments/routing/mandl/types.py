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

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp


@dataclass
class State:
    """
    key: random key used for auto-reset.
    """

    network: chex.Array
    demand: chex.Array
    routes: chex.Array
    capacity: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """ """

    network: chex.Array
    demand_original: chex.Array
    demand_now: chex.Array
    routes: chex.Array
    capacity_left: chex.Array
    action_mask: chex.Array


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
