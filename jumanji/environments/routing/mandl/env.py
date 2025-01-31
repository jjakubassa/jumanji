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

from functools import cached_property
from typing import Final, Literal, Optional

import chex
import matplotlib
from beartype.typing import Sequence

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mandl.types import (
    Observation,
    State,
)
from jumanji.environments.routing.mandl.viewer import MandlViewer
from jumanji.types import TimeStep
from jumanji.viewer import Viewer


class Mandl(Environment[State, specs.BoundedArray, Observation]):
    def __init__(
        self,
        viewer: Optional[Viewer] = None,
        network_name: Literal["mandl1", "ceder1"] = "ceder1",
    ) -> None:
        self.network_name: Final = network_name
        self._viewer = viewer or MandlViewer(
            name="Mandl",
            render_mode="human",
        )
        super().__init__()

    def __repr__(self) -> str:
        raise NotImplementedError

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to an initial state."""
        raise NotImplementedError

    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep[Observation]]:
        raise NotImplementedError

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            A nested `specs.Spec` matching the structure of `Observation`.
        """
        raise NotImplementedError

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        raise NotImplementedError

    def render(self, state: State) -> Optional[chex.ArrayNumpy]:
        raise NotImplementedError
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        raise NotImplementedError
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Close the environment and clean up resources."""
        raise NotImplementedError
        if self._viewer is not None:
            self._viewer.close()
