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

from typing import Optional

import matplotlib.animation
import numpy as np
from beartype.typing import Sequence

from jumanji.environments.routing.mandl.types import State
from jumanji.viewer import Viewer


class MandlViewer(Viewer):
    def __init__(self, name: str, render_mode: str = "human") -> None:
        raise NotImplementedError

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        raise NotImplementedError

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
