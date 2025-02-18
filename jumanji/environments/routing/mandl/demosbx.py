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

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import jumanji
from jumanji import Environment
from jumanji.wrappers import JumanjiToGymWrapper


class FlattenDictWrapper(gym.Wrapper):
    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self._observation_space: Optional[spaces.Box] = None
        self._eps: float = 1e-8

    def _preprocess_value(self, value: Any, key: str) -> NDArray[np.float32]:
        """Preprocess values to avoid numerical issues."""
        arr: NDArray = np.array(value, dtype=np.float32)

        # Always flatten first
        arr = arr.flatten()

        if key == "travel_times":
            # For travel times matrix, handle inf values specially
            reachable: NDArray[np.bool_] = ~np.isinf(arr)
            finite_arr: NDArray = arr[reachable]
            if finite_arr.size > 0:
                max_finite: float = finite_arr.max()
                min_finite: float = finite_arr.min()
                arr[reachable] = (
                    0.5 * (finite_arr - min_finite) / (max_finite - min_finite + self._eps)
                )
                arr[~reachable] = 1.0

        elif key == "node_coordinates":
            # Already normalized in [0,1]
            pass

        elif key in ["desired_departure_times"]:
            min_val: float = arr.min()
            max_val: float = arr.max()  # type: ignore
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val + self._eps)
        elif key in ["passenger_statuses"]:
            # Use fixed size one-hot encoding for passenger statuses
            max_status: int = 5  # Number of possible status types
            one_hot: NDArray[np.float32] = np.zeros((arr.size, max_status), dtype=np.float32)
            for i, val in enumerate(arr):
                if val >= 0:  # Handle -1 padding values
                    one_hot[i, int(val)] = 1
            arr = one_hot.flatten()
        elif key in ["types", "destinations", "origins"]:
            # One-hot encode other categorical variables
            max_val: int = int(arr.max())  # type: ignore
            one_hot = np.zeros((arr.size, max_val + 1), dtype=np.float32)
            for i, val in enumerate(arr):
                if val >= 0:  # Handle -1 padding values
                    one_hot[i, int(val)] = 1
            arr = one_hot.flatten()

        return arr.astype(np.float32)

    def _flatten_value(self, value: Any, key: str = "") -> NDArray[np.float32]:
        """Helper function to flatten individual values."""
        if isinstance(value, dict):
            nested_arrays: list[NDArray[np.float32]] = []
            for k, v in sorted(value.items()):  # Sort keys for consistency
                flat_v: NDArray[np.float32] = self._flatten_value(v, k)
                nested_arrays.append(flat_v)
            return np.concatenate(nested_arrays)

        return self._preprocess_value(value, key)

    def _flatten_obs(self, obs: Dict[str, Any]) -> NDArray[np.float32]:
        """Flatten the observation dictionary into a single array."""
        flattened_arrays: list[NDArray[np.float32]] = []

        for key in sorted(obs.keys()):
            if key == "action_mask":
                continue

            value = obs[key]

            try:
                flat_value: NDArray[np.float32] = self._flatten_value(value, key)
                if flat_value.size > 0:
                    flattened_arrays.append(flat_value)
            except Exception as e:
                print(f"Error processing {key}: {e!s}")
                raise

        result: NDArray[np.float32] = np.concatenate(flattened_arrays)
        return result.astype(np.float32)

    def reset(self, **kwargs: Any) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        if self._observation_space is None:
            flattened = self._flatten_obs(obs)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=flattened.shape, dtype=np.float32
            )
        return self._flatten_obs(obs), info

    def step(
        self, action: NDArray[np.int_]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info


# Create environment
env = jumanji.make("Mandl-v0")
env = JumanjiToGymWrapper(env)
env.render_mode = "rgb_array"

# Print action space information
print("\nAction Space Information:")
print(f"Action Space: {env.action_space}")
print(f"Action Space Shape: {env.action_space.shape}")
print(f"Action Space Sample: {env.action_space.sample()}")

# Print observation space details
print("\nOriginal observation space structure:")
for key, space in env.observation_space.spaces.items():
    if isinstance(space, spaces.Dict):
        print(f"{key}:")
        for subkey, subspace in space.spaces.items():
            print(f"  {subkey}: {subspace}")
    else:
        print(f"{key}: {space}")

# Wrap environment
env = FlattenDictWrapper(env)

# Test reset and action
obs, _ = env.reset()
print("\nObservation shape:", obs.shape)

# Wrap for training
vec_env = DummyVecEnv([lambda: env])
vec_env = VecMonitor(vec_env)

# Create and train model with explicit action space handling
model = A2C(
    "MlpPolicy",
    vec_env,
    verbose=1,
    policy_kwargs={
        "net_arch": [64, 64],
        "normalize_images": False,
    },
)

try:
    print("\n=== Starting training ===")
    model.learn(total_timesteps=int(1e6), progress_bar=True, log_interval=100)
except Exception as e:
    print(f"\nError during training: {e}")
    print("\nStack trace:")
    import traceback

    traceback.print_exc()
