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

import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Tuple

import gymnasium as gym
import numpy as np
import submitit
import tyro
from gymnasium import spaces
from numpy.typing import NDArray
from sbx import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from jumanji import Environment
from jumanji.environments.routing.mandl import Mandl
from jumanji.wrappers import JumanjiToGymWrapper


class FlattenDictWrapper(gym.Wrapper):
    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self._eps: float = 1e-8

        # Initialize observation space immediately
        dummy_obs, _ = env.reset()  # type: ignore
        flattened = self._flatten_obs(dummy_obs)
        self.observation_space = spaces.Box(low=0, high=1, shape=flattened.shape, dtype=np.float32)

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


def make_env(
    rank: int, network_name: Literal["mandl1", "ceder1"], num_flex_routes: int
) -> Callable[[], gym.Env]:
    """
    Creates a function that creates an environment.
    This is needed for SubprocVecEnv to properly handle environment creation in separate processes.
    """

    def _init() -> gym.Env:
        env = Mandl(network_name=network_name, num_flex_routes=num_flex_routes)
        env = JumanjiToGymWrapper(env)
        env.render_mode = "rgb_array"
        env = FlattenDictWrapper(env)
        return env

    return _init


@dataclass
class TrainingConfig:
    """Configuration for training a PPO agent on the Mandl environment."""

    # Environment configuration
    network_name: Literal["ceder1", "mandl1"] = "ceder1"
    num_flex_routes: int = 16

    # Training configuration
    total_timesteps: int = int(1e6)
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048 * 4  # Will be adjusted by num_envs

    # Model configuration
    policy: str = "MlpPolicy"
    hidden_size: int = 256
    n_layers: int = 2
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Environment parallelism
    num_envs: int = -1  # If -1, will use CPU count

    # Output configuration
    output_dir: str = "outputs"
    model_name: str = "ppo_mandl"

    # Submitit configuration (for SLURM)
    use_slurm: bool = False
    slurm_partition: Literal[
        "dev_single",
        "single",
        "dev_multiple",
        "multiple",
        "fat",
        "dev_gpu_4",
        "gpu_4",
        "gpu_8",
        "dev_multiple_i",
        "multiple_il",
        "dev_gpu_4_a100",
        "gpu_4_a100",
        "gpu_4_h100",
    ] = "single"
    slurm_job_name: str = "mandl_ppo"
    slurm_comment: str = "PPO training on Mandl environment"
    slurm_gpus_per_node: int = 0
    slurm_cpus_per_task: int = 80
    slurm_time: int = 60 * 12  # minutes


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set up number of environments
        if self.config.num_envs == -1:
            self.config.num_envs = os.cpu_count() or 4

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Adjust batch size for parallel environments
        self.config.n_steps = self.config.n_steps // self.config.num_envs

    def __call__(self) -> str:
        return self.train()

    def train(self) -> str:
        """Train the agent and return the path to the saved model."""
        # Create and test a single environment first
        test_env = make_env(0, self.config.network_name, self.config.num_flex_routes)()

        # Print action space information
        print("\nAction Space Information:")
        print(f"Action Space: {test_env.action_space}")
        print(f"Action Space Shape: {test_env.action_space.shape}")
        print(f"Action Space Sample: {test_env.action_space.sample()}")
        print("\nObservation Space Information:")
        print(f"Observation Space: {test_env.observation_space}")
        print(f"Observation Space Shape: {test_env.observation_space.shape}")

        # Test reset and action
        obs, _ = test_env.reset()
        print("\nObservation shape:", obs.shape)
        test_env.close()

        # Create multiple environments in parallel
        vec_env = SubprocVecEnv(
            [
                make_env(i, self.config.network_name, self.config.num_flex_routes)
                for i in range(self.config.num_envs)
            ]
        )
        vec_env = VecMonitor(vec_env)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        print(f"\nCreated {self.config.num_envs} parallel environments")

        # Set up network architecture
        net_arch = {
            "pi": [self.config.hidden_size] * self.config.n_layers,
            "vf": [self.config.hidden_size] * self.config.n_layers,
        }

        # Create tensorboard log directory
        tensorboard_log = os.path.join(self.config.output_dir, test_env.unwrapped.network_name)

        # Create and train model
        model = PPO(
            self.config.policy,
            vec_env,
            verbose=1,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            tensorboard_log=tensorboard_log,
            policy_kwargs={
                "net_arch": net_arch,
                "normalize_images": False,
            },
            device=self.config.device,
        )

        try:
            print(f"\n=== Starting training with {self.config.num_envs} parallel environments ===")
            model.learn(
                total_timesteps=self.config.total_timesteps, progress_bar=True, log_interval=1
            )

            # Save the model
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.zip")
            model.save(model_path)
            print(f"Model saved to {model_path}")

            return model_path

        except Exception as e:
            print(f"\nError during training: {e}")
            print("\nStack trace:")
            import traceback

            traceback.print_exc()
            return "Training failed"
        finally:
            vec_env.close()


def main(config: TrainingConfig) -> None:
    """Main function to handle either direct execution or SLURM submission."""
    # Required for multiprocessing on Windows and macOS
    multiprocessing.set_start_method("spawn")

    if config.use_slurm:
        # Submit the job to SLURM
        executor = submitit.AutoExecutor(folder=os.path.join(config.output_dir, "slurm_logs"))
        executor.update_parameters(
            slurm_partition=config.slurm_partition,
            name=config.slurm_job_name,
            slurm_comment=config.slurm_comment,
            gpus_per_node=config.slurm_gpus_per_node,
            cpus_per_task=config.slurm_cpus_per_task,
            slurm_time=config.slurm_time,
            slurm_mem="160G",
        )

        # Adjust num_envs based on SLURM allocated CPUs
        if config.num_envs is None:
            config.num_envs = config.slurm_cpus_per_task

        trainer = Trainer(config)
        job = executor.submit(trainer)

        print(f"Submitted job {job.job_id}")
        print(f"To check status: squeue -j {job.job_id}")
        print("To cancel: scancel", job.job_id)

        # Optional: wait for completion
        # model_path = job.result()
        # print(f"Training completed. Model saved to: {model_path}")
    else:
        # Run directly
        trainer = Trainer(config)
        model_path = trainer.train()
        print(f"Training completed. Model saved to: {model_path}")


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    main(config)
