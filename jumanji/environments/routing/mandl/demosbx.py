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
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Literal

import gymnasium as gym
import submitit
import torch as th
import torch.nn as nn
import tyro
from rich.traceback import install
from sb3_contrib import MaskablePPO

# from sbx import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from torch.utils.checkpoint import Optional

from jumanji.environments.routing.mandl import Mandl
from jumanji.wrappers import JumanjiToGymWrapper

install()


class MandlFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Constants for handling inf values
        self.inf_replacement = 2000.0
        self.max_finite_value = 1000.0

        # Get dimensions from observation space
        self.num_routes = observation_space.spaces["action_mask"].shape[0]
        self.num_nodes = observation_space.spaces["is_terminal"].shape[0]
        self.num_vehicles = observation_space.spaces["fleet_positions"].shape[0]

        # Calculate expected input sizes
        action_mask_size = self.num_routes * (self.num_nodes + 1)  # Include no-op action
        travel_times_size = self.num_nodes * self.num_nodes
        route_stops_size = self.num_routes * observation_space.spaces["route_stops"].shape[1]
        fleet_positions_size = self.num_vehicles * 2

        # Define extractors with correct input sizes
        self.extractors = nn.ModuleDict(
            {
                "action_mask": nn.Sequential(
                    nn.Flatten(), nn.Linear(action_mask_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "travel_times": nn.Sequential(
                    nn.Linear(travel_times_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "route_stops": nn.Sequential(
                    nn.Flatten(), nn.Linear(route_stops_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "fleet_positions": nn.Sequential(
                    nn.Flatten(), nn.Linear(fleet_positions_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "route_types": nn.Sequential(
                    nn.Linear(self.num_routes, 32), nn.LayerNorm(32), nn.ReLU()
                ),
                "route_frequencies": nn.Sequential(
                    nn.Linear(self.num_routes, 32), nn.LayerNorm(32), nn.ReLU()
                ),
                "is_terminal": nn.Sequential(
                    nn.Linear(self.num_nodes, 32), nn.LayerNorm(32), nn.ReLU()
                ),
            }
        )

        # Calculate total feature size
        total_features = (64 * 4) + (32 * 3)  # 4 large (64) + 3 small (32) feature extractors

        # Scalar features remain the same
        self.scalar_features = [
            "current_time",
            "max_route_length",
            "num_fix_routes",
            "num_flex_routes",
            "num_nodes",
            "num_routes",
            "num_vehicles",
        ]
        self.scalar_extractor = nn.Sequential(
            nn.Linear(len(self.scalar_features), 32), nn.LayerNorm(32), nn.ReLU()
        )
        total_features += 32

        # Final combination layers
        self.combination_layer = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensors = []

        # Handle scalar features first
        scalar_features = th.stack(
            [observations[key].squeeze(-1).float() for key in self.scalar_features], dim=1
        )
        encoded_tensors.append(self.scalar_extractor(scalar_features))

        # Process other features
        for key, extractor in self.extractors.items():
            if key in observations:
                x = observations[key].float()

                # Special handling for infinite values
                if key == "travel_times":
                    x = th.where(th.isinf(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.clamp(x, 0.0, self.inf_replacement)
                    x = x / self.inf_replacement
                elif key in ["future_demand", "waiting_demand", "transferring_demand"]:
                    x = th.clamp(x, 0.0, self.max_finite_value)
                    x = x / (x.max() + 1e-8)

                encoded = extractor(x)
                encoded_tensors.append(encoded)

        # Combine all features
        combined = th.cat(encoded_tensors, dim=1)
        return self.combination_layer(combined)


def make_env(
    rank: int,
    network_name: Literal["mandl1", "ceder1"],
    solution_name: Optional[Literal["mandl1", "ceder1"]],
    runtime: float,
    buffer_time: float,
    num_flex_routes: int,
    num_fix_routes: int,
    max_route_length: int,
    num_vehicles_per_fixed_route: int,
    vehicle_capacity: int,
    passenger_init_mode: Literal["evenly_spaced", "rush_hour", "uniform_random", "all_at_start"],
) -> Callable[[], gym.Env]:
    """
    Creates a function that creates an environment.
    This is needed for SubprocVecEnv to properly handle environment creation in separate processes.
    """

    def _init() -> gym.Env:
        env = Mandl(
            network_name=network_name,
            solution_name=solution_name,
            runtime=runtime,
            buffer_time=buffer_time,
            num_fix_routes=num_fix_routes,
            num_flex_routes=num_flex_routes,
            max_route_length=max_route_length,
            num_vehicles_per_fixed_route=num_vehicles_per_fixed_route,
            vehicle_capacity=vehicle_capacity,
            passenger_init_mode=passenger_init_mode,
        )
        env = JumanjiToGymWrapper(env)
        env.render_mode = "rgb_array"
        return env

    return _init


@dataclass
class TrainingConfig:
    """Configuration for training a PPO agent on the Mandl environment."""

    # Environment configuration
    network_name: Literal["ceder1", "mandl1"] = "ceder1"
    solution_name: Optional[Literal["ceder1", "mandl1"]] = None
    runtime: float = 150
    buffer_time: float = 10
    num_flex_routes: int = 16
    num_fix_routes: int = 0
    max_route_length: int = 8
    num_vehicles_per_fixed_route: int = 4
    vehicle_capacity: int = 50
    passenger_init_mode: Literal["evenly_spaced", "rush_hour", "uniform_random", "all_at_start"] = (
        "evenly_spaced"
    )

    # Training configuration
    total_timesteps: int = int(1e6)
    learning_rate: float = 3e-4
    n_steps: int = 150  # * num_envs
    batch_size: int = 150

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

    def __call__(self) -> str:
        return self.train()

    def train(self) -> str:
        """Train the agent and return the path to the saved model."""
        # Create parallel environments
        vec_env = SubprocVecEnv(
            [
                make_env(
                    i,
                    network_name=self.config.network_name,
                    solution_name=self.config.solution_name,
                    runtime=self.config.runtime,
                    buffer_time=self.config.buffer_time,
                    num_fix_routes=self.config.num_fix_routes,
                    num_flex_routes=self.config.num_flex_routes,
                    max_route_length=self.config.max_route_length,
                    num_vehicles_per_fixed_route=self.config.num_vehicles_per_fixed_route,
                    vehicle_capacity=self.config.vehicle_capacity,
                    passenger_init_mode=self.config.passenger_init_mode,
                )
                for i in range(self.config.num_envs)
            ]
        )

        vec_env = VecMonitor(vec_env)
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,  # normalize observations
            norm_reward=True,  # normalize rewards
            # clip_obs=10.,  # clip observations to this value
            # clip_reward=10.,  # clip rewards to this value
            # gamma=0.99,  # discount factor
            epsilon=1e-8,  # small constant to avoid division by zero
        )

        # Set up network architecture
        net_arch = {
            "pi": [self.config.hidden_size] * self.config.n_layers,
            "vf": [self.config.hidden_size] * self.config.n_layers,
        }

        # Create tensorboard log directory
        tensorboard_log = os.path.join(self.config.output_dir, self.config.network_name)

        try:
            # Create and train model
            model = MaskablePPO(
                policy="MultiInputPolicy",
                env=vec_env,
                verbose=1,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                tensorboard_log=tensorboard_log,
                policy_kwargs={
                    "net_arch": net_arch,
                    "features_extractor_class": MandlFeaturesExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "normalize_images": False,
                },
                device=self.config.device,
            )

            # Train the model
            model.learn(total_timesteps=self.config.total_timesteps, progress_bar=True)

            # Save the trained model
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.zip")
            model.save(model_path)
            return model_path

        except Exception as e:
            print(f"\nError during training: {e}")
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
    else:
        # Run directly
        trainer = Trainer(config)
        model_path = trainer.train()
        print(f"Training completed. Model saved to: {model_path}")


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    main(config)
