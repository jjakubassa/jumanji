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
from typing import Callable, Dict

import gymnasium as gym
import hydra
import submitit
import torch as th
import torch.nn as nn
from omegaconf import OmegaConf
from rich.traceback import install
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.checkpoint import Optional

import wandb
from jumanji.environments.routing.mandl.config import NetworkName, PassengerMode, TrainingConfig
from wandb.integration.sb3 import WandbCallback

install()


class MandlFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Constants for handling inf values
        self.inf_replacement = 200_000.0
        self.max_finite_value = 100_000.0

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
                "direct_travel_times": nn.Sequential(
                    nn.Linear(travel_times_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "transfer_travel_times": nn.Sequential(
                    nn.Linear(travel_times_size, 64), nn.LayerNorm(64), nn.ReLU()
                ),
                "network_shortest_times": nn.Sequential(
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
        total_features = (64 * 7) + (32 * 3)  # 7 large (64) + 3 small (32) feature extractors

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
                if key in [
                    "travel_times",
                    "network_shortest_times",
                    "direct_travel_times",
                    "transfer_travel_times",
                ]:
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
    network_name: NetworkName,
    solution_name: Optional[NetworkName],
    runtime: float,
    buffer_time_end: float,
    num_flex_routes: int,
    num_fix_routes: int,
    max_route_length: int,
    total_vehicles: int,
    vehicle_capacity: int,
    vehicles_per_additional_fixed_route: Optional[list[int]],
    passenger_init_mode: PassengerMode,
) -> Callable[[], gym.Env]:
    """Creates a function that creates an environment."""

    from jumanji.environments.routing.mandl import Mandl
    from jumanji.wrappers import JumanjiToGymWrapper

    def _init() -> gym.Env:
        env = Mandl(
            network_name=network_name.value,  # Add .value to get string
            solution_name=solution_name.value if solution_name else None,  # Add .value
            runtime=runtime,
            buffer_time_end=buffer_time_end,
            num_fix_routes=num_fix_routes,
            num_flex_routes=num_flex_routes,
            max_route_length=max_route_length,
            total_vehicles=total_vehicles,
            vehicle_capacity=vehicle_capacity,
            vehicles_per_additional_fixed_route=vehicles_per_additional_fixed_route,
            passenger_init_mode=passenger_init_mode.value,  # Add .value
        )
        env = JumanjiToGymWrapper(env)
        env.render_mode = "rgb_array"
        return env

    return _init


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
        print(self.config)

        if self.config.wandb_entity:
            config_dict = {
                k.lower(): v.lower() if isinstance(v, str) else v
                for k, v in OmegaConf.to_container(self.config, resolve=True).items()
            }

            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_name,
                config=config_dict,
                sync_tensorboard=True,
            )

        # Create parallel environments
        vec_env = DummyVecEnv(
            [
                make_env(
                    i,
                    network_name=self.config.network_name,
                    solution_name=self.config.solution_name,
                    runtime=self.config.runtime,
                    buffer_time_end=self.config.buffer_time_end,
                    num_fix_routes=self.config.num_fix_routes,
                    num_flex_routes=self.config.num_flex_routes,
                    max_route_length=self.config.max_route_length,
                    total_vehicles=self.config.total_vehicles,
                    vehicle_capacity=self.config.vehicle_capacity,
                    vehicles_per_additional_fixed_route=self.config.vehicles_per_additional_fixed_route,
                    passenger_init_mode=self.config.passenger_init_mode,
                )
                for i in range(self.config.num_envs)
            ]
        )

        metric_to_track = (
            "completion_rate",
            "total_waiting_time",
            "avg_waiting_time",
            "max_waiting_time",
            "total_in_vehicle_time",
            "avg_in_vehicle_time",
            "max_in_vehicle_time",
            "total_travel_time",
            "avg_total_travel_time",
            "total_transfers",
            "avg_transfers_per_passenger",
            "avg_vehicle_utilization",
            "percent_empty_vehicles",
            "percent_full_vehicles",
            "ratio_travel_time_direct_to_shortest_path",
            "ratio_travel_time_transfers_to_shortest_path",
        )

        vec_env = VecMonitor(vec_env, info_keywords=metric_to_track)
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
                policy=self.config.policy,
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
            progress_bar = not self.config.use_slurm
            wandb_callback = (
                WandbCallback(model_save_path=tensorboard_log, model_save_freq=10_000_000)
                if self.config.wandb_entity
                else None
            )
            model.learn(
                total_timesteps=self.config.total_timesteps,
                progress_bar=progress_bar,
                callback=wandb_callback,
            )

            # Save the trained model
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.zip")
            model.save(model_path)
            return model_path

        except Exception as e:
            print(f"\nError during training: {e}")
            traceback.print_exc()
            return "Training failed"

        finally:
            wandb.finish()
            vec_env.close()


@hydra.main(version_base=None, config_name="ceder_fix")
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

        print(f"Submitted job {job.job_id} - {config.slurm_job_name}")
        print(f"To check status: squeue -j {job.job_id}")
        print("To cancel: scancel", job.job_id)
    else:
        # Run directly
        trainer = Trainer(config)
        model_path = trainer.train()
        print(f"Training completed. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
