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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hydra.core.config_store import ConfigStore


class NetworkName(str, Enum):
    CEDER1 = "ceder1"
    MANDL1 = "mandl1"


class PassengerMode(str, Enum):
    EVENLY_SPACED = "evenly_spaced"
    RUSH_HOUR = "rush_hour"
    UNIFORM_RANDOM = "uniform_random"
    ALL_AT_START = "all_at_start"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class SlurmPartition(str, Enum):
    DEV_SINGLE = "dev_single"
    SINGLE = "single"
    DEV_MULTIPLE = "dev_multiple"
    MULTIPLE = "multiple"
    FAT = "fat"
    DEV_GPU_4 = "dev_gpu_4"
    GPU_4 = "gpu_4"
    GPU_8 = "gpu_8"
    DEV_MULTIPLE_I = "dev_multiple_i"
    MULTIPLE_IL = "multiple_il"
    DEV_GPU_4_A100 = "dev_gpu_4_a100"
    GPU_4_A100 = "gpu_4_a100"
    GPU_4_H100 = "gpu_4_h100"


@dataclass
class TrainingConfig:
    """Configuration for training a PPO agent on the Mandl environment."""

    # Environment configuration
    network_name: NetworkName = NetworkName.CEDER1
    solution_name: Optional[NetworkName] = None
    runtime: float = 150
    buffer_time_end: float = 10
    num_flex_routes: int = 0
    num_fix_routes: int = 3
    max_route_length: int = 3
    total_vehicles: int = 12
    vehicle_capacity: int = 50
    passenger_init_mode: PassengerMode = PassengerMode.EVENLY_SPACED

    # Training configuration
    total_timesteps: int = int(1e6)
    learning_rate: float = 1e-4
    n_steps: int = 150
    batch_size: int = 150

    # Model configuration
    policy: str = "MultiInputPolicy"
    hidden_size: int = 256
    n_layers: int = 2
    device: Device = Device.AUTO

    # Environment parallelism
    num_envs: int = -1

    # Output configuration
    output_dir: str = "outputs"
    model_name: str = "ppo_mandl"

    # Submitit configuration
    use_slurm: bool = False
    slurm_partition: SlurmPartition = SlurmPartition.SINGLE
    slurm_job_name: str = "ceder1_fix"
    slurm_comment: str = ""
    slurm_gpus_per_node: int = 0
    slurm_cpus_per_task: int = 80
    slurm_time: int = 60 * 48

    # Wandb configuration
    wandb_project: str = "thesis"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None


@dataclass
class CederFlexRoutes(TrainingConfig):
    network_name: NetworkName = NetworkName.CEDER1
    num_flex_routes: int = 12
    num_fix_routes: int = 0
    total_vehicles: int = 12


@dataclass
class MandlFixRoutes(TrainingConfig):
    network_name: NetworkName = NetworkName.MANDL1
    num_flex_routes: int = 0
    num_fix_routes: int = 4
    total_vehicles: int = 99


@dataclass
class MandlFlexRoutes(TrainingConfig):
    network_name: NetworkName = NetworkName.MANDL1
    num_flex_routes: int = 99
    num_fix_routes: int = 0
    total_vehicles: int = 99


cs = ConfigStore.instance()
cs.store(name="ceder_fix", node=TrainingConfig)
cs.store(name="ceder_flex", node=CederFlexRoutes)
cs.store(name="mandl_fix", node=MandlFixRoutes)
cs.store(name="mandl_flex", node=MandlFlexRoutes)
