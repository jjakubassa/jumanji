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

import time
from typing import List, Tuple

import jax
import numpy as np

from jumanji.environments.routing.mandl.env import Mandl
from jumanji.environments.routing.mandl.profiler import (
    ProfilingContext,
    profile_function,
)


def run_episode(env, key, jitted: bool = False) -> Tuple[float, List[float]]:
    """Run a single episode and return total time and step times."""
    # JIT compile if requested
    reset_fn = jax.jit(env.reset) if jitted else env.reset
    step_fn = jax.jit(env.step) if jitted else env.step

    # Reset environment
    state, timestep = reset_fn(key)

    total_reward = 0
    step_times = []

    for _ in range(env.simulation_steps):
        start_time = time.perf_counter()

        # Random actions
        action = jax.random.randint(
            key, shape=(env.num_flex_routes,), minval=-1, maxval=env.num_nodes
        )

        # Step environment
        state, timestep = step_fn(state, action)

        elapsed = time.perf_counter() - start_time
        step_times.append(elapsed)
        total_reward += timestep.reward
        key, _ = jax.random.split(key)

    total_time = sum(step_times)
    return total_time, step_times


@profile_function
def benchmark_episodes(n_episodes: int = 2, jitted: bool = False):
    """Run multiple episodes and profile performance."""
    env = Mandl(num_flex_routes=4)
    key = jax.random.PRNGKey(0)

    episode_times = []
    all_step_times = []
    total_rewards = []

    # Warmup JIT if enabled
    if jitted:
        print("Warming up JIT...")
        _, _ = run_episode(env, key, jitted=True)
        print("Warmup complete.")

    print(f"\nRunning {n_episodes} episodes {'with' if jitted else 'without'} JIT...")
    for i in range(n_episodes):
        key, episode_key = jax.random.split(key)
        episode_time, step_times = run_episode(env, episode_key, jitted)

        episode_times.append(episode_time)
        all_step_times.extend(step_times)

    # Calculate statistics
    avg_episode_time = np.mean(episode_times)
    std_episode_time = np.std(episode_times)
    avg_step_time = np.mean(all_step_times)
    std_step_time = np.std(all_step_times)

    print(f"\nResults ({'JIT' if jitted else 'no JIT'}):")
    print(f"Average episode time: {avg_episode_time:.4f} ± {std_episode_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.4f} ± {std_step_time:.4f} seconds")

    return episode_times, all_step_times


@profile_function
def benchmark_component_tests(n_runs: int = 10, jitted: bool = False):
    """Profile individual components."""
    env = Mandl(num_flex_routes=4)
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)

    # JIT compile functions if requested
    if jitted:
        find_paths = jax.jit(env._find_paths)
        move_vehicles = jax.jit(env._move_vehicles)
        assign_passengers = jax.jit(env._assign_passengers)
    else:
        find_paths = env._find_paths
        move_vehicles = env._move_vehicles
        assign_passengers = env._assign_passengers

    # Warmup JIT if enabled
    if jitted:
        print("Warming up JIT...")
        start = jax.random.randint(key, (), 0, env.num_nodes)
        end = jax.random.randint(key, (), 0, env.num_nodes)
        find_paths(state.network, state.routes, start, end)
        move_vehicles(state.vehicles, state.routes.nodes, state.network.links)
        assign_passengers(
            state.passengers, state.vehicles, state.routes, state.network, state.current_time
        )
        print("Warmup complete.")

    results = {}

    # Profile path finding
    with ProfilingContext("Find paths"):
        times = []
        for _ in range(n_runs):
            start = jax.random.randint(key, (), 0, env.num_nodes)
            end = jax.random.randint(key, (), 0, env.num_nodes)
            start_time = time.perf_counter()
            find_paths(state.network, state.routes, start, end)
            times.append(time.perf_counter() - start_time)
        results["find_paths"] = times

    # Profile vehicle updates
    with ProfilingContext("Move vehicles"):
        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            move_vehicles(state.vehicles, state.routes.nodes, state.network.links)
            times.append(time.perf_counter() - start_time)
        results["move_vehicles"] = times

    # Profile passenger assignments
    with ProfilingContext("Assign passengers"):
        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            assign_passengers(
                state.passengers, state.vehicles, state.routes, state.network, state.current_time
            )
            times.append(time.perf_counter() - start_time)
        results["assign_passengers"] = times

    # Print statistics
    print(f"\nComponent Results ({'JIT' if jitted else 'no JIT'}):")
    for component, times in results.items():
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"{component}: {avg_time:.4f} ± {std_time:.4f} seconds per call")

    return results


if __name__ == "__main__":
    N_EPISODES = 10
    N_COMPONENT_RUNS = 100

    print("Running episode benchmarks...")
    print("\nWithout JIT:")
    unjitted_episode_times, unjitted_step_times = benchmark_episodes(N_EPISODES, jitted=False)

    print("\nWith JIT:")
    jitted_episode_times, jitted_step_times = benchmark_episodes(N_EPISODES, jitted=True)

    print("\nRunning component benchmarks...")
    print("\nWithout JIT:")
    unjitted_component_results = benchmark_component_tests(N_COMPONENT_RUNS, jitted=False)

    print("\nWith JIT:")
    jitted_component_results = benchmark_component_tests(N_COMPONENT_RUNS, jitted=True)

    # Print comparison
    print("\nPerformance Comparison (JIT vs no JIT):")
    print("Episodes:")
    jit_speedup = np.mean(unjitted_episode_times) / np.mean(jitted_episode_times)
    print(f"Average episode speedup with JIT: {jit_speedup:.2f}x")

    print("\nComponents:")
    for component in unjitted_component_results:
        unjitted_mean = np.mean(unjitted_component_results[component])
        jitted_mean = np.mean(jitted_component_results[component])
        speedup = unjitted_mean / jitted_mean
        print(f"{component} speedup with JIT: {speedup:.2f}x")
