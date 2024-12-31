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

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

import jax


class ProfilingContext:
    """Context manager for profiling code blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        # print(f"{self.name}: {elapsed:.4f} seconds")


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with ProfilingContext(func.__name__):
            result = func(*args, **kwargs)
            # Ensure computation is complete before ending profiling
            if isinstance(result, jax.Array):
                result.block_until_ready()
            return result

    return wrapper


@contextmanager
def trace_context(name: Optional[str] = None):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # print(f"{name if name else 'unnamed'}: {elapsed:.4f} seconds")
