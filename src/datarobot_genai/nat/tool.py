# Copyright 2026 DataRobot, Inc. and its affiliates.
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

import asyncio
import inspect
from collections.abc import AsyncGenerator
from collections.abc import Callable
from functools import wraps
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


def _sync_to_async(fn: Callable) -> Callable:
    """Wrap a sync function in an async one."""

    @wraps(fn)
    async def async_wrapper(*args: object, **kwargs: object) -> Any:  # noqa: ANN401
        return await asyncio.to_thread(fn, *args, **kwargs)

    return async_wrapper


def nat_tool(fn: Callable, name: str, description: str | None = None) -> Callable:
    """Decorate a function as a NAT tool."""

    class NatToolConfig(FunctionBaseConfig, name=name):  # type: ignore[call-arg]
        pass

    @register_function(
        config_type=NatToolConfig,
    )
    async def wrapper(config: NatToolConfig, builder: Builder) -> AsyncGenerator:
        # NAT expects a coroutine function, so we wrap sync functions in an async one.
        if not inspect.iscoroutinefunction(fn):
            fn_for_nat = _sync_to_async(fn)
        else:
            fn_for_nat = fn
        yield FunctionInfo.from_fn(
            fn=fn_for_nat,
            description=description,
        )

    return fn
