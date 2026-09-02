"""A resolved step: the function behind a configured name, checked and ready to call.

Split from :mod:`portfolio_optimizer.config.resolve` so that the request a solve step receives can
name these types without importing the resolver that produces them.
"""

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.types import Params

type StepKind = Literal["loader", "assembly", "rule", "solve_order", "build", "solve", "sink"]


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    """A step whose function, parameters, and provenance have been checked."""

    kind: StepKind
    name: str
    qualname: str
    fn: Callable[..., object]
    params: Params | None
    source_sha256: str
    params_sha256: str
    is_external: bool
    is_async: bool = False

    @property
    def params_json(self) -> dict[str, object]:
        """This step's validated params as JSON-safe values, the form the manifest carries."""
        if self.params is None:
            return {}
        return {str(key): value for key, value in self.params.model_dump(mode="json").items()}

    def invoke(self, **engine_args: object) -> object:
        """Call the function with the engine arguments and its validated params.

        For an async step this returns the coroutine; :meth:`invoke_async` awaits it.
        """
        kwargs: dict[str, object] = dict(engine_args)
        if self.params is not None:
            kwargs["params"] = self.params
        return self.fn(**kwargs)

    async def invoke_async(self, *, request: LoadRequest) -> object:
        """Await an async loader, or run a sync one in a worker thread so it cannot block the event loop; only loaders may be async."""
        if not self.is_async:
            return await asyncio.to_thread(self.invoke, request=request)
        result = self.invoke(request=request)
        if not inspect.isawaitable(result):
            msg = f"async step {self.qualname!r} returned {type(result).__name__} instead of an awaitable"
            raise TypeError(msg)
        return await result
