"""A resolved step: the function behind a configured name, checked and ready to call.

Split from :mod:`portfolio_optimizer.config.resolve` so that the request a solve step receives can
name these types without importing the resolver that produces them.
"""

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from portfolio_optimizer.config.models import ConstraintStep
from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.results import ChainState
from portfolio_optimizer.domain.types import Params

type StepKind = Literal["loader", "assembly", "rule", "solve_order", "term", "constraint", "solve", "sink"]


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    """A step whose function, parameters, and provenance have been checked."""

    kind: StepKind
    name: str
    qualname: str
    fn: Callable[..., object]
    params: Params | None
    reads_chain: bool
    source_sha256: str
    params_sha256: str
    is_external: bool
    is_async: bool = False

    def invoke(self, *, chain: ChainState | None = None, **engine_args: object) -> object:
        """Call the function with the engine arguments, its validated params, and — when it declared ``chain`` — the chain.

        For an async step this returns the coroutine; :meth:`invoke_async` awaits it.
        """
        kwargs: dict[str, object] = dict(engine_args)
        if self.params is not None:
            kwargs["params"] = self.params
        if self.reads_chain:
            if chain is None:
                msg = f"step {self.qualname!r} reads the chain but none was supplied"
                raise ValueError(msg)
            kwargs["chain"] = chain
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


@dataclass(frozen=True, slots=True)
class ResolvedConstraint:
    """A constraint as the engine consumes it: its label, the model as configured, and the resolved step behind it."""

    label: str
    spec: ConstraintStep
    step: ResolvedStep

    @property
    def reads_chain(self) -> bool:
        """True when the constraint reads what higher-priority portfolios traded; what the dependency graph is derived from."""
        return self.step.reads_chain

    @property
    def qualname(self) -> str:
        """The step's qualified name."""
        return self.step.qualname
