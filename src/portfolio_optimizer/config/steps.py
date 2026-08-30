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
from portfolio_optimizer.domain.types import Params

type StepKind = Literal["portfolios", "loader", "constraints_loader", "assembly", "rule", "solve_order", "term", "constraint", "solve", "sink"]


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    """A step whose function, parameters, and provenance have been checked."""

    kind: StepKind
    name: str
    qualname: str
    fn: Callable[..., object]
    params: Params | None
    context_name: str | None
    source_sha256: str
    params_sha256: str
    is_external: bool
    is_async: bool = False

    @property
    def needs_context(self) -> bool:
        """True when the function declares the optional context argument for its kind."""
        return self.context_name is not None

    def invoke(self, *, context: object | None = None, **engine_args: object) -> object:
        """Call the function with the engine arguments, its validated params, and its context.

        For an async step this returns the coroutine; :meth:`invoke_async` awaits it.
        """
        kwargs: dict[str, object] = dict(engine_args)
        if self.params is not None:
            kwargs["params"] = self.params
        if self.context_name is not None:
            if context is None:
                msg = f"step {self.qualname!r} requires {self.context_name!r} but none was supplied"
                raise ValueError(msg)
            kwargs[self.context_name] = context
        return self.fn(**kwargs)

    async def invoke_async(self, *, context: object | None = None, **engine_args: object) -> object:
        """Await an async step, or run a sync step in a worker thread so it cannot block the event loop."""
        if not self.is_async:
            return await asyncio.to_thread(self.invoke, context=context, **engine_args)
        result = self.invoke(context=context, **engine_args)
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
        return self.step.needs_context

    @property
    def qualname(self) -> str:
        """The step's qualified name."""
        return self.step.qualname
