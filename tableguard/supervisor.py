"""Pure decision logic for boundary-level tests. Does not move a robot.

All evidence is supplied by the caller. A real vision monitor, compatible action adapter,
and independent evaluator are separate integration requirements.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math

class Lifecycle(str, Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    COMPLETED = 'completed'

class VisualState(str, Enum):
    SATISFIED = 'satisfied'
    VIOLATED = 'violated'
    UNKNOWN = 'unknown'

class Response(str, Enum):
    CONTINUE = 'continue'
    REPAIR = 'repair'
    HOLD = 'hold'
    STOP = 'stop'

@dataclass(frozen=True)
class Goal:
    goal_id: str
    lifecycle: Lifecycle
    visual_state: VisualState
    observed_at_s: float
    dependencies: tuple[str, ...] = ()
    due_now: bool = False
    repair_supported: bool = False

    def __post_init__(self):
        if not self.goal_id or not isinstance(self.goal_id, str):
            raise ValueError('goal_id must be a non-empty string')
        if not isinstance(self.lifecycle, Lifecycle) or not isinstance(self.visual_state, VisualState):
            raise TypeError('Use Lifecycle and VisualState enum values')
        if not math.isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError('observation timestamp must be finite and nonnegative')
        if self.lifecycle == Lifecycle.PENDING and self.due_now:
            raise ValueError('A pending goal cannot be due for completion checking')

@dataclass(frozen=True)
class Context:
    now_s: float
    elapsed_s: float
    at_supported_boundary: bool = True
    attempts: int = 0
    max_attempts: int = 2
    timeout_s: float = 120.0
    freshness_s: float = 1.0
    # Runtime-observed dependency updates, NOT evaluator disturbance labels.
    dependency_updates: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for name in ['now_s','elapsed_s','timeout_s','freshness_s']:
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.timeout_s <= 0 or self.freshness_s <= 0:
            raise ValueError('timeout and freshness must be positive')
        for n in [self.attempts, self.max_attempts]:
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError('attempt counts must be nonnegative integers')
        keys = [key for key, _ in self.dependency_updates]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate dependency update')
        if any(not math.isfinite(t) or t < 0 or t > self.now_s for _, t in self.dependency_updates):
            raise ValueError('dependency timestamps must be finite, nonnegative, and no later than now')

@dataclass(frozen=True)
class Decision:
    response: Response
    reason: str
    affected_ids: tuple[str, ...] = ()
    protected_ids: tuple[str, ...] = ()


def decide(goals: tuple[Goal, ...], ctx: Context) -> Decision:
    """Conservative, stateless local-scope selector with explicit refusal cases.

    HOLD is a REQUEST to a compatible adapter. It is not permission to splice a joint
    action chunk or an implemented emergency stop. The adapter must handle safe boundaries.
    Attempts are incremented by the worker after actual dispatch, not by polling this function.
    """
    if len({g.goal_id for g in goals}) != len(goals):
        raise ValueError('duplicate goal IDs')
    if ctx.elapsed_s >= ctx.timeout_s:
        return Decision(Response.STOP, 'episode_timeout')
    if not goals:
        return Decision(Response.STOP, 'missing_goal_specification')
    if not ctx.at_supported_boundary:
        return Decision(Response.HOLD, 'await_supported_boundary')
    relevant = [g for g in goals if g.lifecycle == Lifecycle.COMPLETED or g.due_now]
    if not relevant:
        return Decision(Response.CONTINUE, 'no_goal_due_yet')
    updates = dict(ctx.dependency_updates)
    unknown = []
    violated = []
    valid = []
    for g in relevant:
        age = ctx.now_s - g.observed_at_s
        dependency_stale = any(updates.get(d, -1) > g.observed_at_s for d in g.dependencies)
        if age < 0 or age > ctx.freshness_s or dependency_stale or g.visual_state == VisualState.UNKNOWN:
            unknown.append(g)
        elif g.visual_state == VisualState.VIOLATED:
            violated.append(g)
        else:
            valid.append(g)
    affected = tuple(g.goal_id for g in unknown + violated)
    protected = tuple(g.goal_id for g in valid)
    if unknown:
        return Decision(Response.HOLD, 'reobserve_unknown_or_stale_goals', affected, protected)
    if not violated:
        return Decision(Response.CONTINUE, 'due_goals_satisfied', (), protected)
    if ctx.attempts >= ctx.max_attempts:
        return Decision(Response.STOP, 'repair_budget_exhausted', affected, protected)
    if len(violated) != 1 or not violated[0].repair_supported:
        return Decision(Response.STOP, 'no_supported_single_goal_repair', affected, protected)
    return Decision(Response.REPAIR, 'request_supported_local_repair', affected, protected)
