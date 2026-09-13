"""Hand-authored unit-test fixtures: not images, predictions, or robot episodes."""
from dataclasses import asdict
from .supervisor import Context, Goal, Lifecycle, VisualState, decide


def fixture_goals(cup_state=VisualState.SATISFIED, cup_lifecycle=Lifecycle.COMPLETED):
    return (
        Goal('plate_in_region', Lifecycle.COMPLETED, VisualState.SATISFIED, 10.0, ('plate',)),
        Goal('fork_left_of_plate', Lifecycle.COMPLETED, VisualState.SATISFIED, 10.0, ('fork','plate')),
        Goal('cup_right_of_plate', cup_lifecycle, cup_state, 10.0, ('cup','plate'), repair_supported=True),
    )


def run_fixture_demo():
    cases = [
        ('normal', fixture_goals(), Context(10.1, 20.0)),
        ('local_violation', fixture_goals(VisualState.VIOLATED), Context(10.1, 20.0)),
        ('cup_not_attempted', fixture_goals(VisualState.VIOLATED, Lifecycle.PENDING), Context(10.1, 20.0)),
        ('occluded_evidence', fixture_goals(VisualState.UNKNOWN), Context(10.1, 20.0)),
        ('stale_evidence', fixture_goals(), Context(12.0, 20.0)),
        ('budget_exhausted', fixture_goals(VisualState.VIOLATED), Context(10.1, 20.0, attempts=2)),
        ('plate_changed', fixture_goals(), Context(10.2, 20.0, dependency_updates=(('plate',10.1),))),
    ]
    rows = []
    for name, goals, ctx in cases:
        d = decide(goals, ctx)
        rows.append({'fixture': name, 'response': d.response.value, 'reason': d.reason,
                     'affected_ids': list(d.affected_ids), 'protected_ids': list(d.protected_ids)})
    return {'kind': 'HAND_AUTHORED_LOGIC_FIXTURES_NOT_ROBOT_RESULTS', 'robot_executed': False, 'decisions': rows}
