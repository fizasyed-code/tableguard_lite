import unittest
from dataclasses import replace
from tableguard.fixtures import fixture_goals, run_fixture_demo
from tableguard.supervisor import Context, Goal, Lifecycle, VisualState, Response, decide
from tableguard.adapters import UnconfiguredRobotAdapter

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.ctx = Context(10.1, 20.0)
    def test_normal_continues(self):
        self.assertEqual(decide(fixture_goals(), self.ctx).response, Response.CONTINUE)
    def test_local_repair_requests_only_cup(self):
        d = decide(fixture_goals(VisualState.VIOLATED), self.ctx)
        self.assertEqual(d.response, Response.REPAIR)
        self.assertEqual(d.affected_ids, ('cup_right_of_plate',))
        self.assertEqual(set(d.protected_ids), {'plate_in_region','fork_left_of_plate'})
    def test_pending_goal_is_not_a_failure(self):
        self.assertEqual(decide(fixture_goals(VisualState.VIOLATED, Lifecycle.PENDING),self.ctx).response, Response.CONTINUE)
    def test_active_not_due_ignored(self):
        self.assertEqual(decide(fixture_goals(VisualState.VIOLATED, Lifecycle.ACTIVE),self.ctx).response, Response.CONTINUE)
    def test_active_due_checked(self):
        goals = list(fixture_goals(VisualState.VIOLATED,Lifecycle.ACTIVE))
        goals[-1] = replace(goals[-1],due_now=True)
        self.assertEqual(decide(tuple(goals), self.ctx).response, Response.REPAIR)
    def test_unknown_holds(self):
        self.assertEqual(decide(fixture_goals(VisualState.UNKNOWN),self.ctx).response, Response.HOLD)
    def test_stale_holds(self):
        self.assertEqual(decide(fixture_goals(),replace(self.ctx,now_s=12.0)).response, Response.HOLD)
    def test_future_frame_holds(self):
        self.assertEqual(decide(fixture_goals(),replace(self.ctx,now_s=9.9)).response, Response.HOLD)
    def test_freshness_boundary_accepted(self):
        self.assertEqual(decide(fixture_goals(),replace(self.ctx,now_s=11.0)).response, Response.CONTINUE)
    def test_budget_exhausted_stops(self):
        self.assertEqual(decide(fixture_goals(VisualState.VIOLATED),replace(self.ctx,attempts=2)).response,Response.STOP)
    def test_last_allowed_attempt(self):
        self.assertEqual(decide(fixture_goals(VisualState.VIOLATED),replace(self.ctx,attempts=1)).response,Response.REPAIR)
    def test_satisfied_after_last_repair_continues(self):
        self.assertEqual(decide(fixture_goals(),replace(self.ctx,attempts=2)).response,Response.CONTINUE)
    def test_timeout_stops(self):
        self.assertEqual(decide(fixture_goals(),replace(self.ctx,elapsed_s=120.0)).response,Response.STOP)
    def test_not_at_boundary_holds(self):
        self.assertEqual(decide(fixture_goals(VisualState.VIOLATED),replace(self.ctx,at_supported_boundary=False)).response,Response.HOLD)
    def test_unsupported_repair_stops(self):
        goals=list(fixture_goals(VisualState.VIOLATED))
        goals[-1]=replace(goals[-1],repair_supported=False)
        self.assertEqual(decide(tuple(goals),self.ctx).response,Response.STOP)
    def test_multiple_violations_stop(self):
        goals=list(fixture_goals(VisualState.VIOLATED))
        goals[1]=replace(goals[1],visual_state=VisualState.VIOLATED,repair_supported=True)
        self.assertEqual(decide(tuple(goals),self.ctx).response,Response.STOP)
    def test_unknown_takes_precedence_over_repair(self):
        goals=list(fixture_goals(VisualState.VIOLATED))
        goals[1]=replace(goals[1],visual_state=VisualState.UNKNOWN)
        self.assertEqual(decide(tuple(goals),self.ctx).response,Response.HOLD)
    def test_dependency_update_invalidates_old_evidence(self):
        ctx=replace(self.ctx,dependency_updates=(('plate',10.05),))
        self.assertEqual(decide(fixture_goals(),ctx).response,Response.HOLD)
    def test_revalidated_dependency_can_continue(self):
        ctx=replace(self.ctx,dependency_updates=(('plate',10.05),))
        goals=tuple(replace(g,observed_at_s=10.1) for g in fixture_goals())
        self.assertEqual(decide(goals,ctx).response,Response.CONTINUE)
    def test_irrelevant_change_does_not_trigger_repair(self):
        ctx=replace(self.ctx,dependency_updates=(('background',10.05),))
        self.assertEqual(decide(fixture_goals(),ctx).response,Response.CONTINUE)
    def test_empty_goals_stop(self):
        self.assertEqual(decide((),self.ctx).response,Response.STOP)
    def test_all_pending_continue(self):
        goals=tuple(replace(g,lifecycle=Lifecycle.PENDING) for g in fixture_goals())
        self.assertEqual(decide(goals,self.ctx).response,Response.CONTINUE)
    def test_duplicate_goals_rejected(self):
        goals=fixture_goals()
        with self.assertRaises(ValueError): decide(goals+(goals[0],),self.ctx)
    def test_nan_timestamp_rejected(self):
        with self.assertRaises(ValueError): replace(fixture_goals()[0], observed_at_s=float('nan'))
    def test_negative_attempt_rejected(self):
        with self.assertRaises(ValueError): replace(self.ctx,attempts=-1)
    def test_nan_context_rejected(self):
        with self.assertRaises(ValueError): replace(self.ctx,now_s=float('nan'))
    def test_future_dependency_rejected(self):
        with self.assertRaises(ValueError): replace(self.ctx,dependency_updates=(('plate',11.0),))
    def test_duplicate_dependency_rejected(self):
        with self.assertRaises(ValueError): replace(self.ctx,dependency_updates=(('plate',10.0),('plate',10.05)))
    def test_pending_due_conflict_rejected(self):
        with self.assertRaises(ValueError): replace(fixture_goals()[0],lifecycle=Lifecycle.PENDING,due_now=True)
    def test_string_enums_rejected(self):
        with self.assertRaises(TypeError): replace(fixture_goals()[0],visual_state='satisfied')
    def test_demo_does_not_claim_execution(self):
        self.assertFalse(run_fixture_demo()['robot_executed'])
    def test_robot_adapter_cannot_fake_repair(self):
        with self.assertRaises(NotImplementedError):
            UnconfiguredRobotAdapter().dispatch_valid_joint_repair(decide(fixture_goals(VisualState.VIOLATED),self.ctx))
    def test_polling_does_not_consume_attempt(self):
        a=decide(fixture_goals(VisualState.VIOLATED),self.ctx)
        b=decide(fixture_goals(VisualState.VIOLATED),self.ctx)
        self.assertEqual(a,b)
        self.assertEqual(self.ctx.attempts,0)
