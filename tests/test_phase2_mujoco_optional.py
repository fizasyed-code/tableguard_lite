"""Explicit opt-in tests of real engine APIs on a synthetic non-robot fixture.

Run with: python -m tableguard.phase2 selftest --with-mujoco
No rendering, pretrained model, robot identity or official task success is tested.
"""
import os
from pathlib import Path
import unittest
from tableguard.common import ROOT
from tableguard.phase2.scene import inventory, load, reset
from tableguard.phase2.bridge import robot_state_from_selected_joints


@unittest.skipUnless(os.environ.get('TABLEGUARD_RUN_MUJOCO_TESTS') == '1', 'Real MuJoCo API tests require explicit --with-mujoco; fixture is NOT SO-101')
class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.scene = ROOT / 'tests/phase2_fixtures/engineering_only.xml'

    def test_compiles_and_has_actual_camera_metadata(self):
        inv = inventory(self.scene)
        self.assertEqual(inv['nu'], 2)
        self.assertEqual(inv['cameras'][0]['name'], 'top')

    def test_actual_step_changes_simulation_time(self):
        mj, model, _ = load(self.scene)
        data = reset(mj, model)
        mj.mj_step(model, data)
        self.assertAlmostEqual(float(data.time), float(model.opt.timestep))

    def test_only_selected_robot_state_is_exposed(self):
        mj, model, _ = load(self.scene)
        data = reset(mj, model)
        pos, vel = robot_state_from_selected_joints(data, ['left_joint','right_joint'])
        self.assertEqual(len(pos), 2)
        self.assertNotIn('object_free', pos)

    def test_missing_keyframe_is_not_silently_replaced(self):
        mj, model, _ = load(self.scene)
        with self.assertRaises(ValueError):
            reset(mj, model, 'invented')
