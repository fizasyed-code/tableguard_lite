import unittest
from tableguard.phase2.bridge import checked_controls, robot_state_from_selected_joints, run

ACTS = [{'name':'left', 'control_limited':True, 'control_range':[-1,1]},
        {'name':'right', 'control_limited':True, 'control_range':[-2,2]}]


class ControlTests(unittest.TestCase):
    def test_valid_vector(self):
        self.assertEqual(checked_controls([.5,1], ['left','right'], ACTS), [.5,1.0])

    def test_order_is_explicit(self):
        self.assertEqual(checked_controls([1.5,.5], ['right','left'], ACTS), [1.5,.5])

    def test_no_padding(self):
        with self.assertRaises(ValueError):
            checked_controls([0], ['left','right'], ACTS)

    def test_no_truncation(self):
        with self.assertRaises(ValueError):
            checked_controls([0,0,0], ['left','right'], ACTS)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            checked_controls([float('nan'),0], ['left','right'], ACTS)

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            checked_controls([float('inf'),0], ['left','right'], ACTS)

    def test_out_of_range_not_clipped(self):
        with self.assertRaises(ValueError):
            checked_controls([1.01,0], ['left','right'], ACTS)

    def test_boundary_values_allowed(self):
        self.assertEqual(checked_controls([-1,2], ['left','right'], ACTS), [-1,2])

    def test_invalid_numeric_types(self):
        for value in ['0', True, [0], None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_controls([value,0], ['left','right'], ACTS)

    def test_none_rejected(self):
        with self.assertRaises(ValueError):
            checked_controls(None, ['left','right'], ACTS)

    def test_unknown_actuator(self):
        with self.assertRaises(ValueError):
            checked_controls([0,0], ['fake','right'], ACTS)

    def test_duplicate_actuator(self):
        with self.assertRaises(ValueError):
            checked_controls([0,0], ['left','left'], ACTS)

    def test_no_policy_import_without_explicit_trust(self):
        from pathlib import Path
        with self.assertRaises(ValueError):
            run(Path('absent.json'), 'instruction', False, Path('results'))

    def test_state_allowlist_does_not_read_objects(self):
        class Joint:
            qpos = [.2]
            qvel = [.1]
        class Data:
            def __init__(self): self.reads = []
            def joint(self, name): self.reads.append(name); return Joint()
            @property
            def qpos(self): raise AssertionError('Full qpos must not be exposed')
        d = Data()
        p, v = robot_state_from_selected_joints(d, ['left_joint','right_joint'])
        self.assertEqual(d.reads, ['left_joint','right_joint'])
        self.assertEqual(set(p), {'left_joint','right_joint'})

    def test_non_scalar_state_rejected(self):
        class Joint: qpos = [1,2,3]; qvel = [0,0,0]
        class Data:
            def joint(self, name): return Joint()
        with self.assertRaises(ValueError):
            robot_state_from_selected_joints(Data(), ['object'])
