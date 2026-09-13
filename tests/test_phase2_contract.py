"""Offline schema tests on invented metadata, NOT a real robot scene."""
import copy
from pathlib import Path
import tempfile
import unittest
from tableguard.phase2.contract import template, validate, create, read


def fake_inventory():
    return {'nu': 2,
            'bodies': [{'id': 0, 'name': 'world', 'parent_id': 0}, {'id': 1, 'name': 'left_base', 'parent_id': 0},
                       {'id': 2, 'name': 'right_base', 'parent_id': 0}, {'id': 3, 'name': 'object', 'parent_id': 0}],
            'joints': [{'name': 'left_joint', 'type': 'mjJNT_HINGE', 'body_id': 1},
                       {'name': 'right_joint', 'type': 'mjJNT_SLIDE', 'body_id': 2},
                       {'name': 'object_free', 'type': 'mjJNT_FREE', 'body_id': 3}],
            'cameras': [{'id': 0, 'name': 'top'}],
            'actuators': [{'name': 'left_motor'}, {'name': 'right_motor'}], 'keyframes': [{'name': 'home'}]}


def valid_config():
    c = template('tests/phase2_fixtures/engineering_only.xml')
    c.update(asset_revision='TEST ONLY', control_contract_reference='Invented unit-test mapping',
             camera_names=['top'], actuator_order=['left_motor','right_motor'], physics_steps_per_action=10)
    c['review'] = {'source':'UNIT TEST NOT ORGANIZER APPROVAL', 'scene_permitted':True, 'policy_permitted':True,
                   'runtime_observations_permitted':True, 'robot_identity_checked':True}
    c['robots'][0].update(base_body='left_base', joints=['left_joint'])
    c['robots'][1].update(base_body='right_base', joints=['right_joint'])
    c['policy'].update(factory='some_module:create', model_id_or_implementation='test', revision='test', backend_device='test')
    return c


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.c, self.i = valid_config(), fake_inventory()

    def test_default_is_not_ready(self):
        self.assertFalse(validate(template())['valid'])

    def test_valid_synthetic_mapping(self):
        self.assertTrue(validate(self.c, self.i)['valid'])

    def test_unknown_camera(self):
        self.c['camera_names'] = ['made_up']
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_unnamed_model_actuator(self):
        self.i['actuators'][0]['name'] = None
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_duplicate_actuator(self):
        self.c['actuator_order'] = ['left_motor','left_motor']
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_no_six_or_twelve_dimension_assumption(self):
        self.assertTrue(validate(self.c, self.i)['valid'])

    def test_missing_actuator(self):
        self.c['actuator_order'] = ['left_motor']
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_foreign_free_joint_is_rejected(self):
        self.c['robots'][0]['joints'].append('object_free')
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_other_robot_joint_is_rejected(self):
        self.c['robots'][0]['joints'] = ['right_joint']
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_robot_base_not_world(self):
        self.c['robots'][0]['base_body'] = 'world'
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_nested_bases_rejected(self):
        self.i['bodies'][2]['parent_id'] = 1
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_false_attestation(self):
        self.c['review']['policy_permitted'] = False
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_missing_review_source(self):
        self.c['review']['source'] = None
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_both_roles_required(self):
        self.c['robots'][1]['role'] = 'left'
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_normalized_actions_rejected(self):
        self.c['action_format'] = 'normalized_lerobot_action'
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_invalid_timing(self):
        for value in [0, -1, True, 1.5, None, 999999]:
            with self.subTest(value=value):
                self.c['physics_steps_per_action'] = value
                self.assertFalse(validate(self.c, self.i)['valid'])

    def test_invalid_wall_budget(self):
        for value in [0, float('nan'), float('inf'), True, '12']:
            with self.subTest(value=value):
                self.c['max_wall_seconds'] = value
                self.assertFalse(validate(self.c, self.i)['valid'])

    def test_keyframe_validation(self):
        self.c['reset_keyframe'] = 'missing'
        self.assertFalse(validate(self.c, self.i)['valid'])
        self.c['reset_keyframe'] = 'home'
        self.assertTrue(validate(self.c, self.i)['valid'])

    def test_factory_not_executed_or_accepted_as_code(self):
        self.c['policy']['factory'] = "__import__('os').system('bad')"
        self.assertFalse(validate(self.c, self.i)['valid'])

    def test_no_configuration_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'contract.json'
            create(path)
            old = path.read_bytes()
            with self.assertRaises(FileExistsError):
                create(path)
            self.assertEqual(old, path.read_bytes())

    def test_source_is_not_mutated(self):
        original = copy.deepcopy(self.c)
        validate(self.c, self.i)
        self.assertEqual(original, self.c)

    def test_non_object_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.json'; p.write_text('[]')
            with self.assertRaises(ValueError):
                read(p)
