"""Validate an explicit integration contract; validation is NOT eligibility certification."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any
from ..common import ROOT, write_json


def template(scene_path: str | None = None) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'purpose': 'Local simulation integration; not an official task specification or policy checkpoint.',
        'scene_path': scene_path,
        'asset_revision': None,
        'review': {'source': None, 'scene_permitted': False, 'policy_permitted': False,
                   'runtime_observations_permitted': False, 'robot_identity_checked': False},
        'robots': [
            {'role': 'left', 'model_declared': 'SO-101', 'base_body': None, 'joints': []},
            {'role': 'right', 'model_declared': 'SO-101', 'base_body': None, 'joints': []}],
        'camera_names': [],
        'width': 640,
        'height': 480,
        'reset_keyframe': None,
        'actuator_order': [],
        'control_contract_reference': None,
        'action_format': 'native_mujoco_ctrl',
        'physics_steps_per_action': None,
        'max_actions': 200,
        'max_wall_seconds': 120.0,
        'policy': {'factory': None, 'model_id_or_implementation': None, 'revision': None,
                   'backend_device': None, 'settings': {}},
        'official_online_cutoff': None,
        'internal_submission_target': '2026-09-15T18:00:00+09:00',
        'note': 'Fill from actual scene/policy and written rule guidance. Do not set review flags merely to bypass checks.'
    }


def create(path: Path, scene_path: str | None = None) -> Path:
    p = Path(path)
    if p.exists():
        raise FileExistsError(f'{p} exists; it was not overwritten. Keep your configured file.')
    write_json(p, template(scene_path))
    return p


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('The contract root must be a JSON object')
    return value


def resolved_scene(cfg: dict[str, Any]) -> Path:
    val = cfg.get('scene_path')
    if not isinstance(val, str) or not val.strip():
        raise ValueError('scene_path is not configured. Select the real task entrypoint first.')
    p = Path(val).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def nonblank(val: Any) -> bool:
    return isinstance(val, str) and bool(val.strip())


def validate(cfg: dict[str, Any], inv: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings = ['Review flags are user attestations, not organizer verification.',
                'Control shape/range validation is not collision avoidance or a proof of task success.',
                'Final hardware eligibility, independent scoring, licensing and submission assets require separate evidence.']
    if cfg.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    for key in ['scene_path', 'asset_revision', 'control_contract_reference']:
        if not nonblank(cfg.get(key)):
            errors.append(f'{key} must be documented')
    if cfg.get('action_format') != 'native_mujoco_ctrl':
        errors.append('Runner accepts native_mujoco_ctrl only. Decode/denormalize model actions in the reviewed policy adapter first.')
    for field in ['width', 'height']:
        v = cfg.get(field)
        if type(v) is not int or not 64 <= v <= 2048:
            errors.append(f'{field} must be an integer between 64 and 2048')
    for field in ['physics_steps_per_action', 'max_actions']:
        v = cfg.get(field)
        if type(v) is not int or not 1 <= v <= 10000:
            errors.append(f'{field} must be an integer between 1 and 10000; obtain action timing from the controller contract')
    t = cfg.get('max_wall_seconds')
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= 3600:
        errors.append('max_wall_seconds must be finite and in (0, 3600]')
    review = cfg.get('review', {})
    if not isinstance(review, dict):
        review = {}
    if not nonblank(review.get('source')):
        errors.append('review.source must identify the actual guidance reviewed')
    for key in ['scene_permitted', 'policy_permitted', 'runtime_observations_permitted', 'robot_identity_checked']:
        if review.get(key) is not True:
            errors.append(f'review.{key} is not confirmed')
    robots = cfg.get('robots')
    if not isinstance(robots, list) or len(robots) != 2 or not all(isinstance(r, dict) for r in robots):
        errors.append('robots must contain exactly two explicit robot entries')
        robots = []
    joints, bases = [], []
    roles = [r.get('role') for r in robots]
    if sorted(str(x) for x in roles) != ['left', 'right']:
        errors.append('Robot roles must be left and right, once each')
    for r in robots:
        label = r.get('role', '?')
        if r.get('model_declared') != 'SO-101':
            errors.append(f'{label}: expected SO-101 declaration, verified from provenance rather than names')
        if not nonblank(r.get('base_body')):
            errors.append(f'{label}: base_body is not configured')
        else:
            bases.append(r['base_body'])
        js = r.get('joints')
        if not isinstance(js, list) or not js or not all(nonblank(x) for x in js):
            errors.append(f'{label}: joints must explicitly list permitted scalar robot joints')
        else:
            joints.extend(js)
    if len(set(bases)) != len(bases):
        errors.append('The two base bodies must be distinct')
    if len(set(joints)) != len(joints):
        errors.append('Repeated or shared robot joints are not accepted')
    for field in ['camera_names', 'actuator_order']:
        arr = cfg.get(field)
        if not isinstance(arr, list) or not arr or not all(nonblank(x) for x in arr):
            errors.append(f'{field} must explicitly list actual names')
        elif len(set(arr)) != len(arr):
            errors.append(f'{field} contains duplicates')
    policy = cfg.get('policy', {})
    if not isinstance(policy, dict):
        policy = {}
    fac = policy.get('factory')
    if not isinstance(fac, str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', fac):
        errors.append('policy.factory must identify a trusted local module:function')
    for key in ['model_id_or_implementation', 'revision', 'backend_device']:
        if not nonblank(policy.get(key)):
            errors.append(f'policy.{key} is required for provenance')
    if not isinstance(policy.get('settings'), dict):
        errors.append('policy.settings must be an object')
    if inv is not None:
        actual_cameras = {c['name'] for c in inv['cameras']}
        for name in cfg.get('camera_names', []) if isinstance(cfg.get('camera_names'), list) else []:
            if name not in actual_cameras:
                errors.append(f'Unknown camera: {name!r}')
        actual_acts = [a['name'] for a in inv['actuators']]
        order = cfg.get('actuator_order', [])
        if not isinstance(order, list) or any(x is None for x in actual_acts) or len(order) != inv['nu'] or set(order) != set(actual_acts):
            errors.append('actuator_order must cover each named model actuator exactly once. No action dimensions are inferred.')
        by_joint = {j['name']: j for j in inv['joints']}
        by_body = {b['name']: b for b in inv['bodies']}
        parents = {b['id']: b['parent_id'] for b in inv['bodies']}
        def under(child: int, base: int) -> bool:
            visited = set()
            while child not in visited:
                if child == base:
                    return True
                visited.add(child)
                nxt = parents.get(child, child)
                if nxt == child:
                    break
                child = nxt
            return False
        base_ids = [by_body[name]['id'] for name in bases if name in by_body]
        if any(i == 0 for i in base_ids):
            errors.append('The world body is not a robot base')
        if len(base_ids) == 2 and (under(base_ids[0], base_ids[1]) or under(base_ids[1], base_ids[0])):
            errors.append('Robot bases must be separate subtrees, not ancestors of each other')
        for r in robots:
            base = by_body.get(r.get('base_body'))
            if base is None:
                errors.append(f'Unknown robot base: {r.get("base_body")!r}')
            for name in r.get('joints', []) if isinstance(r.get('joints'), list) else []:
                j = by_joint.get(name)
                if j is None:
                    errors.append(f'Unknown robot joint: {name!r}')
                elif j['type'] not in {'mjJNT_HINGE', 'mjJNT_SLIDE'}:
                    errors.append(f'{name}: not an allowed scalar robot joint; full/free object state is excluded')
                elif base is not None and not under(j['body_id'], base['id']):
                    errors.append(f'{name}: not under its declared robot base')
        key = cfg.get('reset_keyframe')
        if key is not None and key not in {k['name'] for k in inv['keyframes']}:
            errors.append(f'Unknown reset keyframe: {key!r}')
    return {'kind': 'INTEGRATION_CONTRACT_CHECK_NOT_ELIGIBILITY', 'valid': not errors,
            'errors': errors, 'warnings': warnings, 'scene_checked': inv is not None,
            'policy_imported': False, 'task_success': None}
