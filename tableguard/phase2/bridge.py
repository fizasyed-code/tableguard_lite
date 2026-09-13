"""A guarded native-control runner for a reviewed local policy adapter.

No checkpoint, task policy, goal checker, or automatic recovery implementation is
supplied here. This is a baseline integration path. Never use a hardware adapter in
this module: it steps MuJoCo only. Policy outputs must ALREADY be native controls
in the explicit actuator order, not normalized LeRobot actions.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from ..common import ROOT, write_json
from .contract import read, resolved_scene, validate
from .scene import (load, inventory, reset, camera_choices, make_renderer, new_run_dir, utc_now, sha256)


@dataclass(frozen=True)
class Observation:
    instruction: str
    frame_id: str
    sim_time_s: float
    captured_at_monotonic_s: float
    rgb: Mapping[str, Any]  # Each value: uint8 [H, W, 3], fresh copy.
    robot_joint_positions: Mapping[str, float]
    robot_joint_velocities: Mapping[str, float]


@dataclass(frozen=True)
class Action:
    controls: Sequence[float] | None = None
    stop_requested: bool = False
    reason: str = ''


class Policy(Protocol):
    def reset(self, instruction: str) -> None: ...
    def select_action(self, observation: Observation) -> Action: ...


def checked_controls(values: Sequence[float] | None, order: list[str], actuators: list[dict]) -> list[float]:
    """Reject malformed/nonfinite/out-of-range actions. Never clip or guess units."""
    if values is None:
        raise ValueError('controls is required for a non-stop action')
    try:
        raw = list(values)
    except TypeError as e:
        raise ValueError('controls must be a finite one-dimensional sequence') from e
    if len(raw) != len(order):
        raise ValueError(f'Expected {len(order)} native controls, got {len(raw)}. Do not pad/truncate a model action.')
    if len(set(order)) != len(order):
        raise ValueError('Duplicate actuator names in control order')
    by_name = {a['name']: a for a in actuators}
    result = []
    for name, value in zip(order, raw):
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f'Control for {name} is not a numeric scalar')
        # Reject vectors, including length-1 NumPy arrays, instead of silent coercion.
        if getattr(value, 'ndim', 0) != 0:
            raise ValueError(f'Control for {name} is not scalar')
        try:
            v = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f'Control for {name} is not scalar') from e
        if not math.isfinite(v):
            raise ValueError(f'Nonfinite control for {name}')
        if name not in by_name:
            raise ValueError(f'Unknown actuator: {name}')
        a = by_name[name]
        if a['control_limited']:
            low, high = a['control_range']
            if not low <= v <= high:
                raise ValueError(f'{name}: control {v} outside [{low}, {high}]. No clipping applied.')
        result.append(v)
    return result


def robot_state_from_selected_joints(data: Any, joint_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Read only the explicit scalar robot joints; never return full qpos or body poses."""
    pos, vel = {}, {}
    for name in joint_names:
        item = data.joint(name)
        if len(item.qpos) != 1 or len(item.qvel) != 1:
            raise ValueError(f'{name}: only approved scalar robot joints are supported')
        p, v = float(item.qpos[0]), float(item.qvel[0])
        if not math.isfinite(p) or not math.isfinite(v):
            raise ValueError(f'{name}: nonfinite robot state')
        pos[name], vel[name] = p, v
    return pos, vel


def load_policy(factory: str, settings: dict) -> Policy:
    """Imports trusted Python code, not a security sandbox. Called only on explicit run."""
    module, function = factory.split(':')
    obj = getattr(importlib.import_module(module), function)(dict(settings))
    if not callable(getattr(obj, 'reset', None)) or not callable(getattr(obj, 'select_action', None)):
        raise TypeError('Policy must provide reset(instruction) and select_action(observation)')
    return obj


def run(config_path: Path, instruction: str, trust_policy_code: bool, output_parent: Path) -> dict[str, Any]:
    if not trust_policy_code:
        raise ValueError('Review the local adapter code, then explicitly pass --trust-policy-code. Factories execute Python; no sandbox is provided.')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Supply an actual supported natural-language instruction')
    cfg = read(config_path)
    inv = inventory(resolved_scene(cfg))
    status = validate(cfg, inv)
    if not status['valid']:
        raise ValueError('Contract is incomplete/invalid:\n- ' + '\n- '.join(status['errors']))
    from PIL import Image
    import numpy as np
    mj, m, scene = load(resolved_scene(cfg))
    d = reset(mj, m, cfg.get('reset_keyframe'))
    selected = camera_choices(mj, m, cfg['camera_names'])
    joints = [name for robot in cfg['robots'] for name in robot['joints']]
    actuator_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in cfg['actuator_order']]
    out = new_run_dir(output_parent, 'baseline_probe')
    write_json(out / 'contract_used.json', cfg)
    write_json(out / 'scene_inventory.json', inv)
    from ..hardware import inspect_hardware
    write_json(out / 'hardware_used.json', inspect_hardware(False))
    report = {'kind': 'LOCAL_POLICY_INTEGRATION_RUN_NOT_OFFICIAL_SCORE', 'run_directory': str(out),
              'created_at_utc': utc_now(), 'status': 'starting', 'instruction': instruction,
              'scene_path': str(scene), 'entry_xml_sha256': sha256(scene),
              'policy_provenance_declared': cfg['policy'], 'mujoco_version': mj.__version__,
              'actions_executed': 0, 'policy_calls': 0, 'physics_steps': 0,
              'policy_latency_s': [], 'task_success': None, 'recovery_success': None,
              'eligible_hardware_certified': False,
              'note': 'No independent scorer is attached. Policy stop is NOT task success. Backend/device is declared by the adapter owner, not verified here.',
              'timeout_note': 'Wall budget is checked between blocking calls, not a hard deadline on model inference. Simulation only.'}
    renderer = None
    start = time.perf_counter()
    import json
    with (out / 'events.jsonl').open('w', encoding='utf-8') as log:
        def event(record: dict) -> None:
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
        def observe(index: int, label: str = 'observation') -> Observation:
            frame_id = f'{index:06d}_{label}'
            images, refs = {}, {}
            for camera, camera_id in selected:
                renderer.update_scene(d, camera=camera_id)
                image = renderer.render().copy()
                if image.shape != (cfg['height'], cfg['width'], 3) or image.dtype != np.uint8:
                    raise RuntimeError('Unexpected RGB format from renderer')
                file = out / f'{frame_id}_cam{camera_id}.png'
                Image.fromarray(image).save(file)
                image.setflags(write=False)
                images[camera], refs[camera] = image, file.name
            pos, vel = robot_state_from_selected_joints(d, joints)
            obs = Observation(instruction.strip(), frame_id, float(d.time), time.perf_counter(), images, pos, vel)
            event({'event': label, 'frame_id': frame_id, 'sim_time_s': obs.sim_time_s,
                   'frames': refs, 'robot_joint_positions': pos, 'robot_joint_velocities': vel})
            return obs
        try:
            policy = load_policy(cfg['policy']['factory'], cfg['policy']['settings'])
            policy.reset(instruction.strip())
            renderer = make_renderer(mj, m, cfg['width'], cfg['height'])
            report['status'] = 'running'
            write_json(out / 'run_report.json', report)
            for index in range(cfg['max_actions']):
                if time.perf_counter() - start >= cfg['max_wall_seconds']:
                    report['status'] = 'wall_budget_exhausted'
                    break
                obs = observe(index)
                t0 = time.perf_counter()
                action = policy.select_action(obs)
                duration = time.perf_counter() - t0
                report['policy_latency_s'].append(duration)
                report['policy_calls'] += 1
                if time.perf_counter() - start >= cfg['max_wall_seconds']:
                    report['status'] = 'wall_budget_exhausted_before_dispatch'
                    break
                if not isinstance(action, Action):
                    raise TypeError('select_action must return tableguard.phase2.bridge.Action')
                if type(action.stop_requested) is not bool or not isinstance(action.reason, str):
                    raise TypeError('Action stop_requested/reason types are invalid')
                if action.stop_requested:
                    if action.controls is not None:
                        raise ValueError('A stop action must not also contain controls')
                    report['status'] = 'policy_requested_stop_unscored'
                    event({'event': 'policy_stop', 'reason': action.reason, 'task_success': None})
                    break
                values = checked_controls(action.controls, cfg['actuator_order'], inv['actuators'])
                # Single assignment in the actual model actuator order; no one-arm chunk splicing.
                d.ctrl[actuator_ids] = values
                before = float(d.time)
                warning_before = [int(w.number) for w in d.warning]
                event({'event': 'native_controls_dispatched', 'index': index, 'frame_id': obs.frame_id,
                       'actuator_order': cfg['actuator_order'], 'values': values,
                       'physics_steps_per_action': cfg['physics_steps_per_action'], 'policy_wall_s': duration})
                for _ in range(cfg['physics_steps_per_action']):
                    mj.mj_step(m, d)
                    report['physics_steps'] += 1
                report['actions_executed'] += 1
                expected = before + cfg['physics_steps_per_action'] * float(m.opt.timestep)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or abs(float(d.time) - expected) > max(1e-8, abs(expected) * 1e-9):
                    raise RuntimeError('Invalid dynamics or unexpected simulation reset; run stopped')
                if any(int(w.number) > prev for w, prev in zip(d.warning, warning_before)):
                    raise RuntimeError('New MuJoCo warning; inspect the scene/controller before continuing')
            else:
                report['status'] = 'action_budget_exhausted'
            observe(report['actions_executed'], 'final_observation')
        except Exception as e:
            report['status'] = 'error'
            report['error'] = f'{type(e).__name__}: {e}'
            event({'event': 'error', 'message': report['error']})
        finally:
            if renderer is not None:
                renderer.close()
            report['total_wall_s'] = time.perf_counter() - start
            report['final_sim_time_s'] = float(d.time)
            write_json(out / 'run_report.json', report)
            write_json(ROOT / 'artifacts/phase2_latest_run.json', {'report_path': str(out / 'run_report.json')})
    return report
