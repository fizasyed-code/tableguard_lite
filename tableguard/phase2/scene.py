"""Real MuJoCo scene inspection and RGB capture. No model or task is invented.

MuJoCo is imported lazily, after a requested GL backend is set. Inventory contains
model metadata for developer inspection; it must never be supplied as privileged
object state to a camera-based policy.
"""
from __future__ import annotations
import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ..common import ROOT, write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_dir(parent: Path, prefix: str) -> Path:
    import uuid
    target = Path(parent) / (prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    return target


def scan(directory: Path, max_files: int = 5000) -> dict[str, Any]:
    """List XML candidates only in a selected folder; do not execute/compile them."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'No folder: {directory}. Extract the supplied challenge assets first.')
    if not 1 <= max_files <= 100000:
        raise ValueError('max_files must be between 1 and 100000')
    rows, inspected, truncated = [], 0, False
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', 'venv', '__pycache__'}
                          and not (Path(root) / d).is_symlink())
        for name in sorted(files):
            if inspected >= max_files:
                truncated = True
                break
            p = Path(root) / name
            inspected += 1
            if p.is_symlink() or p.suffix.lower() != '.xml':
                continue
            item = {'path': str(p), 'relative_path': str(p.relative_to(directory)), 'size_bytes': p.stat().st_size}
            try:
                if p.stat().st_size > 8 * 1024 * 1024:
                    item['root_tag'] = 'not_parsed_size_limit'
                else:
                    raw = p.read_bytes()
                    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
                        raise ValueError('DTD/entity declarations are not accepted by the scanner')
                    el = ET.fromstring(raw)
                    item['root_tag'] = el.tag
                    item['model_name'] = el.attrib.get('model')
                    item['entry_candidate'] = el.tag == 'mujoco'
            except (ET.ParseError, OSError, ValueError) as e:
                item.update(root_tag='parse_error', error=str(e), entry_candidate=False)
            rows.append(item)
        if truncated:
            break
    return {'kind': 'ASSET_INVENTORY_NOT_APPROVAL', 'directory': str(directory), 'files_visited': inspected,
            'truncated': truncated, 'xml_files': rows,
            'note': 'A <mujoco> root is only a candidate. Select the entrypoint specified by the organizer; keep relative meshes/includes intact.'}


def load(scene: Path):
    p = Path(scene).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Scene does not exist: {p}. No replacement scene will be generated.')
    if p.suffix.lower() != '.xml':
        raise ValueError('This phase supports a trusted MJCF .xml entrypoint. Use the official environment adapter for other formats.')
    try:
        import mujoco
    except ImportError as e:
        raise RuntimeError('MuJoCo is not installed in this Python environment. Use the organizer-compatible environment or requirements-phase2.txt.') from e
    model = mujoco.MjModel.from_xml_path(str(p))
    return mujoco, model, p


def _names(mj, model, obj, count: int) -> list[dict]:
    return [{'id': i, 'name': mj.mj_id2name(model, obj, i)} for i in range(count)]


def inventory(scene: Path) -> dict[str, Any]:
    mj, m, p = load(scene)
    bodies = _names(mj, m, mj.mjtObj.mjOBJ_BODY, m.nbody)
    for b in bodies:
        b['parent_id'] = int(m.body_parentid[b['id']])
    joints = _names(mj, m, mj.mjtObj.mjOBJ_JOINT, m.njnt)
    for j in joints:
        i = j['id']
        kind = mj.mjtJoint(int(m.jnt_type[i])).name
        j.update(type=kind, body_id=int(m.jnt_bodyid[i]), qpos_address=int(m.jnt_qposadr[i]),
                 dof_address=int(m.jnt_dofadr[i]), limited=bool(m.jnt_limited[i]), range=m.jnt_range[i].tolist())
    actuators = _names(mj, m, mj.mjtObj.mjOBJ_ACTUATOR, m.nu)
    for a in actuators:
        i = a['id']
        a.update(transmission_type=int(m.actuator_trntype[i]), transmission_ids=m.actuator_trnid[i].tolist(),
                 control_limited=bool(m.actuator_ctrllimited[i]), control_range=m.actuator_ctrlrange[i].tolist())
    cameras = _names(mj, m, mj.mjtObj.mjOBJ_CAMERA, m.ncam)
    for c in cameras:
        c['vertical_fov_degrees'] = float(m.cam_fovy[c['id']])
    return {'kind': 'COMPILED_SCENE_INVENTORY_NOT_TASK_SUCCESS', 'created_at_utc': utc_now(),
            'scene_path': str(p), 'entry_xml_sha256': sha256(p),
            'hash_scope': 'Entry XML only; included files/meshes are not covered. Pin the organizer asset revision separately.',
            'mujoco_version': mj.__version__, 'nq': int(m.nq), 'nv': int(m.nv), 'nu': int(m.nu),
            'physics_timestep_s': float(m.opt.timestep), 'bodies': bodies, 'joints': joints,
            'actuators': actuators, 'cameras': cameras,
            'keyframes': _names(mj, m, mj.mjtObj.mjOBJ_KEY, m.nkey),
            'eligibility': 'NOT_CERTIFIED: names and counts do not prove dual-SO-101 identity or official task compatibility.'}


def set_backend(gl: str | None) -> None:
    if gl is None:
        return
    if gl not in {'egl', 'osmesa', 'glfw'}:
        raise ValueError('gl must be egl, osmesa, or glfw')
    import sys
    if 'mujoco' in sys.modules and os.environ.get('MUJOCO_GL') != gl:
        raise RuntimeError('Restart Python before changing the rendering backend; MuJoCo is already imported.')
    os.environ['MUJOCO_GL'] = gl


def reset(mj, m, keyframe: str | None = None):
    d = mj.MjData(m)
    if keyframe is None:
        mj.mj_resetData(m, d)
    else:
        k = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, keyframe)
        if k < 0:
            raise ValueError(f'No such keyframe: {keyframe}')
        mj.mj_resetDataKeyframe(m, d, k)
    mj.mj_forward(m, d)
    return d


def camera_choices(mj, m, requested: list[str] | None) -> list[tuple[str, int]]:
    if not requested:
        if m.ncam == 0:
            raise ValueError('The model has no fixed cameras. Use the official camera setup; no free-view camera is substituted.')
        return [(mj.mj_id2name(m, mj.mjtObj.mjOBJ_CAMERA, i) or f'camera_id_{i}', i) for i in range(m.ncam)]
    choices = []
    for name in requested:
        idx = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, name)
        if idx < 0:
            raise ValueError(f'Unknown camera {name!r}; use the scene inventory.')
        if idx in [i for _, i in choices]:
            raise ValueError('Duplicate selected camera')
        choices.append((name, idx))
    return choices


def make_renderer(mj, model, width: int, height: int):
    if isinstance(width, bool) or isinstance(height, bool) or not 64 <= width <= 2048 or not 64 <= height <= 2048:
        raise ValueError('Rendering width and height must each be integers between 64 and 2048')
    # Buffer allocation only; physics, cameras and robot geometry are unchanged.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    return mj.Renderer(model, height=height, width=width)


def capture(scene: Path, requested: list[str] | None, width: int, height: int,
            keyframe: str | None, output_parent: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np
    mj, m, p = load(scene)
    choices = camera_choices(mj, m, requested)
    d = reset(mj, m, keyframe)
    out = new_run_dir(output_parent, 'capture')
    report = {'kind': 'CAMERA_SMOKE_TEST_NOT_ROBOT_EXECUTION', 'created_at_utc': utc_now(),
              'scene_path': str(p), 'entry_xml_sha256': sha256(p), 'keyframe': keyframe,
              'physics_steps': 0, 'policy_calls': 0, 'task_success': None,
              'sim_time_s': float(d.time), 'gl_backend': os.environ.get('MUJOCO_GL', 'platform_default'),
              'mujoco_version': mj.__version__, 'images': [], 'rendering_completed': False}
    r = None
    try:
        r = make_renderer(mj, m, width, height)
        for name, idx in choices:
            start = time.perf_counter()
            r.update_scene(d, camera=idx)
            image = r.render().copy()
            if image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise RuntimeError(f'Unexpected RGB format: {image.shape}, {image.dtype}')
            slug = re.sub(r'[^A-Za-z0-9_-]', '_', name)[:80]
            path = out / f'camera_{idx:03d}_{slug}.png'
            Image.fromarray(image).save(path)
            report['images'].append({'camera': name, 'camera_id': idx, 'file': str(path),
                                      'sha256': sha256(path), 'width': width, 'height': height,
                                      'pixel_std': float(image.std()),
                                      'render_and_save_wall_s': time.perf_counter() - start})
        report['rendering_completed'] = True
    except Exception as e:
        report['error'] = str(e)
        raise
    finally:
        if r is not None:
            r.close()
        write_json(out / 'capture_report.json', report)
        write_json(ROOT / 'artifacts/phase2_latest_capture.json', {'report_path': str(out / 'capture_report.json')})
    return {'report_path': str(out / 'capture_report.json'), **report}
