from __future__ import annotations
import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from ..common import ROOT, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description='TableGuard Phase 2: actual scene I/O and reviewed policy integration. No checkpoint is included.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight', help='Check this Python environment; no installs or downloads')
    p.add_argument('--probe-devices', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/phase2_preflight.json')
    p = sub.add_parser('init', help='Create a separate unconfigured integration contract; never overwrite')
    p.add_argument('--scene')
    p.add_argument('--output', type=Path, default=ROOT / 'configs/phase2_task_contract.json')
    p = sub.add_parser('scan', help='Find XML candidates in a selected extracted asset folder')
    p.add_argument('--directory', type=Path, default=ROOT / 'assets/challenge')
    p.add_argument('--max-files', type=int, default=5000)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/phase2_scene_candidates.json')
    p = sub.add_parser('inspect', help='Compile a real scene and record cameras, joints, actuators; no motion')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/phase2_scene_inventory.json')
    p = sub.add_parser('capture', help='Save actual initial RGB views; no policy or physics step')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--camera', action='append', help='Actual camera name; repeat for multiple. Default: every fixed camera.')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--keyframe')
    p.add_argument('--gl', choices=['egl', 'osmesa', 'glfw'])
    p.add_argument('--output-parent', type=Path, default=ROOT / 'artifacts/phase2_captures')
    p = sub.add_parser('check', help='Validate an explicit integration contract, not eligibility')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/phase2_task_contract.json')
    p.add_argument('--with-scene', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/phase2_contract_check.json')
    p = sub.add_parser('run', help='Run a reviewed local policy using real MuJoCo native controls; unscored')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/phase2_task_contract.json')
    p.add_argument('--instruction', required=True)
    p.add_argument('--trust-policy-code', action='store_true')
    p.add_argument('--gl', choices=['egl', 'osmesa', 'glfw'])
    p.add_argument('--output-parent', type=Path, default=ROOT / 'results/phase2')
    p = sub.add_parser('selftest', help='Run all offline tests; optional real MuJoCo tests use a non-robot fixture')
    p.add_argument('--with-mujoco', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'preflight':
            from .preflight import inspect
            report = inspect(args.probe_devices)
            write_json(args.output, report)
            print(json.dumps(report, indent=2)); print(f'Saved: {args.output}')
            return 0 if report['scene_dependencies_present'] else 2
        if args.command == 'init':
            from .contract import create
            print(f'Created: {create(args.output, args.scene)}')
            print('This is an unconfigured integration contract, NOT a task policy.')
        elif args.command == 'scan':
            from .scene import scan
            report = scan(args.directory, args.max_files)
            write_json(args.output, report)
            print(json.dumps(report, indent=2)); print(f'Saved: {args.output}')
            if not report['xml_files']:
                print('No XML found. Obtain/extract the actual task assets; do not download an unrelated dataset as a substitute.')
        elif args.command == 'inspect':
            from .scene import inventory
            report = inventory(args.scene)
            write_json(args.output, report)
            print(json.dumps({k: report[k] for k in ['scene_path','nq','nv','nu','cameras','actuators','keyframes','eligibility']}, indent=2))
            print(f'Full inventory saved: {args.output}')
        elif args.command == 'capture':
            from .scene import set_backend, capture
            set_backend(args.gl)
            print(json.dumps(capture(args.scene, args.camera, args.width, args.height, args.keyframe, args.output_parent), indent=2))
        elif args.command == 'check':
            from .contract import read, resolved_scene, validate
            from .scene import inventory
            cfg = read(args.config)
            report = validate(cfg, inventory(resolved_scene(cfg)) if args.with_scene else None)
            write_json(args.output, report)
            print(json.dumps(report, indent=2)); print(f'Saved: {args.output}')
            return 0 if report['valid'] else 2
        elif args.command == 'run':
            from .scene import set_backend
            set_backend(args.gl)
            from .bridge import run
            report = run(args.config, args.instruction, args.trust_policy_code, args.output_parent)
            print(json.dumps(report, indent=2))
            # Zero only means the integration loop ended without an exception, NEVER task success.
            return 2 if report['status'] == 'error' else 0
        elif args.command == 'selftest':
            if args.with_mujoco:
                os.environ['TABLEGUARD_RUN_MUJOCO_TESTS'] = '1'
            else:
                os.environ.pop('TABLEGUARD_RUN_MUJOCO_TESTS', None)
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            report = {'kind': 'SOFTWARE_TESTS_NOT_HACKATHON_EPISODES', 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'skipped': len(result.skipped), 'passed': result.wasSuccessful(),
                      'with_mujoco_requested': args.with_mujoco,
                      'note': 'Optional engine tests use a synthetic engineering fixture, not SO-101, not table setting.'}
            write_json(ROOT / 'artifacts/phase2_unit_test_summary.json', report)
            return 0 if result.wasSuccessful() else 1
        return 0
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        print('No replacement assets, guessed action schema, or robot success were generated.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
