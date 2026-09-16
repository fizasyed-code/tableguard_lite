from __future__ import annotations
import argparse
import json
import sys
import unittest
from pathlib import Path
from .common import ROOT, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description='TableGuard-Lite Phase 1. No robot policy is included.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('hardware', help='Report local hardware without model downloads')
    p.add_argument('--probe-frameworks', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/hardware_report.json')
    sub.add_parser('catalog', help='Show verified public dataset reference list')
    p = sub.add_parser('fetch', help='Download metadata; optionally ONE v2 episode and ONE camera')
    p.add_argument('--repo', default='lerobot/svla_so101_pickplace')
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--media', action='store_true')
    p.add_argument('--camera')
    p.add_argument('--max-mb', type=float, default=50.0)
    p.add_argument('--revision', default='main')
    p = sub.add_parser('visualize', help='Save plots of a downloaded public reference sample')
    p.add_argument('--manifest', type=Path)
    sub.add_parser('demo', help='Run hand-authored decision fixtures, NOT simulated robot episodes')
    sub.add_parser('selftest', help='Run CPU-only offline unit tests')
    args = parser.parse_args()
    try:
        if args.command == 'hardware':
            from .hardware import inspect_hardware
            report = inspect_hardware(args.probe_frameworks)
            write_json(args.output, report)
            print(json.dumps(report, indent=2)); print(f'Saved: {args.output}')
        elif args.command == 'catalog':
            print((ROOT/'configs/datasets.json').read_text(encoding='utf-8'))
        elif args.command == 'fetch':
            from .datasets import fetch_reference
            manifest = fetch_reference(args.repo, episode=args.episode, include_media=args.media,
                                       camera=args.camera, max_mb=args.max_mb, revision=args.revision)
            write_json(ROOT/'artifacts/latest_public_sample.json', {'manifest': str(manifest.relative_to(ROOT))})
            print(f'Saved: {manifest}')
            print('This is reference data, not a trained or challenge-compatible robot policy.')
        elif args.command == 'visualize':
            from .visualize import visualize_sample
            manifest = args.manifest
            if manifest is None:
                pointer = json.loads((ROOT/'artifacts/latest_public_sample.json').read_text(encoding='utf-8'))
                manifest = ROOT / pointer['manifest']
            print(json.dumps(visualize_sample(manifest), indent=2))
        elif args.command == 'demo':
            from .fixtures import run_fixture_demo
            report = run_fixture_demo()
            write_json(ROOT/'artifacts/logic_fixture_results.json', report)
            print(json.dumps(report, indent=2))
        elif args.command == 'selftest':
            suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            report = {'kind':'OFFLINE_UNIT_TESTS_NOT_ROBOT_EVALUATION','tests_run':result.testsRun,
                      'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
                      'passed': result.wasSuccessful()}
            write_json(ROOT/'artifacts/unit_test_summary.json', report)
            return 0 if result.wasSuccessful() else 1
        return 0
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        print('No substitute data, inferred result, or robot success has been generated.', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
