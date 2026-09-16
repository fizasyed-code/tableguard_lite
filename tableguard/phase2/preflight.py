from __future__ import annotations
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any
from ..hardware import inspect_hardware
from .scene import utc_now


def inspect(probe_devices: bool = False) -> dict[str, Any]:
    deps = {}
    for package in ['mujoco', 'numpy', 'Pillow', 'openvino']:
        try:
            deps[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            deps[package] = None
    missing = [p for p in ['mujoco', 'numpy', 'Pillow'] if deps[p] is None]
    return {'kind': 'CURRENT_MACHINE_PREFLIGHT_NOT_TASK_EXECUTION', 'collected_at_utc': utc_now(),
            'python_executable': sys.executable, 'python_311_or_later': sys.version_info >= (3, 11),
            'dependencies': deps, 'missing_scene_dependencies': missing,
            'scene_dependencies_present': not missing and sys.version_info >= (3, 11),
            'hardware': inspect_hardware(probe_devices),
            'rendering_tested': False, 'policy_inference_tested': False, 'robot_task_completed': None,
            'official_eligibility': 'NOT_CERTIFIED',
            'notes': ['No drivers or packages were installed by this command.',
                      'OpenVINO is reported separately; discovery is not a test of application inference.',
                      'The host OS and CPU in this report are yours; no old build-environment report is reused.']}
