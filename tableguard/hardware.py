from __future__ import annotations
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(args: list[str], timeout: int = 12) -> dict[str, Any]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def inspect_hardware(probe_frameworks: bool = False) -> dict[str, Any]:
    """No uploads, installation, model downloads or credentials collection."""
    result: dict[str, Any] = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "operating_system": platform.system(),
        "os_release": platform.release(), "architecture": platform.machine(),
        "cpu_model": platform.processor() or "unresolved",
        "logical_cpu_count": os.cpu_count(),
        "eligibility": "NOT CERTIFIED: check exact CPU model against organizer rules",
        "phase1_discrete_gpu_required": False,
    }
    if platform.system() == "Linux":
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.lower().startswith('model name'):
                    result['cpu_model'] = line.split(':', 1)[1].strip()
                    break
            values = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, val = line.split(':', 1)
                values[key] = int(val.strip().split()[0]) * 1024
            result['ram_total_gib'] = round(values['MemTotal'] / 2**30, 2)
        except (OSError, KeyError, ValueError):
            pass
        if shutil.which('lspci'):
            out = _run(['lspci'])
            result['graphics_devices'] = [s for s in out.get('stdout','').splitlines()
                                          if any(k in s.lower() for k in ['vga', '3d controller', 'display controller'])]
    elif platform.system() == 'Windows':
        ps = shutil.which('powershell') or shutil.which('pwsh')
        if ps:
            commands = {
                'cpu_details': 'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress',
                'ram_details': 'Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress',
                'graphics_devices': 'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress',
            }
            for key, script in commands.items():
                out = _run([ps, '-NoProfile', '-Command', script])
                try:
                    result[key] = json.loads(out.get('stdout', ''))
                except json.JSONDecodeError:
                    result[key] = out
            cpu = result.get('cpu_details')
            if isinstance(cpu, dict) and cpu.get('Name'):
                result['cpu_model'] = cpu['Name']
            mem = result.get('ram_details')
            if isinstance(mem, dict) and mem.get('TotalPhysicalMemory'):
                result['ram_total_gib'] = round(int(mem['TotalPhysicalMemory']) / 2**30, 2)
    elif platform.system() == 'Darwin':
        out = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if out.get('returncode') == 0:
            result['cpu_model'] = out['stdout']
        out = _run(['sysctl', '-n', 'hw.memsize'])
        try:
            result['ram_total_gib'] = round(int(out['stdout']) / 2**30, 2)
        except (KeyError, ValueError):
            pass
    try:
        import psutil
        result['ram_total_gib'] = round(psutil.virtual_memory().total / 2**30, 2)
        result['ram_available_gib'] = round(psutil.virtual_memory().available / 2**30, 2)
    except ImportError:
        pass
    result['project_disk_free_gib'] = round(shutil.disk_usage(Path.cwd()).free / 2**30, 2)
    if shutil.which('nvidia-smi'):
        result['nvidia'] = _run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])
        result['nvidia_memory_unit'] = 'MiB as returned by nvidia-smi'
    else:
        result['nvidia'] = {'status': 'nvidia-smi not found; this does NOT prove no GPU exists'}
    packages = {}
    for name in ['numpy','matplotlib','opencv-python-headless','opencv-python','psutil',
                 'huggingface_hub','pyarrow','torch','openvino','mujoco','lerobot']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result['packages'] = packages
    if probe_frameworks:
        # Separate subprocesses prevent a framework/driver import failure killing this report.
        probes = {
            'openvino_probe': "import openvino as ov; print(ov.Core().available_devices)",
            'torch_probe': "import torch; print({'cuda': torch.cuda.is_available(), 'xpu': hasattr(torch,'xpu') and torch.xpu.is_available()})",
        }
        for key, code in probes.items():
            result[key] = _run([sys.executable, '-c', code], timeout=25)
        result['probe_note'] = 'Device discovery only; no model inference or latency measurement.'
    else:
        result['probe_note'] = 'Framework imports skipped. Use --probe-frameworks after approved installation.'
    return result
