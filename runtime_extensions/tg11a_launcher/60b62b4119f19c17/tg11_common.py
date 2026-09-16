"""Bounded, evidence-preserving helpers for the TableGuard 11 training diagnostic."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import os
import random
import numpy as np

REVISION = '11-cup-multibatch-diagnostic-v1'


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError(f'Unexpectedly large JSON: {path}')
    def reject(value):
        raise ValueError(f'Nonfinite JSON literal: {value}')
    result = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def hashes(folder):
    folder = Path(folder)
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Unexpected symlink in protected directory: {path}')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha(path)
    if not result:
        raise ValueError(f'Empty protected directory: {folder}')
    return result


def verify_hashes(folder, original):
    if hashes(folder) != original:
        raise RuntimeError(f'Protected files changed: {folder}')
    return True


def safe_name(value):
    if (not isinstance(value, str) or not value or value in ('.', '..')
            or any(c in value for c in '/\\:')):
        raise ValueError('Expected one folder name, not a path')
    return value


class Lease:
    """OS-held lease; a persistent .lck file alone does not mean active execution."""
    def __init__(self, path):
        self.path = Path(path)
        self.f = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open('a+b')
        self.f.seek(0, 2)
        if self.f.tell() == 0:
            self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.f.close(); self.f = None
            raise RuntimeError(f'Another worker holds {self.path}; no duplicate started.') from exc
        return self
    def __exit__(self, *args):
        if self.f is not None:
            self.f.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            finally:
                self.f.close(); self.f = None
        return False


def shuffled_samples(n, count, seed):
    """Visit every observation before reuse; no fabricated extra episodes."""
    if n < 1 or count < 1:
        raise ValueError('Positive observation and sample counts required')
    rng = random.Random(seed)
    selected = []
    while len(selected) < count:
        epoch = list(range(n)); rng.shuffle(epoch)
        selected.extend(epoch)
    return selected[:count]


def probe_indices(n, maximum=8):
    if n < 2:
        raise ValueError('At least two observations required')
    return sorted(set(np.rint(np.linspace(0, n - 1, min(n, maximum))).astype(int).tolist()))


def action_metrics(reference, prediction, masks, mean, scale, order, lower, upper):
    """Offline training-episode imitation errors, excluding padded targets."""
    a, b = np.asarray(reference, float), np.asarray(prediction, float)
    mask = np.asarray(masks, bool)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (20, 12):
        raise ValueError('Expected matching [cases,20,12] action arrays')
    if mask.shape != a.shape[:2] or not (~mask).any():
        raise ValueError('Invalid padding mask')
    m, s, lo, hi = [np.asarray(x, float) for x in (mean, scale, lower, upper)]
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in (m, s, lo, hi)):
        raise ValueError('Invalid normalizer or joint limits')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(s <= 0) or np.any(lo >= hi):
        raise ValueError('Invalid actions, scales, or bounds')
    if len(order) != 12 or len(set(order)) != 12:
        raise ValueError('Invalid native joint order')
    right = [i for i, name in enumerate(order) if name.startswith('right_')]
    left = [i for i, name in enumerate(order) if name.startswith('left_')]
    if len(right) != 6 or len(left) != 6:
        raise ValueError('Expected six actuators per arm')
    error = (b - a)[~mask]
    error_rad = error * s
    raw = (b * s + m)[~mask]
    return {
        'kind': 'TRAINING_EPISODE_OFFLINE_ACTION_ERROR_NOT_TASK_SCORE',
        'valid_time_steps_across_cases': int((~mask).sum()),
        'normalized_rmse': float(np.sqrt(np.mean(error**2))),
        'joint_mae_rad': float(np.mean(np.abs(error_rad))),
        'right_arm_mae_rad': float(np.mean(np.abs(error_rad[:, right]))),
        'left_arm_mae_rad': float(np.mean(np.abs(error_rad[:, left]))),
        'per_joint_mae_rad': np.mean(np.abs(error_rad), axis=0).tolist(),
        'maximum_joint_error_rad': float(np.abs(error_rad).max()),
        'predicted_values_outside_joint_bounds': int(((raw < lo) | (raw > hi)).sum()),
        'predictions_clipped': False,
        'held_out_evaluation': False, 'task_success': None,
    }
