"""TableGuard 12: bounded local protocol, evidence, and source integrity.
No model imports, simulator, installation, or process launch on import.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from pathlib import Path
import hashlib
import json
import os
import time
import uuid

REVISION = '16-multiepisode-2201-update-cup-rollout-v1'
SOURCE_PILOT = '20260914T102946Z_569208'
SOURCE_CUP_RUN = '20260913T025459Z_3185ba'
JOINTS = tuple(f'{s}_{j}' for s in ('left', 'right') for j in
    ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'))
OBS_KEYS = {'image_top', 'image_right_wrist', 'state'}


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('JSON too large: ' + str(path))
    def reject(x):
        raise ValueError('Nonfinite JSON: ' + x)
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or value in ('.','..') or any(c in value for c in '/\\:'):
        raise ValueError('Expected a single folder/file name')
    return value


def inside(root, value, exists=True):
    root = Path(root).resolve()
    if not isinstance(value, str) or not value:
        raise ValueError('Missing local path')
    p = Path(value)
    p = (p if p.is_absolute() else root/p).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Path escapes allowed directory: ' + str(p))
    if exists and not p.is_file():
        raise FileNotFoundError(p)
    return p


class Lease(AbstractContextManager):
    """Same byte-0 lock protocol as 10/11. File presence is not liveness."""
    def __init__(self, path): self.path=Path(path); self.f=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f=self.path.open('a+b'); self.f.seek(0,2)
        if self.f.tell()==0: self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.f.close(); self.f=None
            raise RuntimeError('Another process holds ' + str(self.path) + '; no lock removed.') from exc
        return self
    def __exit__(self,*args):
        if self.f is not None:
            try:
                self.f.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            finally: self.f.close(); self.f=None
        return False


def event(out, worker, stage, **details):
    value={'worker':worker,'stage':stage,'unix_time':time.time(),**details}
    print('TG12 '+json.dumps(value,allow_nan=False),flush=True)
    write_json(Path(out)/(worker+'_progress.json'),value)
    with (Path(out)/(worker+'_events.jsonl')).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n')


def inference_hashes(checkpoint):
    """Hash all inference files. Optimizer state is never loaded or needed."""
    checkpoint=Path(checkpoint)
    values={}
    for p in sorted(checkpoint.rglob('*')):
        if p.is_symlink(): raise ValueError('Checkpoint links are not accepted')
        if p.is_file() and p.name!='optimizer_state.pt':
            values[p.relative_to(checkpoint).as_posix()]=sha(p)
    if 'config.json' not in values or not any(n.endswith('.safetensors') for n in values):
        raise ValueError('Complete safetensors checkpoint not found')
    return values


def verify_inference_hashes(checkpoint, before):
    if inference_hashes(checkpoint)!=before: raise RuntimeError('Source inference checkpoint changed')
    return True


def find_completed_training(root):
    """Use the latest successful Notebook 15A multi-episode checkpoint."""
    root=Path(root).resolve()
    parent=root/'artifacts'/'smolvla_train_15a'
    if not parent.is_dir():
        return None,{'status':'waiting_for_training_15a','reason':'No Notebook 15A training folder found.'}
    runs=sorted([p for p in parent.iterdir() if p.is_dir() and (p/'training_report.json').is_file()],
                key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:
        return None,{'status':'waiting_for_training_15a','reason':'No completed Notebook 15A run found.'}
    run=runs[0]
    p=run/'training_report.json'
    r=read_json(p)
    if r.get('status')!='training_run_completed' or r.get('error'):
        return None,{'status':'waiting_for_training_15a','training_status':r.get('status'),
                     'error':r.get('error'),'reason':'Latest Notebook 15A run is not complete.'}
    for key in ('checkpoint_saved','checkpoint_reloaded'):
        if r.get(key) is not True:
            raise ValueError('Notebook 15A prerequisite not verified: '+key)
    if int(r.get('episodes',0)) < 5:
        raise ValueError('Expected at least five successful training episodes.')
    if r.get('updates_completed')!=r.get('updates_requested'):
        raise ValueError('Notebook 15A did not complete all requested updates.')
    checkpoint=(run/'checkpoint_final').resolve()
    if Path(r.get('checkpoint_path','')).resolve()!=checkpoint:
        raise ValueError('Notebook 15A checkpoint path mismatch.')
    for name in ('config.json','contract.json','normalization.json','training_state.json','model.safetensors'):
        if not (checkpoint/name).is_file():
            raise FileNotFoundError(checkpoint/name)
    ts=read_json(checkpoint/'training_state.json')
    if int(ts.get('total_updates',-1))!=int(r.get('checkpoint_total_optimizer_updates',-2)):
        raise ValueError('Notebook 15A total update count mismatch.')
    return {'training_run':str(run),'checkpoint':str(checkpoint),
            'training_report':r,'training_report_sha256':sha(p),
            'expected_hashes_file':None,'reload_verification':None},None
