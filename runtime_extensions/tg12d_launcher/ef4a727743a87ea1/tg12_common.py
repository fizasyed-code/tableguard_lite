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

REVISION = '12d-windows-safe-camera-recording-v1'
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
    """Accept original completed 11, or a separately verified saved checkpoint.
    Never overwrite the failed training report or accept an unverified save.
    """
    root=Path(root).resolve();parent=root/'artifacts'/'smolvla_train_11'
    pointer=parent/'selected_training.json'
    if not pointer.is_file():
        return None,{'status':'waiting_for_training_11','reason':'No selected training run.'}
    selected=read_json(pointer)
    if selected.get('source_pilot')!=SOURCE_PILOT:raise ValueError('Unexpected source pilot.')
    run=parent/safe_name(selected.get('run'));p=run/'training_report.json'
    r=read_json(p) if p.is_file() else {}
    verification=None
    if r.get('status')!='training_run_completed':
        expected='RuntimeError: Reloaded checkpoint predictions differ beyond declared tolerance'
        if r.get('status')!='failed' or r.get('error')!=expected or r.get('integrity_error'):
            return None,{'status':'waiting_for_training_11','training_status':r.get('status'),
                         'reason':'Training is incomplete or has a different failure; no rollout.'}
        vparent=root/'artifacts'/'checkpoint_verify_11a'
        vp=vparent/'selected_verification.json'
        if not vp.is_file():
            return None,{'status':'waiting_for_checkpoint_verification_11a',
                         'reason':'Run 11A; no training or robot started here.'}
        choice=read_json(vp)
        if choice.get('training_run')!=run.name:raise ValueError('Verification belongs to another training run.')
        vfile=vparent/safe_name(choice.get('run'))/'verification_report.json'
        v=read_json(vfile) if vfile.is_file() else {}
        if v.get('status')!='checkpoint_reload_verified':
            return None,{'status':'waiting_for_checkpoint_verification_11a',
                         'verification_status':v.get('status'),'error':v.get('error'),
                         'reason':'No successful reload verification, so no rollout.'}
        for k in ('checkpoint_reloaded_verified','original_prediction_parity_passed',
                  'repeated_fresh_load_parity_passed','original_evidence_unchanged',
                  'source_checkpoint_unchanged','source_data_pack_unchanged','source_recording_unchanged'):
            if v.get(k) is not True:raise ValueError('Verification missing: '+k)
        if v.get('error') or v.get('integrity_error') or v.get('additional_optimizer_updates')!=0:
            raise ValueError('Invalid recovery outcome.')
        if v.get('training_report_sha256')!=sha(p):raise ValueError('Original training report changed after verification.')
        if v.get('checkpoint_hashes_file_sha256')!=sha(run/'final_checkpoint_hashes.json'):
            raise ValueError('Checkpoint identity changed after verification.')
        if Path(v.get('training_run','')).resolve()!=run.resolve():raise ValueError('Training path mismatch.')
        if Path(v.get('checkpoint_path','')).resolve()!=(run/'checkpoint_final').resolve():raise ValueError('Checkpoint path mismatch.')
        if v.get('loader_id')!='initialize_then_fp32_then_strict_safetensors':raise ValueError('Unknown recovery loader.')
        if v.get('loader_sha256')!=sha(Path(__file__).with_name('tg_fp32_loader.py')):raise ValueError('Loader differs from verified version.')
        for key in ('load_1_vs_original','load_2_vs_original','fresh_loads_comparison'):
            test=v.get(key) or {}
            if test.get('passed') is not True or test.get('atol')!=1e-6 or test.get('rtol')!=1e-5:
                raise ValueError('Original reload tolerances were not preserved: '+key)
        verification={'path':str(vfile),'sha256':sha(vfile),'method':v['loader_id']}
    elif r.get('error') or r.get('integrity_error'):
        raise ValueError('Completed training report has an error.')
    for key in ('checkpoint_saved','source_checkpoint_unchanged','source_data_pack_unchanged','source_recording_unchanged'):
        if r.get(key) is not True:raise ValueError('Training prerequisite not verified: '+key)
    if verification is None and r.get('checkpoint_reloaded') is not True:
        raise ValueError('Checkpoint was not reloaded successfully.')
    if r.get('updates_completed')!=r.get('updates_requested') or int(r.get('updates_completed',0))<2:
        raise ValueError('Not all requested training updates completed.')
    checkpoint=(run/'checkpoint_final').resolve()
    if Path(r.get('checkpoint_path','')).resolve()!=checkpoint:raise ValueError('Checkpoint path mismatch.')
    for name in ('config.json','contract.json','normalization.json','training_state.json'):
        if not (checkpoint/name).is_file():raise FileNotFoundError(checkpoint/name)
    ts=read_json(checkpoint/'training_state.json')
    if ts.get('new_updates')!=r['updates_completed']:raise ValueError('Checkpoint update count mismatch.')
    return {'training_run':str(run),'checkpoint':str(checkpoint),
            'training_report':r,'training_report_sha256':sha(p),
            'expected_hashes_file':str(run/'final_checkpoint_hashes.json'),
            'reload_verification':verification},None
