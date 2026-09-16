"""TableGuard recorded episode -> lossless, dual-rate SmolVLA training bridge.

No simulator import, no new demonstrations, no action resampling. This is a
custom PyTorch-compatible chunk pack, NOT a standard LeRobotDataset-v3 export.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
from PIL import Image

JOINTS = tuple(f'{side}_{joint}' for side in ('left', 'right') for joint in
               ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'))
CAMERAS = ('top', 'right_wrist_cam')
REQUIRED_FLAGS = ('completed', 'grasp_test_passed', 'transfer_test_passed', 'grasp_verified',
                  'lift_verified', 'transfer_verified', 'placement_verified', 'release_verified')


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Unexpectedly large JSON: ' + str(path))
    def bad(value):
        raise ValueError('Non-finite JSON literal: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=bad)


def write_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def inside(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError('Missing file reference')
    root = Path(root).resolve()
    p = Path(value)
    p = (p if p.is_absolute() else root / p).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Reference outside selected episode: ' + str(p))
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def vector(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite {shape}, got {a.shape}')
    return a


def validate_episode(run):
    run = Path(run).resolve()
    report_path = run / 'transfer_report.json'
    report = read_json(report_path)
    if any(report.get(k) is not True for k in REQUIRED_FLAGS):
        raise ValueError('Selected episode is not a completely passed cup-transfer trial')
    if report.get('stops') or report.get('graphics_error') or report.get('source_scene_modified'):
        raise ValueError('Stopped, recording-error, or changed-source episode rejected')
    manifest = inside(run, report.get('observation_manifest') or 'expert_observations.jsonl')
    trace = inside(run, report.get('trace_file') or 'grasp_trace.csv')
    if manifest.stat().st_size > 64*1024*1024 or trace.stat().st_size > 128*1024*1024:
        raise ValueError('Episode exceeds the pilot inspection limits')
    protected = {str(p): sha(p) for p in (report_path, manifest, trace)}
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not 2 <= len(records) <= 10000:
        raise ValueError('Expected between 2 and 10000 recorded observations')
    if len(records) != report.get('recorded_observation_action_pairs'):
        raise ValueError('Manifest count does not match report')
    with trace.open(newline='', encoding='utf-8-sig') as f:
        trace_rows = list(csv.DictReader(f))
    if not trace_rows or len(trace_rows) != report.get('physics_steps'):
        raise ValueError('Execution trace count does not match physics_steps')
    mapping = report.get('native_actuator_mapping')
    if not isinstance(mapping, list) or len(mapping) != 12:
        raise ValueError('Missing inspected 12-joint actuator mapping')
    order = tuple(m.get('name') for m in mapping)
    if len(set(order)) != 12 or set(order) != set(JOINTS):
        raise ValueError('Unexpected joint names or duplicates in actuator mapping')
    lo = vector([m['low'] for m in mapping], (12,), 'lower limits')
    hi = vector([m['high'] for m in mapping], (12,), 'upper limits')
    if np.any(lo >= hi):
        raise ValueError('Invalid actuator bounds')
    dt = float(report['timestep_s'])
    if not math.isfinite(dt) or not math.isclose(dt, .005, abs_tol=1e-10):
        raise ValueError('Pilot expects the recorded 0.005 s native action step; no implicit resampling')
    horizon = 20
    stride = dt*horizon
    trace_commands = vector([[float(row[j+'_command_rad']) for j in order] for row in trace_rows],
                            (len(trace_rows), 12), 'trace commands')
    trace_states = vector([[float(row[j+'_position_rad']) for j in order] for row in trace_rows],
                          (len(trace_rows), 12), 'trace states')
    trace_times = vector([float(row['time_s']) for row in trace_rows], (len(trace_rows),), 'trace times')
    if not np.allclose(np.diff(trace_times), dt, atol=1e-8, rtol=0):
        raise ValueError('Discontinuous native execution timestamps')
    if [int(row['step']) for row in trace_rows] != list(range(1, len(trace_rows)+1)):
        raise ValueError('Trace steps are not contiguous one-based post-action steps')
    states, actions, masks, image_refs, timestamps, steps = [], [], [], [], [], []
    tasks = set(); sizes = {}; max_state_error = max_action_error = 0.0
    total_native = 0
    for n, rec in enumerate(records):
        if rec.get('frame_id') != n or tuple(rec.get('action_order', ())) != order:
            raise ValueError(f'Frame {n}: ID or joint ordering mismatch')
        if rec.get('episode_outcome') != 'passed_engineering_trial':
            raise ValueError(f'Frame {n}: not a passed demonstration record')
        semantics = rec.get('action_semantics', '')
        if 'absolute joint position targets' not in semantics or 'radians' not in semantics:
            raise ValueError('Unexpected action semantics; do not reinterpret units')
        task = rec.get('instruction')
        if not isinstance(task, str) or not task.strip() or len(task) > 4096:
            raise ValueError('Missing or invalid recorded instruction')
        tasks.add(task)
        step = rec.get('step')
        if not isinstance(step, int) or step != n*horizon:
            raise ValueError(f'Frame {n}: expected native step {n*horizon}, got {step}')
        t = float(rec['time_s'])
        if not math.isfinite(t) or not math.isclose(t, step*dt, abs_tol=1e-8):
            raise ValueError('Observation timestamp is not pre-action time')
        st = vector(rec['state'], (12,), 'state')
        action = vector(rec['action'], (12,), 'first action')
        valid = rec.get('action_chunk_valid_length')
        expected_valid = min(horizon, len(trace_rows)-step)
        if not isinstance(valid, int) or valid != expected_valid or valid < 1:
            raise ValueError('Invalid action chunk length or an unrecorded action gap')
        if n < len(records)-1 and valid != horizon:
            raise ValueError('Only the final action chunk may be partial')
        if not math.isclose(float(rec['action_chunk_dt_s']), dt, abs_tol=1e-10):
            raise ValueError('Action chunk time base mismatch')
        chunk = vector(rec['action_chunk'], (valid,12), 'action chunk')
        if not np.allclose(chunk[0], action, atol=1e-9, rtol=0):
            raise ValueError('First action differs from action_chunk[0]')
        error = float(np.max(np.abs(chunk-trace_commands[step:step+valid])))
        max_action_error=max(max_action_error, error)
        if error > 1e-8:
            raise ValueError('Action labels do not match commands actually applied in the trace')
        if step > 0:
            error=float(np.max(np.abs(st-trace_states[step-1])))
            max_state_error=max(max_state_error,error)
            if error > 1e-7:
                raise ValueError('State labels are not aligned before the recorded action')
        if np.any(chunk < lo-1e-8) or np.any(chunk > hi+1e-8):
            raise ValueError('Expert action outside reported joint limits')
        paths={}
        for cam in CAMERAS:
            p=inside(run, (rec.get('images') or {}).get(cam))
            if p.stat().st_size > 16*1024*1024:
                raise ValueError('Unexpectedly large camera image')
            protected[str(p)]=sha(p)
            with Image.open(p) as im:
                if im.format not in ('JPEG','PNG') or im.mode!='RGB':
                    raise ValueError('Expected recorded RGB PNG/JPEG camera image')
                size=im.size; im.verify()
            if cam in sizes and sizes[cam]!=size:
                raise ValueError('Camera image dimensions change within the episode')
            sizes[cam]=size; paths[cam]=str(p)
        padded=np.repeat(chunk[-1][None,:],horizon,axis=0);padded[:valid]=chunk
        mask=np.arange(horizon)>=valid
        states.append(st); actions.append(padded);masks.append(mask)
        image_refs.append(paths);timestamps.append(t);steps.append(step);total_native+=valid
    if total_native != len(trace_rows):
        raise ValueError('The observation action-chunks do not cover the complete native trace')
    if len(tasks)!=1:
        raise ValueError('Pilot requires the original single-task episode; no relabelling')
    stat_s=np.asarray(states);stat_a=trace_commands
    def stats(a):
        raw=a.std(axis=0);scale=np.maximum(raw,.01)
        return dict(mean=a.mean(axis=0).tolist(),std=scale.tolist(),raw_std=raw.tolist(),
                    minimum=a.min(axis=0).tolist(),maximum=a.max(axis=0).tolist(),
                    std_floor_rad=.01,constant_or_low_variation_dimensions=np.where(raw<.01)[0].tolist())
    summary=dict(kind='ONE_CUP_EPISODE_DUAL_RATE_BRIDGE_NOT_TASK_EVALUATION',source_run=str(run),
        source_report=str(report_path),source_manifest=str(manifest),episodes=1,observations=len(records),
        referenced_images=len(records)*len(CAMERAS),native_applied_actions=total_native,
        cameras=list(CAMERAS),image_sizes_wh={k:list(v) for k,v in sizes.items()},state_dim=12,action_dim=12,
        action_order=list(order),state_order=list(order),state_order_basis='recorder code uses the same native_actuator_mapping for state and action',
        action_representation='absolute joint position targets',action_units='radians',
        observation_interval_s=stride,native_action_dt_s=dt,action_chunk_length=horizon,
        max_action_trace_error_rad=max_action_error,max_preaction_state_trace_error_rad=max_state_error,
        original_instruction=next(iter(tasks)),joint_lower_rad=lo.tolist(),joint_upper_rad=hi.tolist(),
        standard_lerobot_v3_dataset=False,bridge_format='TableGuard action-chunk NPZ+JSONL for direct LeRobot SmolVLA API',
        evaluation_split_created=False,normalization_source='this training-pilot episode only; not held-out evaluation',
        simulated_execution_launched=False,source_hashes=protected)
    return summary,dict(states_rad=np.asarray(states,dtype=np.float32),actions_rad=np.asarray(actions,dtype=np.float32),
        action_is_pad=np.asarray(masks,dtype=np.bool_),observation_time_s=np.asarray(timestamps,dtype=np.float64),
        native_step=np.asarray(steps,dtype=np.int64)), image_refs, {'observation.state':stats(stat_s),'action':stats(stat_a)}


def write_pack(out, summary, arrays, refs, statistics):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'chunks.npz',**arrays)
    write_json(out/'contract.json',summary)
    write_json(out/'normalization.json',statistics)
    with (out/'index.jsonl').open('w',encoding='utf-8') as f:
        for i,p in enumerate(refs):
            f.write(json.dumps(dict(index=i,images=p,instruction=summary['original_instruction']),allow_nan=False)+'\n')
    changed=[p for p,h in summary['source_hashes'].items() if sha(p)!=h]
    if changed:raise RuntimeError('Source changed during read-only preparation: '+str(changed[:3]))
    (out/'README.md').write_text(
        '# TableGuard cup pilot\n\nOne scripted cup-transfer episode; not a two-arm trained policy.\n'
        'A native-rate chunk bridge, not a standard LeRobotDataset-v3 export.\n'
        'Camera observations: 10 Hz. Applied joint commands: 200 Hz. Each observation labels the '
        'following 20 actual commands; final padding is masked, never counted as data.\n'
        'Camera paths refer to the preserved original episode. Do not remove that folder.\n'
        'No random frame train/test split was created; this is only a training-pipeline smoke test.\n',encoding='utf-8')
    return out


class ChunkDataset:
    """Map-style PyTorch-compatible dataset. Standard-library/NumPy/Pillow only.

    PyTorch DataLoader can batch this object. All observations are pre-action.
    The padded action mask has True only outside the recorded episode.
    """
    def __init__(self, pack):
        self.pack=Path(pack);self.contract=read_json(self.pack/'contract.json')
        with np.load(self.pack/'chunks.npz',allow_pickle=False) as z:
            self.arrays={k:z[k] for k in z.files}
        self.index=[json.loads(x) for x in (self.pack/'index.jsonl').read_text().splitlines() if x.strip()]
        self.stats=read_json(self.pack/'normalization.json')
        if len(self.index)!=len(self.arrays['states_rad']):raise ValueError('Pack index mismatch')
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        if not 0 <= i < len(self):raise IndexError(i)
        row=self.index[i]
        out={}
        for cam in self.contract['cameras']:
            p=Path(row['images'][cam])
            if sha(p)!=self.contract['source_hashes'].get(str(p)):
                raise ValueError('Camera source changed after preparation: '+str(p))
            with Image.open(p) as im:
                out['observation.images.'+cam]=np.array(im,dtype=np.float32).transpose(2,0,1)/255.
        for key,arr in [('observation.state','states_rad'),('action','actions_rad')]:
            st=self.stats[key]
            out[key]=((self.arrays[arr][i]-np.asarray(st['mean'],dtype=np.float32))/np.asarray(st['std'],dtype=np.float32)).astype(np.float32)
        out['action_is_pad']=self.arrays['action_is_pad'][i].copy()
        out['task']=row['instruction']
        return out
