"""One explicitly selected checkpoint verification; no installation or training."""
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from tg11_common import Lease, read_json, write_json, sha

SOURCE_PILOT='20260914T102946Z_569208'
TRAINING_RUN='20260914T175053Z_f5dedc'


def show_result(run):
    p=run/'verification_report.json'
    r=read_json(p) if p.is_file() else {}
    print('\nTABLEGUARD 11A — FINAL SUMMARY',flush=True)
    print('Run:',run)
    for k in ('status','stage','error','additional_optimizer_updates','checkpoint_reloaded_verified',
              'original_prediction_parity_passed','repeated_fresh_load_parity_passed',
              'original_evidence_unchanged','source_checkpoint_unchanged','source_data_pack_unchanged',
              'source_recording_unchanged','verified_in_sample_right_arm_mae_rad','task_success'):
        print(k+':',r.get(k))
    for key in ('load_1_vs_original','load_2_vs_original','fresh_loads_comparison'):
        print(key+':',r.get(key))
    audit=r.get('load_1') or {}
    print('INITIAL DTYPES:',audit.get('initialized_tensor_dtypes'))
    print('TENSORS AFFECTED BY LOW-PRECISION ROUNDTRIP:',audit.get('lower_precision_roundtrip_affected_tensors'))
    print('LOADED TENSORS EXACTLY MATCH SAVED:',audit.get('loaded_tensors_equal_saved'))
    if r.get('status')=='checkpoint_reload_verified':
        print('NEXT: run TableGuard_12A_Verified_Checkpoint_Rollout.ipynb once; do not rerun training.')
    else:
        print('Rollout remains blocked. Keep this report and verification_traceback.txt.')
    print('Original failed training report stays unchanged; this is a separate verification record.')
    return r


def child_env(root,prefix):
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','CONDA_PROMPT_MODIFIER'):env.pop(k,None)
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(root/'models'/'hf_cache'),
        HF_HUB_CACHE=str(root/'models'/'hf_cache'/'hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    old=str(Path(sys.prefix)).lower()
    paths=[p for p in env.get('PATH','').split(os.pathsep) if p and not p.lower().startswith(old)]
    env['PATH']=os.pathsep.join(map(str,[prefix,prefix/'Scripts',prefix/'Library'/'bin',prefix/'bin']))+os.pathsep+os.pathsep.join(paths)
    env['CONDA_PREFIX']=str(prefix)
    return env


def run(root,sources):
    root=Path(root).resolve()
    if not (root/'tableguard').is_dir():raise FileNotFoundError('Existing project not found: '+str(root))
    art=root/'artifacts';parent=art/'checkpoint_verify_11a';parent.mkdir(exist_ok=True)
    trainparent=art/'smolvla_train_11';train=trainparent/TRAINING_RUN;source=art/'smolvla_09'/SOURCE_PILOT
    selected=read_json(trainparent/'selected_training.json')
    if selected!={'run':TRAINING_RUN,'source_pilot':SOURCE_PILOT}:
        raise RuntimeError('Selected training run changed. No automatic recovery of an unknown run.')
    prefix=Path.home()/'.tableguard'/'envs'/'vla312'
    python=prefix/('python.exe' if os.name=='nt' else 'bin/python')
    shared=art/'smolvla_openvino_10'
    with ExitStack() as stack:
        stack.enter_context(Lease(parent/'launcher.lck'))
        pointer=parent/'selected_verification.json'
        if pointer.is_file():
            previous=read_json(pointer)
            if previous.get('training_run')!=TRAINING_RUN:raise ValueError('Verification pointer uses another training run.')
            name=previous.get('run')
            if not isinstance(name,str) or name in ('','.','..') or any(x in name for x in '/\\:'):raise ValueError('Invalid pointer')
            print('Reading saved verification; no model load or new training.',flush=True)
            return show_result(parent/name)
        if any(p.is_dir() for p in parent.iterdir()):raise RuntimeError('Unselected verification folder exists; keep evidence.')
        for k in ('active_pilot.lock',):
            if (source.parent/k).exists():raise RuntimeError('Original pilot marker exists. Do not remove active locks.')
        for p in art.glob('*/active_run.lock'):raise RuntimeError('Robot run marker exists: '+str(p))
        stack.enter_context(Lease(trainparent/'launcher.lck'))
        stack.enter_context(Lease(shared/'launcher.lck'))
        with Lease(trainparent/'worker.lck'), Lease(shared/'worker.lck'), Lease(parent/'worker.lck'):pass
        if not python.is_file():raise FileNotFoundError('Existing vla312 environment missing: '+str(python))
        if not (train/'checkpoint_final'/'model.safetensors').is_file():raise FileNotFoundError('Final trained weights missing')
        out=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]);out.mkdir()
        for name,text in sources.items():
            compile(text,name,'exec');(out/name).write_text(text,encoding='utf-8',newline='\n')
        write_json(out/'settings.json',dict(output=str(out),training_run=str(train),source_pilot=str(source),
            worker_leases=[str(parent/'worker.lck'),str(trainparent/'worker.lck'),str(shared/'worker.lck')]))
        write_json(pointer,dict(run=out.name,training_run=TRAINING_RUN))
        print('NEW 11A VERIFICATION:',out,flush=True)
        print('No training, installation, checkpoint overwrite or robot. Original tolerance retained.',flush=True)
        log=out/'verify.log';status=out/'verify.log.status.json';p=None;start=time.monotonic();last=0
        state=dict(status='starting',parent_pid=os.getpid(),started_unix=time.time())
        write_json(status,state); print('Log:',log,flush=True)
        try:
            with log.open('x',encoding='utf-8') as writer:
                p=subprocess.Popen([str(python),'-u','-X','faulthandler',str(out/'tg11a_verify.py'),str(out/'settings.json')],
                    cwd=out,env=child_env(root,prefix),stdout=writer,stderr=subprocess.STDOUT)
                state.update(status='running',pid=p.pid);write_json(status,state)
                with log.open(encoding='utf-8',errors='replace') as reader:
                    while True:
                        for line in reader.readlines():
                            if line.startswith('TG11A '):print(line.rstrip(),flush=True)
                        elapsed=time.monotonic()-start
                        if p.poll() is not None:break
                        if elapsed-last>=20:
                            print('11A worker active | elapsed %.0f seconds'%elapsed,flush=True);last=elapsed
                            state['elapsed_s']=elapsed;write_json(status,state)
                        if elapsed>900:raise TimeoutError('Verification exceeded 15 minutes; no automatic retry.')
                        time.sleep(.2)
            if p.returncode!=0:
                print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-18:]),flush=True)
                raise RuntimeError('Checkpoint verification failed; preserve '+str(log))
            state['status']='finished'
            result=show_result(out)
            if result.get('status')!='checkpoint_reload_verified':raise RuntimeError('Worker returned without verified checkpoint.')
            return result
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
            (out/'launcher_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            show_result(out)
            raise
        finally:
            if p is not None and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            state.update(returncode=p.returncode if p else None,elapsed_s=time.monotonic()-start)
            write_json(status,state)
