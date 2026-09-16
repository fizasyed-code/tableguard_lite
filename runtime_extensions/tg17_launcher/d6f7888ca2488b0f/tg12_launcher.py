"""TableGuard 12 one-run coordinator. No dependency on previous kernel globals."""
from pathlib import Path
from contextlib import ExitStack
import hashlib
import html
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from tg12_common import (REVISION,SOURCE_PILOT,SOURCE_CUP_RUN,Lease,read_json,write_json,sha,
                        find_completed_training,safe_name,inference_hashes)


def tail(path,n=10):
    if not Path(path).is_file():return '(not recorded)'
    with Path(path).open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-16000));text=f.read().decode('utf-8',errors='replace')
    return '\n'.join(text.splitlines()[-n:]) or '(empty log)'


def show_result(run):
    run=Path(run);p=run/'rollout_report.json'
    r=read_json(p) if p.is_file() else {}
    pr=read_json(run/'policy_report.json') if (run/'policy_report.json').is_file() else {}
    lines=['TABLEGUARD 17 — FINAL SUMMARY','Run: '+str(run)]
    for key in ('status','task_success','end_reason','error','policy_calls','physics_steps',
                'rollout_simulation_s','worker_wall_time_s','backend','checkpoint_total_updates',
                'source_scene_and_assets_unchanged'):
        lines.append(key+': '+str(r.get(key)))
    ev=r.get('evaluator') or {}
    lines.append('CUP MILESTONES: '+str({k:ev.get(k) for k in ('grasp_verified','lift_verified','transfer_verified','placement_verified','release_verified','retreat_verified')}))
    lines.append('COMMAND FILTER: '+str(r.get('command_filter')))
    lines.append('Source checkpoint unchanged: '+str(pr.get('source_checkpoint_unchanged')))
    lines.append('Policy error: '+str(pr.get('error')))
    lines.append('Complete two-arm task / selective repair: NOT evaluated by this cup integration test.')
    lines.append('Fresh simulated camera feedback; physics pauses for inference. Not a wall-clock real-time result.')
    if (run/'replay.html').is_file():lines.append('OPEN REPLAY IN CHROME/EDGE: '+str(run/'replay.html'))
    if not r or r.get('status') in ('failed','initializing','running'):
        lines.append('Simulator log:\n'+tail(run/'simulator.log'))
        lines.append('Policy log:\n'+tail(run/'policy.log'))
    text='\n'.join(lines)
    try:
        from IPython.display import display,HTML,Image
        display(HTML('<pre style="white-space:pre-wrap;line-height:1.4;font-size:13px">'+html.escape(text)+'</pre>'))
        images=r.get('final_images') or {}
        if images.get('top'):
            path=run/images['top']
            if path.is_file():display(Image(filename=str(path),width=600))
    except ImportError:print(text,flush=True)
    return r


def child_env(root,prefix):
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','CONDA_PROMPT_MODIFIER'):env.pop(k,None)
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(root/'models'/'hf_cache'),
        HF_HUB_CACHE=str(root/'models'/'hf_cache'/'hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    old=str(Path(sys.prefix)).lower()
    others=[p for p in env.get('PATH','').split(os.pathsep) if p and not p.lower().startswith(old)]
    env['PATH']=os.pathsep.join(map(str,[prefix,prefix/'Scripts',prefix/'Library'/'bin',prefix/'bin']))+os.pathsep+os.pathsep.join(others)
    env['CONDA_PREFIX']=str(prefix)
    return env


def cpu_name():
    if os.name=='nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                return winreg.QueryValueEx(k,'ProcessorNameString')[0]
        except OSError:pass
    return platform.processor() or 'not reported'


def run_children(run,settings):
    children={};files={};started=time.monotonic();last=0;sim_started=False
    def launch(name,python,script):
        log=run/(name+'.log');f=log.open('x',encoding='utf-8');files[name]=f
        state={'status':'starting','parent_pid':os.getpid(),'started_unix':time.time()}
        write_json(run/(name+'.status.json'),state)
        print('Log:',log,flush=True)
        try:p=subprocess.Popen([str(python),'-u','-X','faulthandler',str(run/script),str(run/'settings.json')],
            cwd=str(run),env=child_env(Path(settings['root']),Path(python).parent),stdout=f,stderr=subprocess.STDOUT)
        except BaseException:
            state.update(status='failed_to_start',error=traceback.format_exc());write_json(run/(name+'.status.json'),state);raise
        state.update(status='running',pid=p.pid);write_json(run/(name+'.status.json'),state);children[name]=(p,state)
        return p
    try:
        policy=launch('policy',settings['model_python'],'tg12_policy.py')
        while not (run/'ipc'/'policy_ready.json').is_file():
            if policy.poll() is not None:raise RuntimeError('Policy startup failed. '+tail(run/'policy.log',15))
            if time.monotonic()-started>180:raise TimeoutError('Policy startup exceeded 180 seconds')
            if (run/'cancel.request').exists():raise KeyboardInterrupt()
            if time.monotonic()-last>=20:
                print('12 STATUS: loading saved model | elapsed %.0fs'%(time.monotonic()-started),flush=True);last=time.monotonic()
            time.sleep(.1)
        sim=launch('simulator',settings['simulator_python'],'tg12_simulator.py');sim_started=True
        while sim.poll() is None:
            if policy.poll() is not None:
                if (run/'ipc'/'simulator_done.json').exists():break
                raise RuntimeError('Policy worker exited before simulator. '+tail(run/'policy.log',15))
            if time.monotonic()-started>settings['wall_timeout_s']+180:raise TimeoutError('Total rollout deadline exceeded')
            if (run/'cancel.request').exists():raise KeyboardInterrupt()
            if time.monotonic()-last>=20:
                sp=read_json(run/'simulator_progress.json') if (run/'simulator_progress.json').is_file() else {}
                pp=read_json(run/'policy_progress.json') if (run/'policy_progress.json').is_file() else {}
                print('12 STATUS: sim=%s | policy=%s | wall=%.0fs | sim_t=%s | policy_calls=%s'%(
                    sp.get('stage','starting'),pp.get('stage','starting'),time.monotonic()-started,
                    sp.get('simulation_time_s'),sp.get('policy_calls')),flush=True)
                last=time.monotonic()
            time.sleep(.1)
        sim.wait(timeout=15)
        if sim.returncode!=0:raise RuntimeError('Simulator worker failed. '+tail(run/'simulator.log',15))
        policy.wait(timeout=60)
        if policy.returncode!=0:raise RuntimeError('Policy failed finalization. '+tail(run/'policy.log',15))
    finally:
        if any(p.poll() is None for p,_ in children.values()):
            (run/'cancel.request').touch(exist_ok=True)
        # Cooperative shutdown first; terminate only children started by this invocation.
        for name in ('simulator','policy'):
            if name not in children:continue
            p,state=children[name]
            if p.poll() is None:
                try:p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            state.update(status='finished' if p.returncode==0 else 'failed_or_interrupted',returncode=p.returncode,
                         wall_elapsed_s=time.monotonic()-started)
            write_json(run/(name+'.status.json'),state)
        for f in files.values():f.close()


def run_12(root,sources,backend='pytorch_cuda',start_new=False):
    root=Path(root).expanduser().resolve()
    if not (root/'tableguard').is_dir():raise FileNotFoundError('Expected existing TableGuard_Starter project: '+str(root))
    if backend not in ('pytorch_cuda','pytorch_cpu'):raise ValueError('Select explicit PyTorch backend; old 10C IR has wrong weights for notebook11')
    existing_parent=root/'artifacts'/'learned_rollout_17'
    existing_pointer=existing_parent/'selected_rollout.json'
    if existing_pointer.is_file() and not start_new:
        chosen=read_json(existing_pointer)
        print('Viewing saved 12 evidence; no new workers. Saved labels are not a liveness check.',flush=True)
        return show_result(existing_parent/safe_name(chosen['run']))
    training,waiting=find_completed_training(root)
    if waiting:
        text='TABLEGUARD 17 — PREREQUISITE STATUS\n'+json.dumps(waiting,indent=2)
        print(text,flush=True);return waiting
    # Discover before creating an attempt, so a missing prerequisite does not leave a phantom run.
    model_env=Path.home()/'.tableguard'/'envs'/'vla312'
    model_python=model_env/('python.exe' if os.name=='nt' else 'bin/python')
    if not model_python.is_file():raise FileNotFoundError('Existing model environment missing; nothing installed: '+str(model_python))
    if Path(sys.prefix).name.lower()!='tableguard':
        raise RuntimeError('Select Python (TableGuard). Model inference is invoked in vla312 automatically.')
    import importlib.util
    for m in ('mujoco','numpy','PIL'):
        if importlib.util.find_spec(m) is None:raise RuntimeError('Required simulator package missing in current kernel: '+m+'; no install attempted')
    cup=root/'artifacts'/'cup_transfer_07c'/SOURCE_CUP_RUN
    cup_report=read_json(cup/'transfer_report.json')
    for flag in ('completed','transfer_test_passed','grasp_verified','lift_verified','placement_verified','release_verified'):
        if cup_report.get(flag) is not True:raise ValueError('Source 07C is not completely passed: '+flag)
    scene=Path(cup_report['scene'])
    scene=(scene if scene.is_absolute() else cup/scene).resolve()
    if not scene.is_relative_to(cup.resolve()) or not scene.is_file():raise ValueError('Expected original generated 07C scene inside its run folder')
    if sha(scene)!=cup_report['scene_sha256']:raise ValueError('Original 07C scene changed')
    contract=read_json(Path(training['checkpoint'])/'contract.json')
    if Path(contract['source_run']).resolve()!=cup.resolve():raise ValueError('Training source is not the selected 07C episode')
    parent=root/'artifacts'/'learned_rollout_17';parent.mkdir(exist_ok=True)
    shared=root/'artifacts'/'smolvla_openvino_10';trainparent=root/'artifacts'/'smolvla_train_11'
    with ExitStack() as locks:
        locks.enter_context(Lease(parent/'launcher.lck'))
        pointer=parent/'selected_rollout.json'
        if pointer.is_file() and not start_new:
            chosen=read_json(pointer)
            print('Reading saved 12 attempt. No simulation, model, or retry launched.',flush=True)
            return show_result(parent/safe_name(chosen['run']))
        if not pointer.exists() and any(p.is_dir() for p in parent.iterdir()):raise RuntimeError('Unselected 12 evidence exists; do not overwrite it')
        if (root/'artifacts'/'smolvla_09'/'active_pilot.lock').exists():raise RuntimeError('09 pilot marker present; finish/inspect it, no deletion attempted')
        for p in (root/'artifacts').glob('*/active_run.lock'):
            raise RuntimeError('Existing robot-run marker: '+str(p)+'; no new worker started')
        locks.enter_context(Lease(shared/'launcher.lck'));locks.enter_context(Lease(trainparent/'launcher.lck'))
        # Probe active workers. The model worker holds these again during inference.
        for path in (shared/'worker.lck',trainparent/'worker.lck',parent/'policy_worker.lck'):
            with Lease(path):pass
        run=parent/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6]);run.mkdir()
        (run/'ipc').mkdir()
        for name,text in sources.items():
            safe_name(name);compile(text,name,'exec')
            (run/name).write_text(text,encoding='utf-8',newline='\n')
        sr=training['training_report']
        # Notebook 15A reload-verifies its checkpoint but does not create the older
        # Notebook-11 final_checkpoint_hashes.json file. Snapshot the inference
        # checkpoint identity here, before either worker starts, and use that as
        # the immutable baseline throughout this rollout.
        checkpoint_hash_baseline=run/'checkpoint_inference_hashes_at_start.json'
        write_json(checkpoint_hash_baseline,inference_hashes(Path(training['checkpoint'])))
        settings=dict(revision=REVISION,root=str(root),output=str(run),
            checkpoint=training['checkpoint'],training_run=training['training_run'],
            reload_verification=training.get('reload_verification'),
            expected_hashes_file=str(checkpoint_hash_baseline),training_versions=sr.get('versions',{}),
            checkpoint_total_updates=sr['checkpoint_total_optimizer_updates'],
            scene=str(scene),scene_sha256=cup_report['scene_sha256'],instruction=contract['original_instruction'],
            simulator_python=sys.executable,model_python=str(model_python),backend=backend,cpu_name=cpu_name(),
            max_sim_s=45.,wall_timeout_s=1200,seed=12,bounded_commands=True,
            policy_leases=[str(shared/'worker.lck'),str(trainparent/'worker.lck'),str(parent/'policy_worker.lck')])
        write_json(run/'settings.json',settings)
        write_json(run/'source_training_report.json',sr)
        write_json(run/'checkpoint_reload_verification.json',training.get('reload_verification') or {'basis':'original training verification'})
        write_json(pointer,{'run':run.name,'training_run':Path(training['training_run']).name,'revision':REVISION})
        print('NEW 17 LEARNED ROLLOUT:',run,flush=True)
        print('Checkpoint: Notebook 15A multi-episode weights (5 successful episodes) | backend:',backend,flush=True)
        print('Actual MuJoCo physics + fresh RGB feedback; no IK/playback. Original 07C scene.',flush=True)
        print('Task scope: cup-transfer integration; not full two-arm task, camera verifier, or repair.',flush=True)
        print('Raw and range/slew-filtered model commands both saved. Physics pauses during inference.',flush=True)
        print('To cancel without interrupting a busy cell, create cancel.request in this run folder.',flush=True)
        error=None
        try:run_children(run,settings)
        except BaseException as exc:
            error=exc;(run/'launcher_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            p=run/'rollout_report.json';r=read_json(p) if p.is_file() else {}
            r.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',task_success=False if r.get('physics_steps',0)>0 else None,
                     launcher_error=type(exc).__name__+': '+str(exc))
            write_json(p,r)
        finally:
            pr=read_json(run/'policy_report.json') if (run/'policy_report.json').is_file() else {}
            p=run/'rollout_report.json'
            if p.is_file():
                r=read_json(p);r['checkpoint_integrity_verified']=pr.get('source_checkpoint_unchanged')
                if pr.get('source_checkpoint_unchanged') is not True and r.get('task_success') is True:
                    r.update(task_success=False,status='failed',integrity_error='Checkpoint integrity not verified')
                write_json(p,r)
                try:
                    from tg12_replay import build_replay
                    build_replay(run)
                except Exception as exc:print('Replay export issue:',type(exc).__name__,str(exc),flush=True)
            result=show_result(run)
        if error is not None:
            print('12 stopped. Inspect the printed outcome; no automatic retry.',flush=True)
        return result
