"""Actual SmolVLA policy process: fresh RGB/state -> native action chunk.
Receives no object positions, expert targets, success flags, phase labels or
recorded action arrays. Never loads optimizer_state.pt or commands a robot.
"""
from pathlib import Path
from contextlib import ExitStack
import importlib.metadata as md
import json
import os
import sys
import time
import traceback
import numpy as np
from tg12_common import (Lease, read_json, write_json, sha, inside, event,
    inference_hashes, verify_inference_hashes, OBS_KEYS)
from tg12_control import validate_contract


def main(settings_file):
    cfg=read_json(settings_file);out=Path(cfg['output']);ipc=out/'ipc'
    checkpoint=Path(cfg['checkpoint'])
    report=dict(status='initializing',policy_calls=0,backend=cfg['backend'],
        robot_steps=0,checkpoint=str(checkpoint),error=None,
        observations='fresh RGB cameras, current joint positions, supported instruction only',
        recorded_action_playback=False,simulator_object_state_received=False,
        task_success=None,source_checkpoint_unchanged=None)
    protected=None
    try:
        event(out,'policy','import_model_libraries_no_installation')
        import torch
        from transformers import AutoTokenizer
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        report['versions']={n:md.version(n) for n in ('torch','lerobot','transformers','numpy')}
        previous=cfg['training_versions']
        for n in ('torch','lerobot','transformers','numpy'):
            if n in previous and report['versions'][n]!=previous[n]:
                raise RuntimeError('Version changed since training: '+n+'; nothing reinstalled')
        device={'pytorch_cuda':'cuda','pytorch_cpu':'cpu'}.get(cfg['backend'])
        if device is None: raise ValueError('Only explicit PyTorch CUDA/CPU supported by this integration step')
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic slow CPU fallback')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        report['device_name']=torch.cuda.get_device_name(0) if device=='cuda' else cfg['cpu_name']
        report['openvino_policy_inference']=False
        c=read_json(checkpoint/'contract.json');stats=read_json(checkpoint/'normalization.json')
        order,lo,hi=validate_contract(c,stats)
        event(out,'policy','hash_completed_training_checkpoint')
        protected=inference_hashes(checkpoint)
        expected=read_json(cfg['expected_hashes_file'])
        if protected!={k:v for k,v in expected.items() if Path(k).name!='optimizer_state.pt'}:
            raise RuntimeError('Checkpoint inference files differ from the rollout-start hash baseline')
        report['checkpoint_identity_sha256']=__import__('hashlib').sha256(json.dumps(protected,sort_keys=True).encode()).hexdigest()
        mc=SmolVLAConfig.from_pretrained(checkpoint,local_files_only=True)
        mc.device='cpu';mc.load_vlm_weights=False;mc.compile_model=False
        if mc.rtc_config is not None or mc.chunk_size!=20 or mc.action_feature.shape!=(12,):
            # Shape is list in some serializations; check below without changing it.
            if mc.rtc_config is not None or mc.chunk_size!=20 or list(mc.action_feature.shape)!=[12]:
                raise ValueError('Unexpected checkpoint action/RTC configuration')
        if mc.adapt_to_pi_aloha or mc.use_delta_joint_actions_aloha:
            raise ValueError('Unexpected action coordinate transformation')
        if set(mc.image_features)!={'observation.images.top','observation.images.right_wrist_cam'}:
            raise ValueError('Checkpoint cameras do not match the trained two-camera interface')
        event(out,'policy','load_notebook11_checkpoint_fp32_before_weights',device=device)
        from tg_fp32_loader import load_fp32_policy
        policy, load_audit=load_fp32_policy(checkpoint)
        report['checkpoint_load_audit']=load_audit
        policy.to(device).eval()
        policy.requires_grad_(False);policy.config.device=device
        tok=AutoTokenizer.from_pretrained(checkpoint/'tokenizer',local_files_only=True)
        instruction=cfg['instruction']
        if instruction!=c['original_instruction']: raise ValueError('Unsupported task: this checkpoint was trained on the recorded cup instruction')
        text=[instruction.rstrip()+'\n']
        full=tok(text,add_special_tokens=True,truncation=False)['input_ids']
        if any(len(x)>mc.tokenizer_max_length for x in full): raise ValueError('Instruction exceeds token length; not truncated')
        kw=dict(padding='max_length',max_length=mc.tokenizer_max_length,truncation=False,return_tensors='pt')
        tokens=tok(text,**kw);native=policy.model.vlm_with_expert.processor.tokenizer(text,**kw)
        if not torch.equal(tokens['input_ids'],native['input_ids']) or not torch.equal(tokens['attention_mask'],native['attention_mask']):
            raise RuntimeError('Tokenizer mismatch')
        mean=np.asarray(stats['action']['mean'],np.float64);std=np.asarray(stats['action']['std'],np.float64)
        state_mean=np.asarray(stats['observation.state']['mean'],np.float32);state_std=np.asarray(stats['observation.state']['std'],np.float32)
        report.update(status='ready',action_order=order,denoising_steps=mc.num_steps,
                      checkpoint_total_updates=cfg['checkpoint_total_updates'])
        write_json(out/'policy_report.json',report);write_json(ipc/'policy_ready.json',{'status':'ready','pid':os.getpid(),'backend':cfg['backend']})
        event(out,'policy','ready_for_fresh_camera_observations',backend=cfg['backend'])
        reqid=0;deadline=time.monotonic()+cfg['wall_timeout_s']
        while time.monotonic()<deadline:
            if (out/'cancel.request').exists() or (ipc/'simulator_done.json').exists(): break
            rp=ipc/f'request_{reqid:05d}.json'
            if not rp.is_file():time.sleep(.02);continue
            req=read_json(rp)
            required={'request_id','simulation_time_s','instruction','input_file','input_sha256'}
            if set(req)!=required or req.get('request_id')!=reqid or req.get('instruction')!=instruction:
                raise ValueError('Protocol mismatch or unexpected privileged observation fields')
            inp=inside(ipc,req['input_file'])
            if sha(inp)!=req['input_sha256']:raise ValueError('Observation changed in transit')
            with np.load(inp,allow_pickle=False) as z:
                if set(z.files)!=OBS_KEYS:raise ValueError('Only fresh images and joint state are accepted')
                st=np.array(z['state'],copy=True)
                images=[np.array(z[k],copy=True) for k in ('image_top','image_right_wrist')]
            if st.shape!=(12,) or not np.isfinite(st).all():raise ValueError('Invalid proprioception')
            for im in images:
                if im.shape!=(480,640,3) or im.dtype!=np.uint8:raise ValueError('Image contract mismatch')
            obs={
                'observation.images.top':torch.from_numpy(images[0].astype(np.float32).transpose(2,0,1)/255).unsqueeze(0).to(device),
                'observation.images.right_wrist_cam':torch.from_numpy(images[1].astype(np.float32).transpose(2,0,1)/255).unsqueeze(0).to(device),
                'observation.state':torch.from_numpy((st.astype(np.float32)-state_mean)/state_std).unsqueeze(0).to(device),
                OBS_LANGUAGE_TOKENS:tokens['input_ids'].to(device),
                OBS_LANGUAGE_ATTENTION_MASK:tokens['attention_mask'].bool().to(device)}
            # Common-random-number inference: keep diffusion noise fixed across requests.
            # Fresh observations and state still enter every 100 ms.
            seed=cfg['seed']
            generator=torch.Generator(device=device).manual_seed(seed)
            noise=torch.randn((1,20,mc.max_action_dim),device=device,generator=generator,dtype=torch.float32)
            if device=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():pred=policy.predict_action_chunk(obs,noise=noise)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if tuple(pred.shape)!=(1,20,12) or not torch.isfinite(pred).all():raise RuntimeError('Nonfinite or incorrectly shaped policy action')
            normalized=pred.detach().cpu().numpy()[0].astype(np.float64)
            raw=normalized*std+mean
            if not np.isfinite(raw).all():raise RuntimeError('Nonfinite denormalized actions')
            name=f'response_{reqid:05d}.npz';target=ipc/name;tmp=target.with_suffix('.tmp')
            with tmp.open('xb') as f:np.savez_compressed(f,requested_targets_rad=raw,normalized_prediction=normalized)
            os.replace(tmp,target)
            write_json(ipc/f'response_{reqid:05d}.json',{
                'request_id':reqid,'request_sha256':sha(rp),'output_file':name,'output_sha256':sha(target),
                'inference_s':elapsed,'seed':seed,'backend':cfg['backend'],
                'raw_joint_range_violations':int(((raw<lo)|(raw>hi)).sum())})
            report['policy_calls']+=1;report['latest_inference_s']=elapsed
            if reqid%10==0:
                event(out,'policy','action_chunk_generated',request_id=reqid,inference_s=elapsed)
                write_json(out/'policy_report.json',report)
            reqid+=1
        else:raise TimeoutError('Policy process exceeded the bounded rollout wall time')
        report['status']='stopped_after_rollout'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        (out/'policy_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(ipc/'policy_failed.json',{'error':report['error']})
        event(out,'policy','failed',error=report['error'])
        raise
    finally:
        if protected is not None:
            try:report['source_checkpoint_unchanged']=verify_inference_hashes(checkpoint,protected)
            except Exception as exc:report.update(status='failed',source_checkpoint_unchanged=False,integrity_error=str(exc))
        write_json(out/'policy_report.json',report)


if __name__=='__main__':
    settings=read_json(sys.argv[1])
    with ExitStack() as stack:
        for p in settings['policy_leases']:stack.enter_context(Lease(p))
        main(sys.argv[1])
