"""Recheck a saved 200-update checkpoint. Never train or alter original evidence."""
from pathlib import Path
from contextlib import ExitStack
import gc
import importlib.metadata as md
import json
import os
import sys
import time
import traceback
import numpy as np
from tg11_common import (Lease, read_json, write_json, hashes, verify_hashes, sha)
from tg_fp32_loader import load_fp32_policy

EXPECTED_ERROR = 'RuntimeError: Reloaded checkpoint predictions differ beyond declared tolerance'
LOADER_ID = 'initialize_then_fp32_then_strict_safetensors'


def event(out, stage, **details):
    value = dict(stage=stage, unix_time=time.time(), **details)
    print('TG11A ' + json.dumps(value, allow_nan=False), flush=True)
    write_json(out/'progress.json', value)


def compare(reference, prediction):
    if (reference.shape != prediction.shape or reference.ndim != 3
            or reference.shape[1:] != (20, 12) or not np.isfinite(reference).all()
            or not np.isfinite(prediction).all()):
        raise ValueError('Invalid or inconsistent probe predictions.')
    error = np.abs(reference.astype(np.float64)-prediction.astype(np.float64))
    return {'passed': bool(np.allclose(prediction, reference, atol=1e-6, rtol=1e-5)),
            'atol': 1e-6, 'rtol': 1e-5, 'maximum_normalized_difference': float(error.max()),
            'values_outside_tolerance': int(np.sum(error > 1e-6+1e-5*np.abs(reference))),
            'shape': list(reference.shape)}


def validate_training_report(report):
    if (report.get('status') != 'failed' or report.get('error') != EXPECTED_ERROR
            or report.get('integrity_error')):
        raise ValueError('Recovery is only for the inspected prediction-reload failure.')
    for key in ('checkpoint_saved', 'source_checkpoint_unchanged',
                'source_data_pack_unchanged', 'source_recording_unchanged'):
        if report.get(key) is not True:
            raise ValueError('Original training evidence is incomplete: ' + key)
    if report.get('updates_completed') != 200 or report.get('updates_requested') != 200:
        raise ValueError('All 200 requested updates must already be complete.')
    if report.get('checkpoint_total_optimizer_updates') != 201:
        raise ValueError('Unexpected total checkpoint update count.')
    tolerance = report.get('checkpoint_reload_tolerances')
    if tolerance != {'atol': 1e-6, 'rtol': 1e-5}:
        raise ValueError('Original reload tolerance differs. No tolerance was changed.')


def main(config_path):
    cfg = read_json(config_path)
    out = Path(cfg['output']); train = Path(cfg['training_run']).resolve()
    source = Path(cfg['source_pilot']).resolve()
    final = train/'checkpoint_final'; pack = source/'data_pack'
    report = dict(kind='SAVED_CHECKPOINT_RELOAD_VERIFICATION_NO_TRAINING_OR_ROBOT',
        status='in_progress', stage='preflight', error=None,
        training_run=str(train), checkpoint_path=str(final),
        additional_optimizer_updates=0, robot_steps=0, task_success=None,
        checkpoint_reloaded_verified=False, original_prediction_parity_passed=False,
        repeated_fresh_load_parity_passed=False, original_evidence_unchanged=None,
        loader_id=LOADER_ID, loader_sha256=sha(Path(__file__).with_name('tg_fp32_loader.py')),
        source_checkpoint_unchanged=None, source_data_pack_unchanged=None,
        source_recording_unchanged=None)
    path=out/'verification_report.json'
    write_json(path, report)
    old_hashes = final_hashes = None
    try:
        event(out, 'validate_saved_training_no_update')
        r=read_json(train/'training_report.json'); validate_training_report(r)
        if Path(r['checkpoint_path']).resolve()!=final:
            raise ValueError('Final checkpoint path mismatch.')
        if Path(read_json(train/'training_settings.json')['source_pilot']).resolve()!=source:
            raise ValueError('Original source pilot changed.')
        metadata=read_json(final/'training_state.json')
        if metadata.get('new_updates')!=200 or metadata.get('total_updates')!=201:
            raise ValueError('Checkpoint training metadata does not match.')
        # Preserve all files in this original training run, including previous failure evidence.
        old_hashes=hashes(train)
        write_json(out/'original_training_files.json',old_hashes)
        final_hashes=read_json(train/'final_checkpoint_hashes.json')
        verify_hashes(final,final_hashes)
        verify_hashes(source/'smoke_checkpoint',read_json(train/'source_checkpoint_hashes.json'))
        verify_hashes(pack,read_json(train/'source_data_pack_hashes.json'))
        c=read_json(pack/'contract.json')
        for p,digest in c['source_hashes'].items():
            if not Path(p).is_file() or sha(p)!=digest:
                raise ValueError('Original observation/trace changed: '+p)
        for name in ('contract.json','normalization.json'):
            if sha(final/name)!=sha(pack/name):
                raise ValueError('Normalizer/contract mismatch.')
        import torch
        from transformers import AutoTokenizer
        from recording_bridge import ChunkDataset
        from tg11a_probes import evaluate_probes
        versions={k:md.version(k) for k in ('torch','torchvision','lerobot','transformers','numpy','safetensors')}
        for k,v in r['versions'].items():
            if versions.get(k)!=v:
                raise RuntimeError('Environment changed since training: '+k)
        if versions['lerobot']!='0.6.1' or versions['transformers']!='5.5.4':
            raise RuntimeError('Unexpected runtime; no install/downgrade attempted.')
        if not torch.cuda.is_available():
            raise RuntimeError('Use the original CUDA device; no cross-device tolerance comparison.')
        if torch.cuda.get_device_name(0)!=r.get('training_device'):
            raise RuntimeError('CUDA hardware name differs from original training.')
        report.update(versions=versions, device_name=torch.cuda.get_device_name(0),
            training_report_sha256=sha(train/'training_report.json'),
            checkpoint_hashes_file_sha256=sha(train/'final_checkpoint_hashes.json'),
            original_training_error=r['error'], original_training_status=r['status'],
            new_updates=200,total_updates=201,
            original_reload_maximum_normalized_difference=r.get('checkpoint_reload_maximum_normalized_difference'))
        torch.set_num_threads(min(8,os.cpu_count() or 1)); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        data=ChunkDataset(pack)
        definition=read_json(train/'probe_definition.json');indices=definition['frame_indices']
        if definition['noise_seeds']!=[7,19]:raise ValueError('Original noise seeds differ.')
        with np.load(train/'after_predictions.npz',allow_pickle=False) as z:
            original=z['normalized_prediction'].copy()
            original_targets=z['normalized_target'].copy();original_pad=z['action_is_pad'].copy()
        if original.shape!=(2*len(indices),20,12):raise ValueError('Unexpected reference shape.')
        if (train/'reloaded_predictions.npz').is_file():
            with np.load(train/'reloaded_predictions.npz',allow_pickle=False) as z:
                report['original_failed_comparison']=compare(original,z['normalized_prediction'])
        report['reference_file_sha256']=sha(train/'after_predictions.npz')
        tokenizer=AutoTokenizer.from_pretrained(final/'tokenizer',local_files_only=True)
        old_tokenizer=AutoTokenizer.from_pretrained(source/'smoke_checkpoint'/'tokenizer',local_files_only=True)
        text=[c['original_instruction'].rstrip()+'\n']
        if tokenizer(text)['input_ids']!=old_tokenizer(text)['input_ids']:
            raise ValueError('Saved tokenizer changed.')
        first = None
        for repetition in (1,2):
            report['stage']='fp32_before_load_'+str(repetition);write_json(path,report)
            event(out,report['stage'], additional_training_updates=0)
            model, details=load_fp32_policy(final)
            report['load_'+str(repetition)]=details
            write_json(out/('tensor_audit_'+str(repetition)+'.json'),details)
            model.to('cuda').eval(); model.config.device='cuda'
            event(out,'original_probe_comparison',load=repetition,cases=original.shape[0],atol=1e-6,rtol=1e-5)
            metrics,pred=evaluate_probes(model,data,indices,tokenizer,'cuda',out,'load_'+str(repetition))
            with np.load(out/('load_'+str(repetition)+'_predictions.npz'),allow_pickle=False) as z:
                if not np.array_equal(z['normalized_target'],original_targets) or not np.array_equal(z['action_is_pad'],original_pad):
                    raise ValueError('The original input/target alignment changed.')
            result=compare(original,pred); report['load_'+str(repetition)+'_vs_original']=result
            write_json(path,report)
            if not result['passed']:
                raise RuntimeError('FP32-first reload still differs from original post-training predictions. No acceptance criteria relaxed.')
            if first is not None:
                report['fresh_loads_comparison']=compare(first,pred)
                if not report['fresh_loads_comparison']['passed']:
                    raise RuntimeError('Independent fresh loads differ beyond the original tolerance.')
            else:first=pred.copy()
            report['verified_in_sample_right_arm_mae_rad']=metrics['right_arm_mae_rad']
            del model;gc.collect();torch.cuda.empty_cache()
        report.update(checkpoint_reloaded_verified=True,original_prediction_parity_passed=True,
            repeated_fresh_load_parity_passed=True,status='checkpoint_reload_verified',stage='complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=type(exc).__name__+': '+str(exc))
        (out/'verification_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        try:
            if old_hashes is not None:
                verify_hashes(train,old_hashes);report['original_evidence_unchanged']=True
            if final_hashes is not None:
                verify_hashes(final,final_hashes)
                verify_hashes(source/'smoke_checkpoint',read_json(train/'source_checkpoint_hashes.json'))
                report['source_checkpoint_unchanged']=True
                verify_hashes(pack,read_json(train/'source_data_pack_hashes.json'));report['source_data_pack_unchanged']=True
                c=read_json(pack/'contract.json')
                report['source_recording_unchanged']=all(Path(p).is_file() and sha(p)==d for p,d in c['source_hashes'].items())
                if not report['source_recording_unchanged']:raise RuntimeError('Source recording changed.')
        except BaseException as exc:
            report.update(status='failed',integrity_error=type(exc).__name__+': '+str(exc),original_evidence_unchanged=False)
        write_json(path,report)
        event(out,'verification_finished',status=report['status'],error=report.get('error'),robot_steps=0,additional_updates=0)
    if report['status']!='checkpoint_reload_verified':raise RuntimeError('Verification incomplete; rollout remains blocked.')


if __name__=='__main__':
    settings=read_json(sys.argv[1])
    with ExitStack() as stack:
        for p in settings['worker_leases']:stack.enter_context(Lease(p))
        main(sys.argv[1])
