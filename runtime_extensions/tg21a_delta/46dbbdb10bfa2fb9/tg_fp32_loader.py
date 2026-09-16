"""Lossless FP32 initialization BEFORE loading locally saved SmolVLA weights.
No package patches, model download, optimizer, or robot code.
"""
from pathlib import Path
from collections import Counter


def load_fp32_policy(checkpoint):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint).resolve()
    weight_files = list(checkpoint.glob('*.safetensors'))
    if len(weight_files) != 1 or weight_files[0].name != 'model.safetensors':
        raise ValueError('Expected the original unsharded model.safetensors checkpoint.')
    cfg = SmolVLAConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device = 'cpu'
    cfg.load_vlm_weights = False
    if cfg.compile_model or cfg.rtc_config is not None:
        raise ValueError('Unexpected compiled or RTC checkpoint; no configuration workaround.')
    if cfg.chunk_size != 20 or list(cfg.action_feature.shape) != [12]:
        raise ValueError('Expected the saved 20 x 12 action interface.')

    # The order matters. .float() AFTER from_pretrained cannot undo loading
    # FP32 checkpoint values into lower-precision parameter storage.
    policy = SmolVLAPolicy(cfg)
    native_dtypes = {k: v.dtype for k, v in policy.state_dict().items()}
    details = {'loader': 'initialize_then_fp32_then_strict_safetensors',
               'initialized_tensor_dtypes': dict(Counter(map(str, native_dtypes.values()))),
               'lower_precision_roundtrip_affected_tensors': 0,
               'maximum_roundtrip_weight_difference': 0.0,
               'roundtrip_examples': [], 'saved_tensors_checked': 0,
               'loaded_tensors_equal_saved': False,
               'lower_precision_roundtrip_is_diagnostic_only': True}
    # Check the potential loss without applying it to the policy.
    with safe_open(str(weight_files[0]), framework='pt', device='cpu') as store:
        for key in store.keys():
            disk = store.get_tensor(key)
            if disk.is_floating_point() and disk.dtype != torch.float32:
                raise ValueError('This recovery expects all saved floating tensors to be FP32: ' + key)
            native = native_dtypes.get(key)
            if disk.dtype == torch.float32 and native in (torch.float16, torch.bfloat16):
                rounded = disk.to(native).to(torch.float32)
                difference = float((disk - rounded).abs().max()) if disk.numel() else 0.0
                if difference > 0:
                    details['lower_precision_roundtrip_affected_tensors'] += 1
                    details['maximum_roundtrip_weight_difference'] = max(
                        details['maximum_roundtrip_weight_difference'], difference)
                    if len(details['roundtrip_examples']) < 8:
                        details['roundtrip_examples'].append(
                            {'tensor': key, 'initial_dtype': str(native), 'maximum_difference': difference})
    policy.float()
    missing, unexpected = load_model(policy, str(weight_files[0]), strict=True, device='cpu')
    if missing or unexpected:
        raise RuntimeError('Strict state load reported unmatched tensors.')
    stored_state = policy.state_dict()
    with safe_open(str(weight_files[0]), framework='pt', device='cpu') as store:
        for key in store.keys():
            disk = store.get_tensor(key)
            actual = stored_state.get(key)
            if actual is None or actual.dtype != disk.dtype or not torch.equal(actual.cpu(), disk):
                raise RuntimeError('Loaded parameter/buffer differs from saved tensor: ' + key)
            details['saved_tensors_checked'] += 1
    details['loaded_tensors_equal_saved'] = True
    del stored_state
    policy.eval()
    policy.requires_grad_(False)
    return policy, details
