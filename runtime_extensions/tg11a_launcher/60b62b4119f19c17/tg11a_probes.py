"""Exact original 11 probe functions; no training loop is included."""
import numpy as np
from tg11_common import action_metrics, write_json

def make_batch(dataset, indices, tokenizer, cfg, device):
    import torch
    from torch.utils.data import default_collate
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    batch = default_collate([dataset[int(i)] for i in indices])
    text = [s.rstrip() + '\n' for s in batch.pop('task')]
    full = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
    if any(len(t) > cfg.tokenizer_max_length for t in full):
        raise ValueError('Recorded instruction exceeds tokenizer limit; no silent truncation')
    tokens = tokenizer(text, padding='max_length', max_length=cfg.tokenizer_max_length,
                       truncation=False, return_tensors='pt')
    batch[OBS_LANGUAGE_TOKENS] = tokens['input_ids'].long()
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens['attention_mask'].bool()
    return {k: v.to(device) for k, v in batch.items()}

def evaluate_probes(policy, dataset, indices, tokenizer, device, out, label):
    """Same recorded samples/noise before and after; never label these held out."""
    import torch
    cfg = policy.config
    values, targets, masks, labels = [], [], [], []
    policy.eval()
    for i in indices:
        batch = make_batch(dataset, [i], tokenizer, cfg, device)
        obs = {k: v for k, v in batch.items() if k not in ('action', 'action_is_pad')}
        for seed in (7, 19):
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim),
                                generator=gen, device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = policy.predict_action_chunk(dict(obs), noise=noise)
            if tuple(pred.shape) != (1, 20, 12) or not torch.isfinite(pred).all():
                raise RuntimeError('Nonfinite prediction or changed action interface')
            values.append(pred.detach().cpu().numpy()[0].copy())
            targets.append(batch['action'].detach().cpu().numpy()[0].copy())
            masks.append(batch['action_is_pad'].detach().cpu().numpy()[0].copy())
            labels.append({'frame_index': int(i), 'noise_seed': seed,
                           'observation_time_s': float(dataset.arrays['observation_time_s'][i])})
    values, targets, masks = np.asarray(values), np.asarray(targets), np.asarray(masks)
    norm = dataset.stats['action']; c = dataset.contract
    metrics = action_metrics(targets, values, masks, norm['mean'], norm['std'],
                             c['action_order'], c['joint_lower_rad'], c['joint_upper_rad'])
    metrics.update(cases=labels, unique_recorded_observations=len(indices), noise_seeds=[7, 19])
    np.savez_compressed(out / (label + '_predictions.npz'), normalized_prediction=values,
                        normalized_target=targets, action_is_pad=masks)
    write_json(out / (label + '_metrics.json'), metrics)
    return metrics, values
