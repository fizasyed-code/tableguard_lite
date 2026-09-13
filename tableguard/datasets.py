"""Bounded downloads from public Hugging Face dataset repositories.

No model weights, remote Python code, pickle files, or full-dataset downloads.
The default sample reader supports LeRobot v2 per-episode media. v3 metadata
is inspected, but media extraction requires explicit episode/shard indexing.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from .common import ROOT, write_json


def _validate_repo(repo_id: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Expected a public Hugging Face namespace/repository ID')
    return repo_id


def validate_relative_file(name: str) -> str:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe repository path: {name!r}')
    return str(p)


def summarize_metadata(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get('features', {})
    if not isinstance(features, dict):
        raise ValueError('Metadata features must be a dictionary')
    action = features.get('action', {})
    cameras = {k: {'dtype': v.get('dtype'), 'shape': v.get('shape')}
               for k, v in features.items() if isinstance(v, dict) and v.get('dtype') in ('video','image')}
    return {
        'format': info.get('codebase_version'), 'robot_type': info.get('robot_type'),
        'total_episodes': info.get('total_episodes'), 'total_frames': info.get('total_frames'),
        'fps': info.get('fps'), 'action_shape': action.get('shape'),
        'cameras': cameras, 'data_path': info.get('data_path'), 'video_path': info.get('video_path'),
        'compatibility': 'NOT A CERTIFICATION of task, action semantics, or policy compatibility',
    }


def episode_files_v2(info: dict[str, Any], episode: int, camera: str | None = None) -> list[str]:
    """Construct paths only using the repository metadata's templates."""
    version = str(info.get('codebase_version', ''))
    if not version.startswith('v2.'):
        raise ValueError('Media sampling currently supports LeRobot v2 only. v3 needs episode-to-shard offsets; metadata inspection is still supported.')
    total = info.get('total_episodes')
    if not isinstance(episode, int) or episode < 0 or not isinstance(total, int) or episode >= total:
        raise ValueError('Episode index out of range or invalid total_episodes')
    chunk_size = info.get('chunks_size')
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('Missing/invalid chunks_size')
    features = info.get('features', {})
    videos = sorted(k for k,v in features.items() if isinstance(v,dict) and v.get('dtype') == 'video')
    if not videos:
        raise ValueError('No video camera in metadata; use metadata-only inspection')
    if camera is None:
        camera = videos[0]
    if camera not in videos:
        raise ValueError(f'Camera {camera!r} not in {videos}')
    fields = {'episode_chunk': episode // chunk_size, 'episode_index': episode, 'video_key': camera}
    paths = []
    for key in ['data_path','video_path']:
        template = info.get(key)
        if not isinstance(template, str):
            raise ValueError(f'Missing {key}')
        try:
            paths.append(validate_relative_file(template.format(**fields)))
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f'Unsupported {key} template: {template!r}') from e
    return paths


def fetch_reference(repo_id: str = 'lerobot/svla_so101_pickplace', *,
                    episode: int = 0, include_media: bool = False, camera: str | None = None,
                    max_mb: float = 50.0, revision: str = 'main',
                    output_root: Path | None = None) -> Path:
    _validate_repo(repo_id)
    if not 0 < max_mb <= 1024:
        raise ValueError('max_mb must be greater than 0 and no more than 1024')
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
    except ImportError as e:
        raise RuntimeError('Install requirements-phase1.txt before requesting public data.') from e
    api = HfApi(token=False)
    try:
        remote = api.dataset_info(repo_id=repo_id, revision=revision, timeout=20)
    except Exception as e:
        raise RuntimeError('Public dataset metadata request failed. Check internet access, repository availability, and revision. No credentials are required for the listed public references.') from e
    commit = remote.sha
    if not commit:
        raise RuntimeError('Cannot pin a dataset revision; stopping rather than mixing revisions')
    siblings = {s.rfilename for s in remote.siblings or []}
    root = Path(output_root or ROOT / 'data' / 'public') / repo_id.replace('/','__') / commit
    root.mkdir(parents=True, exist_ok=True)
    cap = int(max_mb * 1_000_000)
    used = 0
    manifest: dict[str, Any] = {
        'repo_id': repo_id, 'repo_type': 'dataset', 'resolved_revision': commit,
        'requested_revision': revision, 'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'partial', 'files': [], 'warnings': [], 'media_requested': include_media,
        'purpose': 'Public reference data only; not a dual-arm table-setting controller or TableGuard benchmark',
    }
    def save_manifest():
        write_json(root / 'download_manifest.json', manifest)
    def fetch_file(name: str, per_file_cap: int | None = None) -> Path:
        nonlocal used
        name = validate_relative_file(name)
        if siblings and name not in siblings:
            raise FileNotFoundError(f'Repository has no file {name!r} at {commit}')
        metadata = get_hf_file_metadata(hf_hub_url(repo_id, name, repo_type='dataset', revision=commit), token=False, timeout=20)
        size = metadata.size
        if size is None:
            raise RuntimeError(f'Cannot verify size of {name}; refusing an unbounded download')
        if used + size > cap or (per_file_cap is not None and size > per_file_cap):
            raise RuntimeError(f'Download budget exceeded by {name} ({size:,} bytes). Increase --max-mb deliberately or select a smaller episode.')
        if metadata.commit_hash and metadata.commit_hash != commit:
            raise RuntimeError('Revision mismatch in file metadata')
        actual = Path(hf_hub_download(repo_id=repo_id, filename=name, repo_type='dataset', revision=commit, local_dir=root, token=False, etag_timeout=20))
        actual_size = actual.stat().st_size
        if actual_size != size:
            raise RuntimeError(f'File size mismatch for {name}')
        digest = hashlib.sha256()
        with actual.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        used += size
        manifest['files'].append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
        manifest['downloaded_bytes'] = used
        save_manifest()
        return actual
    save_manifest()
    try:
        info_path = fetch_file('meta/info.json', per_file_cap=2_000_000)
        info = json.loads(info_path.read_text(encoding='utf-8'))
        manifest['metadata_summary'] = summarize_metadata(info)
        for name in ['README.md','LICENSE','LICENSE.txt','LICENSE.md','meta/tasks.jsonl']:
            if name in siblings:
                fetch_file(name, per_file_cap=2_000_000)
        if include_media:
            paths = episode_files_v2(info, episode, camera)
            for name in paths:
                fetch_file(name)
            manifest['episode_index'] = episode
        manifest['status'] = 'complete'
        save_manifest()
    except Exception as e:
        manifest['status'] = 'incomplete'
        manifest['error'] = str(e)
        save_manifest()
        raise
    return root / 'download_manifest.json'
