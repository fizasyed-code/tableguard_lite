from __future__ import annotations
import json
from pathlib import Path
from .common import ROOT, write_json


def visualize_sample(manifest_path: Path) -> dict:
    """Save separate plots for actual downloaded video frames and recorded actions.

    No inference, goal detection, recovery or robot execution is performed here.
    Media and Parquet rows are visualized independently; alignment is not certified.
    """
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('The download manifest is not complete; resolve the earlier download error first')
    base = manifest_path.parent
    allowed = [base / x['path'] for x in manifest.get('files',[])]
    videos = [p for p in allowed if p.suffix.lower() == '.mp4']
    parquets = [p for p in allowed if p.suffix.lower() == '.parquet']
    if not videos:
        raise ValueError('No video sample downloaded. Fetch a v2 episode with --media first.')
    tag = manifest['repo_id'].replace('/','__') + '_' + manifest['resolved_revision'][:10]
    out = ROOT / 'artifacts' / 'public_reference' / tag / f"episode_{manifest.get('episode_index',0):06d}"
    out.mkdir(parents=True, exist_ok=True)
    report = {'kind':'PUBLIC_REFERENCE_PLAYBACK_NOT_MODEL_OUTPUT', 'repo_id':manifest['repo_id'],
              'revision':manifest['resolved_revision'], 'images': [], 'action_plot': None, 'warnings': []}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f'OpenCV cannot decode {video.name}. Check the codec; no frames were fabricated.')
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if count <= 0:
                raise RuntimeError('Video frame count unavailable; inspect this codec manually')
            indices = sorted(set([0, count//2, count-1]))
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    report['warnings'].append(f'Failed to decode frame {index}; skipped, not replaced')
                    continue
                fig, ax = plt.subplots(figsize=(9,6))
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                ax.set_title(f'Public reference playback | frame {index} / {count-1}')
                fig.text(0.5, 0.025, 'Not TableGuard inference or robot execution', ha='center', fontsize=10)
                image_path = out / f'frame_{index:06d}.png'
                fig.savefig(image_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                report['images'].append(str(image_path.relative_to(ROOT)))
            report['video_frame_count'] = count
            report['video_fps'] = fps
        finally:
            cap.release()
    for parquet in parquets:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            report['warnings'].append('pyarrow not installed: action plot skipped')
            break
        table = pq.read_table(parquet)
        report['parquet_rows'] = table.num_rows
        report['parquet_columns'] = table.column_names
        if 'action' in table.column_names:
            action = np.asarray(table.column('action').to_pylist(), dtype=float)
            if action.ndim != 2 or not np.isfinite(action).all():
                raise ValueError('Expected finite two-dimensional recorded action vectors')
            x = np.arange(action.shape[0])
            fig, ax = plt.subplots(figsize=(10,5))
            for channel in range(action.shape[1]):
                ax.plot(x, action[:,channel], label=f'channel {channel}')
            ax.set(xlabel='Recorded row index', ylabel='Raw stored action value (units unverified)',
                   title='Public demonstration actions — NOT TableGuard predictions')
            ax.legend(loc='best', ncol=2)
            fig.tight_layout()
            image_path = out / 'recorded_actions.png'
            fig.savefig(image_path, dpi=150)
            plt.close(fig)
            report['action_plot'] = str(image_path.relative_to(ROOT))
    write_json(out / 'inspection.json', report)
    return report
