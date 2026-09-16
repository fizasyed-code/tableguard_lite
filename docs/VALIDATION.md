# Validation record

Prepared 12 September 2026.

## Executed in the preparation runtime

- Python 3.13.5; CPU execution; no required CUDA device.
- `python -m tableguard selftest`: 49 tests, no failures or errors. See `artifacts/unit_test_output.txt` and `artifacts/unit_test_summary.json`.
- `python -m tableguard demo`: seven hand-authored structured goal scenarios. These are logic fixtures, not images, model predictions or robot episodes.
- `python -m tableguard hardware`: local discovery executed; report is named `build_environment_not_user_machine.json` to prevent confusion with the user's hardware.

- `notebooks/00_Start_Here.ipynb`: all offline cells executed successfully; live download intentionally disabled.
- Video-rendering helper: decoded three frames from a temporary generated codec-test video. This was a software rendering test only, not a public dataset or robot episode. No synthetic video is shipped as a public sample.

## Not demonstrated here

- Real internet download using the supplied Python downloader: unavailable because this container could not resolve external hosts. Web research did read the public cards. Downloader tests use an explicitly mocked Hub and temporary test bytes.
- Visualization of an actual public episode: not run against a public download. No reference video is bundled.
- Fresh Python 3.11 dependency installation: not executed. Recommended target remains 3.11 for later Intel integration; requirement ranges are not a full target lockfile. Pin the versions that actually work on your computer.
- VLA inference, GPU/VRAM benchmarks, Intel accelerator execution, MuJoCo rendering, object perception, bimanual manipulation, recovery performance or final submission eligibility.

The result is a tested Phase 1 preparation/supervisor component, not a completed robot application. Never present 49 passing unit tests as 49 successful robot trials.
