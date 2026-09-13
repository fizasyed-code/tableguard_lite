# TableGuard-Lite — Phase 1 starter

**Prepared 12 September 2026. Python-first; CPU-only preparation.**

This pack starts implementation of the agreed TableGuard-Lite project. It is not the full hackathon robot application. The official scene, compatible policy, visual monitor, joint action adapter, eligible Intel machine, and final track permissions still need to be connected and verified.

## What actually works in this pack

- Local CPU / RAM / GPU discovery report. It does not upload information or download model weights.
- 49 offline unit tests for a bounded task supervisor, validation, and dataset handling. Network calls are mocked in downloader tests.
- A hand-authored goal-state demonstration that requests continue / repair / hold / stop and identifies affected and protected goals.
- Public-dataset metadata inspection and a bounded downloader for one LeRobot v2 episode plus one camera, with revision and SHA-256 provenance.
- Video-frame and recorded-action visualization after a real download.
- A notebook that keeps setup, tests, data inspection, and saved artifacts in this one project folder.

**No VLA model is loaded. No simulated or physical robot is controlled. No public dataset, checkpoint or robot performance result is bundled.** Live download and visualization against public data were not executable in the preparation environment because external network access from Python was unavailable. See `docs/VALIDATION.md` for the exact tests run.

## Start immediately — no GPU and no pip install needed for these steps

Extract the ZIP. Open a terminal **inside `TableGuard_Starter`** (the folder containing this README and the `tableguard` package).

```bash
python -m tableguard hardware
python -m tableguard selftest
python -m tableguard demo
```

Use `python3` instead of `python` where appropriate. Target Python 3.11 for the next integration stage; the supplied pure-Python starter was also tested under Python 3.13 in its preparation environment.

Outputs:

- `artifacts/hardware_report.json` — your local machine, after YOU run the hardware command.
- `artifacts/unit_test_summary.json` — software test counts, not robot-task success.
- `artifacts/logic_fixture_results.json` — synthetic structured fixture decisions; no actual repair is performed.

`artifacts/build_environment_not_user_machine.json` records the preparation runtime only. Do not submit it as your target hardware evidence.

## Create an isolated Phase 1 environment

Do not change a working organizer robotics environment just to inspect a dataset. This preparation environment installs neither PyTorch nor LeRobot, MuJoCo, OpenVINO, nor GPU drivers.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
# Activation is not necessary; avoids PowerShell execution-policy changes.
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\.venv\Scripts\python.exe -m tableguard hardware
```

### Ubuntu / Linux

With Python 3.11 and its venv support already installed:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-phase1.txt
.venv/bin/python -m tableguard hardware
```

If Python 3.11 is not available, establish it through your existing Python/conda tooling; do not modify system Python blindly. For final Intel execution, the linked official installer documents a Python 3.11 `intel_dev_env` on Ubuntu 24.04. Follow that separate approved setup and test the actual policy.

In commands below, `python` means the interpreter from your selected environment. On Windows use `.\.venv\Scripts\python.exe`; on Linux use `.venv/bin/python` unless the environment is activated.

## Inspect public data — metadata first

```bash
python -m tableguard catalog
python -m tableguard fetch --repo lerobot/svla_so101_pickplace
```

Read the saved `download_manifest.json` and `meta/info.json` before requesting video. The default is a small reference, not the final robotics task. Its name contains SO-101 but publisher metadata says `so100_follower`.

Now request **one** episode, **one** camera, and at most 50 MB including downloaded metadata:

```bash
python -m tableguard fetch --repo lerobot/svla_so101_pickplace --episode 0 --media --max-mb 50
python -m tableguard visualize
```

The camera defaults to the first sorted video key in the actual metadata. Set `--camera <exact_camera_key>` to select another. A failed download returns a nonzero exit status and an incomplete manifest; it is not silently replaced with a generated video.

A metadata-only command for the larger simulation reference:

```bash
python -m tableguard fetch --repo gpudad/so101_pick_cube
```

**LeRobot v3 media sampling is intentionally not implemented in this starter.** V3 may put multiple episodes in shared files. The program refuses to guess shard or timestamp offsets. Use the official format-specific loader after pinning a compatible version. The dataset card's older training/import examples are not an approved runtime recipe.

## Public reference selection

| Repository | Publisher-reported data | Use and limitation |
|---|---|---|
| `lerobot/svla_so101_pickplace` | 50 episodes; 11,939 frames; 6-value actions; v2.1; `so100_follower` metadata | Start with a small loading/video inspection example. Single arm, not the final task. |
| `gpudad/so101_pick_cube` | 10,993 episodes; 1,456,901 frames; three camera views; SO-101 MuJoCo cube-to-bin; v3.0 | Optional simulator-domain reference. Still single arm. |
| `lerobot/aloha_sim_transfer_cube_human` | 50 episodes; 20,000 frames; 14-value actions; ALOHA; v2.0 | Optional bimanual format example. Different embodiment and task. |

These public datasets do not supply a tested dual-SO-101 table-setting policy, approved recovery actions, or TableGuard failure labels. Do not merge their raw action vectors or assume downloading more examples creates policy compatibility. Review the data cards and licenses; preserve third-party terms.

## Our actual TableGuard evaluation data

Use the approved challenge scene to record **development data** and **held-out episodes**. Proposed starting design: 8 development configurations; 12 held-out configurations from normal, benign-change, local-violation, and unknown-evidence families. Run 3 compared methods on each of the 12 held-out configurations: 36 attempted episodes.

Those are planning targets, not existing data or a policy-training sample size. Keep official evaluation separate from custom robustness tests. Split by episode/initial configuration, never random adjacent frames. Runtime gets only allowed observations. Evaluator labels, disturbance IDs and privileged simulator object states stay outside the controller and prompts.

## Hardware plan — engineering budgets, not measured requirements

- This starter: no discrete GPU; a 16 GB RAM development computer is a comfortable planning target for small-sample inspection. Pure-Python tests need far less, but this is not a benchmark minimum.
- Compact frozen-policy experiments, if approved and compatible: a single 8–12 GB VRAM GPU is a tentative starting budget, with 16 GB offering more headroom. Actual peak allocation and latency must be measured at the real image sizes, camera count and action chunk settings.
- Optional fine-tuning: a 16–24 GB VRAM starting budget may be useful for a compact model with reduced batch size/appropriate fine-tuning method; it is not a guaranteed fit. Dataset collection and controller compatibility can still dominate. Training is not in the deadline-critical scope.
- Final execution: the supplied challenge says Intel Core Ultra Series 2/3. Targeting 32 GB system RAM is our planning recommendation, not an official minimum. CPU, iGPU and NPU support depends on the actual model, backend, drivers and rules. Shared graphics memory is not dedicated NVIDIA VRAM.

The SmolVLA guide describes a 450M-parameter base model. At two bytes per parameter, weights alone are approximately 0.9 GB decimal; runtime needs additional memory for activations, buffers, caches, other components, and possibly precision/conversion copies. This calculation is NOT a full memory estimate. A generic base model is not automatically compatible with the required task.

## Supervisor integration contract

`tableguard/supervisor.py` accepts explicitly supplied goal observations and returns a decision. It does not infer goals from images. The real visual checker is M3's next integration task.

- Pending placements are not failures.
- Completed or due goals require fresh evidence.
- A runtime-observed change to a dependency invalidates older evidence, but does not automatically mark the goal violated.
- Unknown/stale evidence produces a hold request.
- Exactly one supported violated goal can produce a bounded repair request.
- Multiple/unsupported violations and exhausted budgets stop explicitly.
- The worker increments attempts on actual dispatch; polling the supervisor does not consume attempts.
- `CONTINUE` is not a statement that the entire task is complete.
- `HOLD` and `STOP` are requests. A real adapter must implement the controller's supported boundary/stop mechanism; the supplied stub deliberately raises rather than pretending to move or stop a robot.

To close this loop, M2 must connect a task-compatible policy and valid joint action interface. Never delete one arm's command from a jointly generated chunk or teleport an object to fake recovery.

## Notebook route

Install notebook tools in the chosen preparation environment:

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name tableguard-phase1 --display-name "Python (TableGuard Phase 1)"
python -m jupyter lab
```

Open `notebooks/00_Start_Here.ipynb` and choose the TableGuard kernel. It runs offline hardware and logic checks first. The public download cell is disabled by default; set `DOWNLOAD_PUBLIC_SAMPLE = True` when ready. The notebook saves visualizations after a genuine successful download. It never labels fixture outputs as robot results.

## Immediate five-person assignments (12–15 September, KST)

| Owner | First concrete task | Required evidence |
|---|---|---|
| M1 — you | Run tests; approve one-repair scope; confirm full brief and exact cutoff | Hardware report, requirement decision log, task/repair acceptance criteria |
| M2 — robotics | Obtain approved scene/policy and connect baseline on eligible hardware | Real instruction-to-action run; action dimensions/units, cameras and boundary contract |
| M3 — vision/data | Fetch one reference episode; inspect camera/action schemas; define checker on actual scene | Saved frames, data manifest and supported goal definitions |
| M4 — Intel | Verify exact CPU and permitted machine access; run official stack verifier separately | Exact target configuration and actual model/device execution, not only device discovery |
| M5 — evaluation/demo | Maintain independent test manifest and logs; review outputs | Repeatable checks, media plan and submission checklist |

12 Sep: baseline and hardware decision. 13 Sep: one real supported repair with fresh verification. 14 Sep: matched evaluation and feature freeze by 18:00 KST. 15 Sep: clean-run test and internal submission target 18:00 KST. These are team targets, not a newly verified online cutoff. Any earlier official deadline takes precedence.

## Progress gate

After the commands above, share `artifacts/hardware_report.json` and `artifacts/unit_test_summary.json`, plus the approved scene/policy details when available. Do not buy or rent large GPUs before checking the required target and actual model workload.

Source links and source-status boundaries: `docs/SOURCES.md`. Exact validation status: `docs/VALIDATION.md`.

## Current project status

This repository contains our TableGuard-Lite engineering prototype.

### Demonstrated
- Custom MuJoCo environment with two simulated SO-101 arms.
- Camera rendering and observation/action recording.
- Right-arm cup grasp, lift, transfer, placement, and release.
- Left-arm fork approach and grasp in a development trial.

### Still in progress
- Complete fork lift, transfer, and placement.
- Complete coordinated two-object sequence.
- SmolVLA task-specific training and learned-policy execution.
- Intel/OpenVINO policy deployment and optimization measurements.
- Camera-based goal verification and preservation-aware repair.
- Robustness evaluation and final submission materials.

The current robot demonstrations use scripted engineering controllers.
They are not trained-policy results.

### Reproduction
This initial upload contains source code and notebooks.
The required scene assets and selected recordings must also be provided.
Machine-specific paths need adjustment before running on another computer.
