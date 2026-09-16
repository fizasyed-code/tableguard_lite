# TableGuard-Lite — Phase 2: scene I/O and policy connection

You have already run the Phase 1 commands. This update adds real MuJoCo scene
loading, model inventory, RGB image capture, explicit action/observation contracts,
and a guarded local-policy integration loop. It preserves the original package.

**This is not a finished hackathon entry.** No official scene, robot meshes,
pretrained checkpoint, visual detector, official scorer, or working recovery policy
is included. The package does not substitute toy arms or prerecorded actions for
the required dual-SO-101 table-setting task.

## 1. Install this update in the same project

Extract `TableGuard_Phase2_Scene_Integration.zip` INSIDE `TableGuard_Starter`.
The resulting path should be `TableGuard_Starter/TableGuard_Phase2_Update/`.
From the terminal already opened in `TableGuard_Starter`, run:

```text
python TableGuard_Phase2_Update/apply_phase2.py --target .
python -m tableguard.phase2 --help
python -m tableguard.phase2 preflight
```

The installer adds new files only. It refuses conflicting existing files, checks
payload hashes, and does not alter your Phase 1 reports, supervisor, datasets,
Python environment, or other code. Identical files are skipped on a second run.

A preflight exit code of 2 with `missing_scene_dependencies` is a dependency
report, not a failed hackathon episode. The report is saved anyway:
`artifacts/phase2_preflight.json`.

### Python environment

Prefer the environment required by the actual organizer task/policy. The linked
Intel guide describes Ubuntu 24.04 and Python 3.11. That does not establish what
hardware you personally have, and the shared guide is not the missing full brief.

If there is no established policy environment yet, the following adds only the
scene tools to your selected Python environment:

```text
python -m pip install -r requirements-phase2.txt
python -m tableguard.phase2 preflight
python -m tableguard.phase2 selftest
```

The requirements file contains compatibility ranges, not a verified official lock.
It does not install Torch, LeRobot, OpenVINO, drivers, or a policy. Do not upgrade a
working approved policy stack to satisfy this helper; align the helper with it.
After a working installation, record `python -m pip freeze` in your environment log.

Optional device discovery, after framework installation:

```text
python -m tableguard.phase2 preflight --probe-devices
```

This is discovery only. Final Intel model inference must be demonstrated separately.

## 2. Obtain the actual assets — do not guess them

The event capture requires two simulated SO-101 arms in MuJoCo, natural-language
instructions, camera reasoning, coordinated multistep table setting and execution
on Intel Core Ultra Series 2/3. It links to a fuller brief but does not provide a
named scene/checkpoint in the text we have.

Place the organizer-supplied or explicitly permitted complete asset bundle under
`assets/challenge/`. Keep meshes, include files, textures and relative paths intact.
You may instead use an external asset directory and pass that path to the commands.
Only compile trusted files: MJCF can reference native plugins and local assets.

```text
python -m tableguard.phase2 scan --directory assets/challenge
```

Output: `artifacts/phase2_scene_candidates.json`.
A `<mujoco>` XML root is only a candidate, NOT proof of the correct entrypoint.
Ask for the actual task launch instructions. No file found means no file found:
the tool does not silently generate a replacement scene.

**When the official environment owns reset/camera/control through its own API,
use that environment's API for task execution.** A raw MJCF reset is only a scene
smoke test; do not bypass prescribed initialization, task logic or scoring.

## 3. Inspect the chosen entrypoint

Replace `ACTUAL_ENTRYPOINT.xml` below with the real relative or absolute path from
the task bundle. The name is a placeholder, not a supplied asset.

```text
python -m tableguard.phase2 inspect --scene "assets/challenge/ACTUAL_ENTRYPOINT.xml"
```

Output: `artifacts/phase2_scene_inventory.json` with actual cameras, body tree,
joint types, actuator names/control ranges, keyframes and timestep. No motion is
executed. `nq`, `nv`, `nu` are different quantities; full `qpos` can include objects.
Do not assume the policy vector has 6, 12, or 14 dimensions from a dataset name.
This inventory is developer metadata, not an input to the vision policy.
The recorded hash covers the entry XML only; pin the asset repository revision
and record included asset provenance separately.

## 4. Capture real camera images

```text
python -m tableguard.phase2 capture --scene "assets/challenge/ACTUAL_ENTRYPOINT.xml"
```

Default: every fixed camera in the scene, 640 x 480, default MJCF initial state.
Use the task-required camera subset/resolution for policy integration. Explicit
options use ACTUAL names from the inventory:

```text
python -m tableguard.phase2 capture --scene "assets/challenge/ACTUAL_ENTRYPOINT.xml" --camera "ACTUAL_CAMERA_NAME"
```

Add `--keyframe "ACTUAL_KEYFRAME_NAME"` only when required by the task. No keyframe
is guessed. Capture uses `mj_forward` to compute initial geometry and performs zero
physics steps and zero policy calls. It is an image-I/O smoke test, not a completed task.

Each run gets a new folder in `artifacts/phase2_captures/`, with PNG files and
`capture_report.json`. The latest report pointer is
`artifacts/phase2_latest_capture.json`. The report says `task_success: null`.

### Rendering troubleshooting

On a normal desktop, try the platform default first. For Linux without a display,
EGL or OSMesa requires the corresponding OpenGL libraries/driver support. For example:

```text
python -m tableguard.phase2 capture --scene "assets/challenge/ACTUAL_ENTRYPOINT.xml" --gl egl
```

`--gl osmesa` is another Linux option when correctly installed. Do not use Linux
backend advice blindly on Windows. Restart the Python kernel after changing the
rendering backend. Do not treat a blank image or an import success as camera readiness.
Inspect the saved images yourself. A selected-camera error should be fixed using
actual inventory names, not by replacing the camera with an unrelated free view.

## 5. Configure the policy interface AFTER inspection

```text
python -m tableguard.phase2 init --scene "assets/challenge/ACTUAL_ENTRYPOINT.xml"
```

This creates `configs/phase2_task_contract.json` only if it does not already exist.
Fill the fields from the actual brief/assets/controller:

- Source reviewed and scene/policy/observation permissions. Review flags are your
  attestations, not independent organizer approval. Never set them just to run code.
- Two distinct robot base bodies and permitted scalar joint names under each.
- Allowed camera names/resolution and documented reset keyframe, if any.
- Every actuator's name in the order produced by your adapter, native control units,
  control-contract reference and the documented physics steps per action.
- Actual implementation/checkpoint revision and backend/device declaration.
- A trusted local policy factory, e.g. `integrations.approved_policy_template:create_policy`
  ONLY after you implement that file with the REAL policy API.

```text
python -m tableguard.phase2 check --with-scene
```

Unfilled fields cause explicit errors. This is intentional. Passing validates the
written mapping, not policy quality, robot identity, licensing or eligibility.

## 6. The exact connection point

Your implementation in `integrations/approved_policy_template.py` must provide:

```python
from tableguard.phase2.bridge import Action, Observation

# factory(settings: dict) -> object implementing these methods:
# reset(instruction: str) -> None
# select_action(observation: Observation) -> Action
```

`Observation` contains the actual instruction, frame ID/timestamps, RGB images,
and ONLY the explicitly selected scalar robot-joint positions/velocities.
It does not include full qpos, object poses, disturbance labels or evaluator case IDs.
This API separation is not a Python security sandbox; trusted adapter code can do
anything its process permissions allow, so audit the implementation.

The selected policy adapter must perform its own documented preprocessing,
normalization, model inference, action-chunk handling and decoding. Return:
`Action(controls=...)` in native MuJoCo actuator units and the configured order.
Normalized model outputs cannot be written to `data.ctrl` unchanged unless the
actual controller contract establishes that equivalence.

At a valid policy stop, return `Action(stop_requested=True, reason="...")` with no
controls. A stop is not scored as successful completion. Reset all model memory and
cached chunks between episodes. The default template raises NotImplementedError;
there is no hidden constant-action or replay policy.

When—and only when—that real connection is implemented and the contract passes:

```text
python -m tableguard.phase2 run --instruction "THE ACTUAL SUPPORTED INSTRUCTION" --trust-policy-code
```

This runner captures fresh RGB, queries the actual policy, rejects invalid controls,
steps MuJoCo and saves frame-linked actions and outcomes. It never writes object
poses to claim a successful repair. It stops on invalid values, new MuJoCo warnings,
or configured budgets. It is not a collision-avoidance or safety-certified controller.
Wall budgets are checked BETWEEN blocking operations; they cannot interrupt a hung
model call. This code is simulation-only, not for physical robot actuation.

Outputs: `results/phase2/baseline_probe_<timestamp>_<id>/`, including policy/config
provenance, actual local hardware, frames, events and run report. `task_success`
and `recovery_success` remain null until a separate permitted evaluator is attached.
The declared backend is not independently verified by this runner.

## 7. What is deliberately not integrated yet

Phase 1's supervisor remains tested, but this baseline runner does NOT call it.
First demonstrate a genuine task-compatible baseline. Then connect a real visual
monitor, supported task-boundary events, bounded repair dispatch and independent
scoring. Do not infer task completion from a model's stop signal.

```text
Now:      actual assets -> inspect -> camera capture -> policy contract -> baseline
Next:     real goal checker -> supported repair -> fresh verification -> fair tests
```

For this project's internal schedule, September 12 targets baseline integration,
September 13 one supported repair, September 14 evidence and feature freeze,
September 15 final checks/submission. These are INTERNAL targets; the exact online
cutoff and full partner rubric still need written confirmation.

## 8. Test status and immediate handoff

See `docs/PHASE2_VALIDATION.md`. The offline suite checks schema rejection, explicit
actuator mapping, selected-joint access, file preservation and existing supervisor
behavior. Optional engine tests use a clearly labeled two-joint engineering fixture,
NOT robot arms and NOT the official scene:

```text
python -m tableguard.phase2 selftest --with-mujoco
```

Do not use `tests/phase2_fixtures/engineering_only.xml` for a hackathon demonstration.
Camera rendering must be tested with `capture` and the real permitted scene.

Next handoff: `artifacts/phase2_preflight.json`, the real scene inventory, and the
complete challenge brief plus actual policy/scene launch instructions. No passwords,
API tokens, private keys, or large model weights are needed for that review.
