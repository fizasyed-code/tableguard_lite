# Validation record — 12 September 2026

## What was executed in the assistant's build environment

- `python -m tableguard.phase2 selftest`: 100 tests discovered; 96 passed,
  4 explicit MuJoCo engine tests skipped; 0 failures and 0 errors.
- The original 49 Phase 1 offline tests are included in that count.
- Added offline tests cover invalid/duplicate actuator mapping, action shape,
  NaN/infinity/range rejection, selected-joint-only observation access, rejection
  of free-object joints, invalid/missing review fields, camera/keyframe names,
  non-destructive contract creation, XML scanning and run-directory isolation.
- All Python source files compiled successfully.
- The notebook's offline preflight/tests/empty-asset-scan path was executed;
  camera and policy cells are intentionally skipped without an actual scene.
- The updater was checked against a fresh extraction of the original starter:
  it added files, preserved all original file bytes, skipped identical files on
  rerun and refused conflicting files without overwriting them.

## What was NOT executed here

MuJoCo was not installed in the build environment. An attempted dependency install
failed because network/DNS access was unavailable. Accordingly, no claim is made
that the MuJoCo loader, camera renderer, or full local-policy runner was executed
here. Four real-engine API tests are opt-in and were skipped, not counted as passes.
The optional fixture is a synthetic two-joint engineering model, NOT SO-101 and
NOT table setting. No model inference, Intel device execution, task success,
selective robot repair, or independent benchmark was run.

The implementation uses documented MuJoCo Python APIs. It still needs validation
with the exact task assets and policy environment. The requirement ranges are not
an environment lock or a guarantee of compatibility with every MuJoCo version.

## What you should verify locally

1. Current Python/dependencies and exact processor using phase2 preflight.
2. The actual permitted scene compiles and lists the expected cameras/joints.
3. Saved RGB images show the real approved scene and appropriate camera views.
4. The written native-control/observation mapping matches the actual policy.
5. Real policy inference and coordinated execution, with independent scoring.

A zero CLI exit status is not a task-success score. Where no scorer is attached,
`task_success` and `recovery_success` remain JSON null.
