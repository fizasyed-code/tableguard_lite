# TableGuard-Lite
## Verifiable, Safety-Aware Bimanual Physical AI

<p align="center">
  <img src="assets/cover.png" alt="TableGuard-Lite" width="900">
</p>

**TableGuard-Lite** is a dual-arm Physical-AI system for natural-language table setting in MuJoCo using two simulated SO-101 manipulators. It combines **three RGB camera views, 12-joint proprioception, SmolVLA/LeRobot policies, bounded delta-action execution, runtime physical safety guards, traceable evaluation, and Intel/OpenVINO deployment evidence**.

> **Core idea:** the VLA proposes actions; TableGuard decides whether those actions are physically safe to execute.

---

## Current verified status

| Component | Status | Evidence boundary |
|---|---|---|
| Dual SO-101 MuJoCo environment | ✅ Complete | 3 RGB cameras, 12-joint interface |
| Full bimanual reference task | ✅ Passed | Scripted/reference engineering execution |
| Cup reference primitive | ✅ Passed | Grasp → lift → +25 mm transfer → place → release |
| Fork reference primitive | ✅ Passed | Physical grasp → 25 mm table-supported slide → stable release |
| Plate preservation | ✅ Passed in reference task | Plate remains centered/preserved |
| SmolVLA training/inference infrastructure | ✅ Complete | Training, save/reload, CUDA inference, delta actions |
| Learned 59X2 local specialist | ✅ Locally validated | Three-camera collision-recovery / local approach specialist |
| Full learned bimanual task | ⚠️ Not yet verified |
| Intel/OpenVINO deployment proof | ✅ Verified | Earlier checkpoint showed speedup; final 59X2 also exported and ran successfully |
| Final 59X2 OpenVINO export | ✅ Passed | Full export, CPU parity, benchmark, and standalone IR execution completed |

The successful full bimanual demonstration is intentionally labeled **reference execution**, not learned VLA execution.

---

## Task

Given a natural-language table-setting instruction, TableGuard-Lite aims to:

- preserve the centered plate,
- move/place the cup approximately **+25 mm** to the right,
- move/place the fork approximately **25 mm** to the left,
- preserve already-correct objects while manipulating the remaining object,
- stop or hold when execution becomes physically unsafe or insufficiently observed.

The fork reference primitive uses a **table-supported physical slide**. Airborne fork carrying is not claimed.

---

## System architecture

```text
Natural-language instruction
        │
        ├── Top RGB camera
        ├── Left wrist RGB camera
        ├── Right wrist RGB camera
        └── 12-joint proprioceptive state
                        │
                        ▼
                 SmolVLA / LeRobot
                        │
                        ▼
              20 × 12 delta-action chunk
                        │
                        ▼
               Runtime safety supervisor
          ┌─────────────┼─────────────┐
          │             │             │
      joint limits   contact /     object /
      + slew limit   collision     scene guards
          │             │             │
          └─────────────┴─────────────┘
                        │
                        ▼
                Dual SO-101 MuJoCo
                        │
                        ▼
              trace / replay / audit
```

### Control contract

- MuJoCo timestep: **0.005 s**
- Predicted action chunk: **20 × 12**
- Closed-loop replan interval in the learned MuJoCo controller: **50 ms**
- Joint command slew limit: **0.35 rad/s**
- Contact threshold used by verification logic: **0.02 N**

The **50 ms value is the simulator/control replan interval**, not an Intel CPU inference claim.

---

## Reference-task result

A complete bimanual engineering/reference sequence has passed in one continuous MuJoCo episode:

### Right arm — cup
1. approach,
2. physical grasp,
3. lift,
4. translate approximately +25 mm in world X,
5. supported placement,
6. release.

### Left arm — fork
1. approach,
2. physical grasp,
3. table-supported slide approximately 25 mm left,
4. stable force-verified release.

### Final state
- plate preserved,
- cup placement preserved after fork manipulation,
- fork placed and released,
- no hidden object-pose assignment, weld, or external lift is used in the reference execution.

> This is **reference / expert engineering evidence** and is not presented as learned VLA task completion.

---

## Learned SmolVLA evidence

The learned stack uses:

- natural-language instruction,
- top RGB,
- left-wrist RGB,
- right-wrist RGB,
- 12 joint positions,
- SmolVLA / LeRobot,
- state-relative delta joint targets,
- receding-horizon closed-loop execution.

The current strongest learned checkpoint is a **9,901-update local approach / collision-recovery specialist (59X2)**.

### What is verified
- training pipeline works,
- checkpoint save/reload works,
- CUDA inference works,
- three-camera observations are used,
- closed-loop learned control executes in MuJoCo,
- 59X2 is validated as a local specialist in its trained state region,
- runtime safety remains active during learned execution.

### What is not claimed
The full learned bimanual table-setting task has **not yet completed successfully end-to-end**. The latest full learned attempt was stopped by the physical safety guard.

This distinction is deliberate: **no scripted action substitution is presented as learned-policy success**.

---

## Runtime physical safety

TableGuard-Lite uses explicit execution guards rather than sending unconstrained policy outputs directly to the simulator.

Safety mechanisms include:

- joint-range enforcement,
- **0.35 rad/s** command slew limiting,
- unexpected robot-contact detection,
- contact-force checks,
- cup / fork workspace checks,
- plate-displacement preservation,
- cup displacement checks during approach validation,
- source-scene / asset integrity checks,
- requested-vs-applied command logging,
- fail-stop behavior on invalid or unsafe execution.

This turns the system from:

```text
AI predicts action → execute
```

into:

```text
AI predicts action
        ↓
physical authorization
        ↓
execute / hold / stop
```

---

## Offline supervisor evaluation

The repository also contains a deterministic **supervisor-logic evaluation matrix** covering normal, benign-change, local-violation, and unknown-evidence cases.

These evaluations are useful for checking the state machine and goal-preservation logic, but they must **not** be interpreted as 36 successful physical or learned robot episodes.

If the current branch still reproduces the existing offline benchmark, the reported logic-level comparison is:

| Method | Trials | Task Completion Rate | Goal Preservation Rate | Avg Attempts |
|---|---:|---:|---:|---:|
| Baseline VLA logic | 12 | 50.0% | 92.0% | 1.0 |
| Heuristic fallback logic | 12 | 50.0% | 80.0% | 1.5 |
| TableGuard supervised logic | 12 | 100.0% | 100.0% | 1.25 |

> **Important:** this table is an **offline supervisor benchmark**, not an end-to-end learned MuJoCo success rate.

Only keep these numbers in the public README if the current main branch reproduces them with the published evaluation command.

---

## Intel / OpenVINO deployment evidence

TableGuard-Lite includes an OpenVINO deployment path.

### Final 59X2 checkpoint — verified deployment

The final **9,901-update, three-camera 59X2 checkpoint** successfully completed:

- full OpenVINO export,
- export-wrapper / original-policy parity checks,
- validated Transformers mask + KV-cache conversion compatibility,
- OpenVINO Intel CPU inference,
- numerical parity checks,
- paired CPU benchmark,
- standalone saved-IR inference in a fresh process with no PyTorch import,
- source-checkpoint integrity verification.

The benchmark passed functionally, but OpenVINO did **not** improve CPU latency for this final checkpoint. The recorded latency-reduction fraction was negative (`-0.362310948...`), so the final 59X2 OpenVINO CPU path was slower than PyTorch CPU on that benchmark. This result is reported as a **successful deployment / portability result**, not a speedup result.

### Earlier-checkpoint speedup result

A previous TableGuard checkpoint was successfully exported and executed using **OpenVINO on an Intel i7-13700K CPU**.

| CPU inference metric | PyTorch | OpenVINO |
|---|---:|---:|
| Median latency | 936.151 ms | 673.520 ms |
| Estimated p95 | 1039.758 ms | 739.680 ms |
| Measured calls | 12 | 12 |

Additional evidence:

- median speedup: **1.39×**
- median latency reduction: **28.1%**
- numerical comparison: **7 cases from 3 recorded observations**
- maximum joint-target difference: **2.09 × 10⁻⁷ rad**
- standalone exported-model execution: **passed**
- source checkpoint unchanged

### Evidence boundary

The earlier 1.39× speedup belongs to an **earlier TableGuard checkpoint** and must remain labeled that way.

The final 59X2 checkpoint now has its own **successful OpenVINO export, parity, benchmark, and standalone IR execution evidence**, but its CPU benchmark did not show a speedup.

Neither result is a whole-system real-time certification, and the measured CPU inference latencies are not evidence of a ≤50 ms real-time control loop.

---

## Quickstart

The lightweight logic / supervisor tools can be inspected without loading the full VLA model.

```powershell
# Run the unit / logic test suite
python -m tableguard selftest

# Inspect evaluation cases
python -m tableguard eval-cases

# Run the deterministic supervisor benchmark
python -m tableguard benchmark

# Check repository portability
python scripts/verify_reproducibility.py

# Inspect Intel / OpenVINO device availability
python scripts/verify_intel_system.py
```

Before publishing result badges, rerun these commands on the final public branch and update counts to match the actual output.

---

## Repository organization

```text
TableGuard-Lite/
├── tableguard/              # runtime, safety, evaluation and helpers
├── configs/                 # evaluation / task configuration
├── integrations/            # OpenVINO / policy adapters
├── scripts/                 # reproducibility and hardware checks
├── notebooks/               # representative training / rollout / demo notebooks
├── assets/                  # cover and lightweight visuals
├── docs/                    # architecture, evidence boundary, pitch material
├── tests/                   # unit / logic tests
├── README.md
└── pyproject.toml
```

For the public hackathon repo, avoid committing:

- model checkpoints,
- `.safetensors` weights,
- Hugging Face caches,
- local Conda / venv environments,
- raw experiment artifacts,
- large generated videos,
- private machine paths,
- API tokens or `.env` files.

---

## Representative notebooks

For judging, keep a small, readable subset of notebooks that represent the project lifecycle:

1. environment / full reference task,
2. SmolVLA training or fine-tuning,
3. learned closed-loop rollout,
4. local recovery / safety demonstration,
5. Intel/OpenVINO deployment proof,
6. final replay / video export.

The long numbered debugging chain should remain outside the top-level public presentation unless needed for provenance.

---

## Reproducibility and evidence integrity

TableGuard-Lite follows a simple rule:

> **A failed or partial experiment remains labeled failed or partial.**

The project therefore separates three kinds of evidence:

1. **Reference execution** — complete scripted bimanual task.
2. **Learned-policy evidence** — camera-conditioned SmolVLA closed-loop behavior and local recovery.
3. **Deployment evidence** — final 59X2 OpenVINO export/execution is verified; an earlier checkpoint separately demonstrated a 1.39× CPU speedup.

This separation is intentional and prevents the public demo from overstating what has actually been verified.

---

## Current limitations

- Full learned cup+fork completion is not yet verified.
- Final 59X2 OpenVINO export and CPU execution are verified, but the final-checkpoint CPU benchmark did not show an OpenVINO speedup.
- Earlier Intel/OpenVINO CPU latency is not sufficient for a 50 ms real-time loop.
- The validated fork primitive is a table-supported slide rather than airborne carry.
- Additional held-out object / visual variations would be needed for a broader generalization claim.

---

## Hackathon positioning

**TableGuard-Lite is not just a robot policy — it is a verifiable Physical-AI execution stack.**

The project focuses on the gap between:

> “the AI generated an action”

and

> “the action was physically authorized, observable, bounded, and verifiable.”

---

## License

Add the repository's actual license here. Do not link to `LICENSE` until that file is present in the public repository.
