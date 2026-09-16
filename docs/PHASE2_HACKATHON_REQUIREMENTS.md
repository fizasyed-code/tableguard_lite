# Requirement-to-implementation status — 12 September 2026

Source [U] is the user's supplied event capture, not a guessed replacement brief.
The public page supplements the record; the linked full online brief was not readable
through the currently available connected sources. A missing detail is not permission.

| Stated online requirement | Current Phase 2 support | Still to demonstrate |
|---|---|---|
| Online team, 1–5 members, one track [U:67–82] | Documentation only | Actual platform registration and correct track |
| Two simulated SO-101 arms in MuJoCo [U:250–252] | Real scene loader, metadata inspection, explicit two-subtree mapping | Actual permitted robot assets and provenance; names alone are insufficient |
| Interpret natural-language instructions [U:250] | Runner passes the exact instruction to a supplied local policy | Real task-compatible instruction-conditioned policy; no model included |
| Reason over camera observations [U:250] | RGB capture and image-only/selected-robot-state observation interface | Actual model/visual checker; no hidden object-state inputs |
| Coordinate both arms; multistep table setting [U:250] | Full named control vector submitted atomically to MuJoCo | Actual control semantics, joint policy and independent official-task scoring |
| Run on Core Ultra Series 2/3; OpenVINO named [U:250–258] | Current-machine report; explicit device/implementation provenance fields | Eligible machine and actual required model execution path; discovery does not prove it |
| Original, MIT-compliant entry [U:418] | New helper code is MIT licensed, inherited license retained | Separate audit of reused models, meshes, data and software; do not relicense third-party assets |
| Text/tags, cover, video, slides [U:442–455] | Not generated as submission assets in this phase | Real evidence from the frozen application |
| Public repository, demo platform and application URL [U:457–461] | Local trace files only | Accepted local-simulation access arrangement and accessible repository |
| General criteria + partner rubric [U:463–481] | Traceable logs as infrastructure for evidence | Full partner rubric and honest measured results |

Do not import the separate on-site defect-detection/Anomalib application as a new
online task obligation. Shared installation documentation is guidance, not proof
that every package it names is mandatory for every scored online component.

## Internal timeline, not an official cutoff

- 12 September: actual environment, scene, cameras and baseline route.
- 13 September: one supported repair and fresh visual verification, if baseline works.
- 14 September 18:00 KST: internal feature freeze and evidence audit.
- 15 September 18:00 KST: internal submission target, before 16 September.

[U:556] lists an event-wide end-of-submissions time of 17 September 03:30 KST;
its online applicability is not established. Any earlier confirmed cutoff overrides
our internal targets. No official weights or scene/checkpoint names are invented.
