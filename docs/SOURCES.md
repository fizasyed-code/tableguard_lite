# Sources and boundaries

Checked 12 September 2026. Web cards/documentation establish the statements below; GPU/RAM ranges are explicitly our engineering planning budgets, not official or measured minima.

1. User-supplied `Pasted markdown(20260910-133349).md`, online task lines 250–258: two simulated SO-101 arms, MuJoCo, natural-language instructions, camera reasoning, coordinated table setting, Intel Core Ultra Series 2/3. Source is the supplied capture; the full partner brief has not been retrieved. Public event banner: https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon
2. Intel installation and functional tests: https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html — Ubuntu 24.04 / Python 3.11 environment. Shared installation guide is not proof that all packages are mandatory for the online track.
3. SmolVLA architecture, 450M parameter base model, and setup-specific fine-tuning guidance: https://huggingface.co/docs/lerobot/smolvla
4. SmolVLA authors' paper: https://arxiv.org/abs/2506.01844 — compact model and consumer hardware/CPU deployment design; not a benchmark on our task or machine.
5. Small dataset card: https://huggingface.co/datasets/lerobot/svla_so101_pickplace — v2.1, 50 episodes, 11,939 frames, robot_type so100_follower, six action entries. Not dual-arm table setting.
6. Simulation reference: https://huggingface.co/datasets/gpudad/so101_pick_cube — publisher reports SO-101 MuJoCo cube-to-bin, 10,993 episodes and 1,456,901 frames. Single arm, not the final task.
7. Bimanual reference: https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human — ALOHA, 50 episodes, 20,000 frames, 14-value action vectors. Different embodiment.
8. Hub download documentation: https://huggingface.co/docs/huggingface_hub/guides/download
9. Single-file metadata, revision and download API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
10. MuJoCo overview: https://mujoco.readthedocs.io/en/stable/overview.html

This pack contains no public dataset media or model weights. Dataset license tags are provenance, not permission to relabel or remove original conditions. The generated source files may be adapted for the project; review all components under the hackathon's actual reuse rules.
