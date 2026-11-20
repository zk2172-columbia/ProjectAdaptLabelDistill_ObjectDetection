# One-command and container validation

The unified entry point and portable CUDA image passed bounded engineering acceptance on RTX 4090. This validates execution, not paper accuracy or production readiness.

## Tested runtime

- Source: `fae7579c675c534bfa24217417d42a6eeb825975` (runtime code unchanged from `34e6121`; later commits add container permissions, compiler and mount instructions).
- Image ID: `sha256:865cdf54cb757b772c2687eb791c05a58ac59df9e914f789105e3dca1319c2c2`.
- Python 3.12, PyTorch 2.13.0+cu130, TorchVision 0.28.0+cu130, Triton 3.7.1.
- NVIDIA RTX 4090, driver 580.178.04.
- 110 resolved CUDA packages match the pinned constraints.
- 47 regression tests passed inside the final image as a nonroot user, with read-only root filesystem and no network.
- 91 packaged source files matched the inspected checkout byte-for-byte.

## Actual GPU smoke

The same `scripts/run_ald.py` documented in the README ran inside Docker with read-only data and weight mounts, nonroot UID, no network, and a writable output directory. Temporary storage used `/tmp:rw,exec,nosuid,size=1g` for Triton runtime compilation.

The test used synthetic COCO splits with two images each, local SAM3 weights, seed 42, teacher batch 1 and student batch 2. Teacher and student each ran one epoch; student image size was 64. The student used `yolo11n.yaml` (random initialization). Pseudo-label cutoff and confidence gate were zero to exercise the training path. These are smoke-test settings, not recommended accuracy settings.

| Stage | Result | Wall time |
|---|---|---:|
| Adapt | 2 training steps; finite loss; 1,106 checkpoint keys merged, zero skipped | 54.5 s |
| Label | 400 raw detections; 157 retained boxes across 2 images | 8.1 s |
| Distill + evaluate | 1 epoch; finite losses; student checkpoint and held-out metrics exported | 6.6 s |

The pipeline completed in about 71 seconds (74 seconds including container orchestration). Student checkpoint SHA-256: `03971c4aa4ac488a373a41dbc8c1f55a273aefc73da67ce9ea348d6e109fb460`.

Synthetic student AP was zero; this short run establishes wiring and artifact integrity only. COCO values of -1 denote absent area categories. Input hashes and source checkout remained unchanged, and no GPU process remained after completion.

The first container attempt stopped because Triton needed a C compiler. The Dockerfile now supplies `build-essential`; the local installation guide lists the prerequisite. A separate nonroot compile-and-execute probe verified compiler and Python headers before the successful GPU rerun.

## Evidence and limits

The host audit bundle `audit-onecommand-20261120/` retains the manifest, exact container invocation/inspection, source and input hashes, environment reports, all test and stage logs, failed-attempt logs, raw and merged teacher checkpoints, student checkpoint and metrics. These large/private artifacts are not tracked in Git. The image was built locally and was not published to a registry.

The per-stage limit was 900 seconds and the overall limit 1,800 seconds, with one GPU task and current host compute-policy checks. No full training was launched. CPU backend package resolution passed, but a fresh full CPU installation was not repeated in this change. Earlier fixed-split evaluation regression is documented in [GPU acceptance](GPU_ACCEPTANCE_20261120.md); it was not rerun for this wrapper/container-only change.
