# Installation and portable environments

The supported setup targets **Linux x86_64 with Python 3.12**, Git and Python venv support. A minimal server also needs the OpenCV runtime libraries (Debian/Ubuntu: libgl1 and libglib2.0-0); the Docker image supplies these. Choose `cu130` for GPU training or `cpu` for CPU smoke tests/inference. The installer creates a new environment and never modifies an existing one.

## Local installation

From a fresh repository checkout:

```bash
python3.12 scripts/install_environment.py --backend cu130
source .venv-ald/bin/activate
python scripts/run_ald.py --help
```

The installer installs pinned runtime dependencies, reconstructs the pinned SAM3 source plus the repository patch, checks dependencies/imports, and records package versions in the environment directory. Dataset and model-weight downloads are not part of installation.

To inspect the commands without changing files:

```bash
python3.12 scripts/install_environment.py --backend cu130 --dry-run
```

To check package resolution in a fresh environment without installing the ALD runtime:

```bash
python3.12 scripts/install_environment.py \
  --backend cu130 --venv .venv-resolve --resolve-only
```

This creates a disposable environment with installer tooling and a package-resolution report. Some package indexes require downloading wheels to resolve metadata. It does not load model weights, initialize a GPU or train.

For CPU-only checks:

```bash
python3.12 scripts/install_environment.py --backend cpu --venv .venv-cpu-new
source .venv-cpu-new/bin/activate
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output /tmp/ald-smoke-fresh
```

If installation fails, the partial environment and logs remain available for diagnosis. Choose a new directory for the next attempt; the installer never deletes or overwrites one. If `SAM3/` already exists, its recorded upstream revision and patch hash must match this checkout.

## Backend selection

| Backend | PyTorch / TorchVision | Intended use |
|---|---|---|
| `cu130` | 2.13.0+cu130 / 0.28.0+cu130 | CUDA 13.0 GPU training and inference |
| `cpu` | 2.13.0+cpu / 0.28.0+cpu | CPU tests and inference; not teacher training |

CUDA libraries are installed with the PyTorch wheels; a separately installed CUDA toolkit is not needed for this workflow. The host still needs a compatible NVIDIA driver. Consult [PyTorch's official installation matrix](https://pytorch.org/get-started/previous-versions/) for wheel compatibility.

Triton is backend-specific: CUDA PyTorch chooses its matching version; the CPU environment includes the import dependency required by SAM3. The old teacher-wide Triton 3.3 pin is no longer applied to CUDA.

`environments/common.constraints.txt` pins shared dependencies. Backend requirements select Torch/TorchVision; `cu130.constraints.txt` additionally pins the resolved CUDA and transitive package versions. Each installed environment records `ald-environment.json`, `installed-packages.txt` and `packages-report.json`. The historical `verified-cpu.lock.txt` remains unchanged as the record of the earlier CPU audit, not a CUDA installation recipe.

## Docker

The Dockerfile uses an immutable Python 3.12 base digest and the same installer. Build it on Linux x86_64:

```bash
docker build --build-arg ALD_BACKEND=cu130 -t ald:cu130 .
docker run --rm ald:cu130 --help
```

The image includes code and dependencies only. The build context excludes local environments, Git history, datasets, checkpoints, credentials and run artifacts. No model access token is needed during the image build.

GPU containers require the host driver and [NVIDIA Container Toolkit](https://docs.docker.com/engine/containers/gpu/). With trusted SAM3 and YOLO checkpoints already available, run the same unified command with read-only data/weight mounts and writable output:

```bash
mkdir -p "$PWD/runs"
docker run --rm --init --gpus device=0 \
  --user "$(id -u):$(id -g)" --shm-size=2g \
  --read-only --tmpfs /tmp:rw,nosuid,size=1g --network none \
  --mount type=bind,src=/absolute/path/to/data,dst=/data,readonly \
  --mount type=bind,src=/absolute/path/to/weights,dst=/weights,readonly \
  --mount type=bind,src="$PWD/runs",dst=/runs \
  ald:cu130 \
  --split-root /data/my_dataset \
  --classes-file /data/classes.yaml \
  --base-checkpoint /weights/sam3.pt \
  --student-model /weights/yolo11n.pt \
  --output /runs/experiment-01 \
  --teacher-epochs 20 --student-epochs 100
```

Add `--dry-run` to inspect the plan first. Use a fresh output directory for execution. Set epochs and batch sizes for your dataset and compute budget; this example is not a launch authorization or a benchmark recipe. The user mapping keeps generated files owned by the invoking host user.

For offline operation, both checkpoint paths must point to local files. A named YOLO model such as `yolo11n.pt` may download weights on first use when network access is available. `yolo11n.yaml` explicitly trains from scratch.

## Moving to another machine

Recreate a local environment from the same repository commit and backend, rather than copying a virtual environment with absolute paths. For a fully bundled runtime, transfer the built image:

```bash
docker image inspect ald:cu130 --format '{{.Id}}'
docker save ald:cu130 | gzip > ald-cu130.tar.gz
# On the destination:
gzip -dc ald-cu130.tar.gz | docker load
```

Move datasets and trusted checkpoints separately and use the same mounted paths. Preserve each run's manifest and environment report alongside its outputs. Loading the same image preserves its userspace environment; it does not remove host-driver or GPU-memory requirements.

## Validation scope

Build and runtime check results are recorded during repository integration. A successful package resolution or image build does not establish numerical paper reproduction. CUDA training checks must use the target machine's current compute policy and resource availability.
