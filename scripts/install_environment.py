"""Install ALD into a new Python 3.12 environment without changing shared packages."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "f6e51f59500a87c576c2df2323ce56b9fd7a12de"


def validate_source(root):
    source = root / "SAM3"
    if not source.exists():
        return False
    metadata = source / ".ald-source.json"
    if not metadata.is_file():
        raise ValueError("Existing SAM3 has no ALD provenance. Use a fresh clone; it will not be overwritten.")
    record = json.loads(metadata.read_text())
    patch_hash = hashlib.sha256((root / "patches/sam3-image-training.patch").read_bytes()).hexdigest()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if record.get("base_sha") != BASE or head != BASE or record.get("patch_sha256") != patch_hash:
        raise ValueError("Existing SAM3 does not match the pinned ALD source/patch. Use a fresh clone.")
    return True


def installation_commands(root, target, backend, resolve_only, have_source, interpreter):
    python = str(target / "bin/python")
    requirement = str(root / "environments" / ("requirements-" + backend + ".txt"))
    commands = [[interpreter, "-m", "venv", str(target)]]
    # Pin installer/build tooling too, in the new environment only.
    commands.append([python, "-m", "pip", "install", "pip==26.2", "setuptools==84.1.0", "wheel==0.46.3"])
    install = [python, "-m", "pip", "install", "-r", requirement,
               "--report", str(target / "packages-report.json")]
    if resolve_only:
        install.extend(["--dry-run", "--ignore-installed"])
        return commands + [install]
    commands.append(install)
    if not have_source:
        commands.append([python, str(root / "scripts/setup_sam3.py")])
    commands += [
        [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(root / "SAM3")],
        [python, "-m", "pip", "check"],
        [python, "-c", "import torch,torchvision,ultralytics,ijson; from sam3.model_builder import build_sam3_image_model; print('ALD imports passed; torch='+torch.__version__+' CUDA='+str(torch.version.cuda))"],
    ]
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "cu130"), default="cu130")
    p.add_argument("--venv", type=Path, default=ROOT / ".venv-ald", help="New directory only; never reuses an existing environment")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print commands without creating files or downloading packages")
    action.add_argument("--resolve-only", action="store_true", help="Create a fresh environment and resolve package metadata, without installing the ALD runtime")
    args = p.parse_args()
    target = args.venv.expanduser().absolute()
    if target.exists() or target.is_symlink():
        p.error(f"Refusing to modify existing environment: {target}")
    if sys.version_info[:2] != (3, 12):
        p.error("Run this installer with Python 3.12.")
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        p.error("These pinned environments target Linux x86_64.")
    try:
        have_source = False if args.resolve_only else validate_source(ROOT)
    except (ValueError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    commands = installation_commands(ROOT, target, args.backend, args.resolve_only, have_source, sys.executable)
    for cmd in commands:
        print("+ " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)
    freeze = subprocess.check_output([str(target / "bin/python"), "-m", "pip", "freeze"], text=True)
    (target / "installed-packages.txt").write_text(freeze)
    manifest = {
        "backend": args.backend,
        "python": sys.version,
        "mode": "resolve-only" if args.resolve_only else "installed",
        "requirements_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "environments").glob("*.txt"))},
        "sam3_base": BASE,
        "sam3_patch_sha256": hashlib.sha256((ROOT / "patches/sam3-image-training.patch").read_bytes()).hexdigest(),
        "gpu_execution_verified": False,
    }
    (target / "ald-environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Resolved package metadata." if args.resolve_only else f"Ready: source {target}/bin/activate")


if __name__ == "__main__":
    main()
