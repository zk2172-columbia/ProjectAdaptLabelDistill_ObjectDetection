"""Run Adapt -> Label -> Distill from one command, with isolated outputs and logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, required=True,
                   help="COCO train/, label/, val/, eval/ directories")
    p.add_argument("--classes-file", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True,
                   help="Trusted local full SAM3 checkpoint (used for adaptation and merging)")
    p.add_argument("--output", type=Path, required=True, help="Fresh run directory; never overwritten")
    p.add_argument("--student-model", default="yolo11n.pt",
                   help="YOLO model name (may download), local .pt, or .yaml for random initialization")
    p.add_argument("--teacher-epochs", type=positive, default=20)
    p.add_argument("--student-epochs", type=positive, default=100)
    p.add_argument("--teacher-batch", type=positive, default=1)
    p.add_argument("--student-batch", type=positive, default=16)
    p.add_argument("--student-imgsz", type=positive, default=640)
    p.add_argument("--student-amp", action="store_true",
                   help="Enable student AMP checks (may download an additional validation model)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=int, default=0, help="GPU index within currently visible devices")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pseudo-cutoff", type=probability, default=0.25)
    p.add_argument("--zerogate", type=probability, default=0.5)
    p.add_argument("--stage-timeout", type=positive, default=None,
                   help="Optional wall-clock seconds per stage; timeout stops its child processes")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and print the plan; no files written or training started")
    return p


def validate_inputs(args):
    for field in ("split_root", "classes_file", "base_checkpoint", "output"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {args.output}")
    if args.output == ROOT or args.output.is_relative_to(ROOT / "SAM3"):
        raise ValueError("Output cannot overwrite source or the SAM3 package")
    if args.output == args.split_root or args.output.is_relative_to(args.split_root):
        raise ValueError("Output must be outside the source dataset")
    if args.device < 0 or args.workers < 0:
        raise ValueError("device and workers must be nonnegative")
    if args.student_imgsz % 32:
        raise ValueError("student-imgsz must be a multiple of 32")
    for path in (args.classes_file, args.base_checkpoint):
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.split_root.name):
        raise ValueError("Dataset folder name must be a safe path component")
    classes = yaml.safe_load(args.classes_file.read_text())
    classes = classes.get("classes", []) if isinstance(classes, dict) else []
    if not classes or [c.get("id") for c in classes] != list(range(len(classes))):
        raise ValueError("Class YAML must list contiguous zero-based ids in order")
    names = [c.get("name") for c in classes]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Class names must be unique nonempty strings")
    seen = set()
    splits = {}
    categories = None
    for split in ("train", "label", "val", "eval"):
        source = args.split_root / split / "_annotations.coco.json"
        if not source.is_file():
            raise ValueError(f"Missing COCO file: {source}")
        doc = json.loads(source.read_text())
        images = doc.get("images", [])
        filenames = [im["file_name"] for im in images]
        if not filenames or len(set(filenames)) != len(filenames):
            raise ValueError(f"{split}: images must be nonempty with unique filenames")
        if len({im["id"] for im in images}) != len(images):
            raise ValueError(f"{split}: duplicate COCO image IDs")
        if seen.intersection(filenames):
            raise ValueError(f"Overlapping split filenames in {split}")
        seen.update(filenames)
        for name in filenames:
            path = (source.parent / name).resolve()
            if not path.is_relative_to(source.parent.resolve()) or not path.is_file():
                raise ValueError(f"{split}: missing or out-of-split image: {name}")
        current = sorted((c["id"], c["name"]) for c in doc.get("categories", []))
        if categories is None:
            categories = current
        if current != categories or len(current) != len(names) or {name for _, name in current} != set(names):
            raise ValueError(f"{split}: COCO categories must match the class YAML names across splits")
        splits[split] = {"images": len(images), "annotations": len(doc.get("annotations", [])),
                         "annotation_sha256": sha256(source)}
    model = Path(args.student_model).expanduser()
    if model.is_file():
        args.student_model = str(model.resolve())
    elif "/" in args.student_model or "\\" in args.student_model:
        raise ValueError(f"Missing local student model: {model}")
    elif not re.fullmatch(r"yolo11[nslmx]\.(pt|yaml|yml)", args.student_model):
        raise ValueError("Use a local model file or a standard yolo11[nslmx].pt/.yaml name")
    return splits


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_plan(args, splits):
    return {
        "stages": ["adapt", "label", "distill"],
        "output": str(args.output),
        "inputs": {"split_root": str(args.split_root), "classes_file": str(args.classes_file),
                   "base_checkpoint": str(args.base_checkpoint), "splits": splits},
        "settings": {k: getattr(args, k) for k in (
            "teacher_epochs", "student_epochs", "teacher_batch", "student_batch", "student_imgsz",
            "student_model", "student_amp", "workers", "device", "seed", "pseudo_cutoff", "zerogate", "stage_timeout")},
        "student_initialization": (
            "random" if args.student_model.endswith((".yaml", ".yml"))
            else "local pretrained" if Path(args.student_model).is_file()
            else "pretrained (download on first use may be required)"),
        "outputs": {"teacher": str(args.output / "artifacts/run/teacher/merged_checkpoint.pt"),
                    "student_directory": str(args.output / "artifacts/run/weights"),
                    "logs": str(args.output / "logs"), "status": str(args.output / "status.json")},
        "note": "Convenience recipe, not a promise of paper reproduction. Verify your split independence.",
    }


def configure(args):
    config = args.output / "configs"
    work = args.output / "artifacts"
    merged = work / "run/teacher/merged_checkpoint.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts/configure_dataset.py"),
                    "--split-root", str(args.split_root), "--classes-file", str(args.classes_file),
                    "--output", str(config), "--work-root", str(work),
                    "--teacher-checkpoint", str(merged)], check=True)
    def read(name):
        return yaml.safe_load((config / (name + ".yaml")).read_text())
    def write(name, value):
        (config / (name + ".yaml")).write_text(yaml.safe_dump(value, sort_keys=False))
    teacher = read("teacher_adaptation")
    teacher["sam3"]["base_checkpoint_path"] = str(args.base_checkpoint)
    teacher["sam3"]["generated_config_path"] = str(work / "run/teacher/configs/sam3_train.yaml")
    teacher["sam3"]["use_cluster"] = 0
    teacher["sam3"]["num_gpus"] = 1
    teacher["overrides"].update({
        "trainer.model.checkpoint_path": str(args.base_checkpoint),
        "trainer.model.load_from_HF": False, "trainer.model.device": "cpu",
        "trainer.max_epochs": args.teacher_epochs,
        "scratch.max_data_epochs": args.teacher_epochs,
        "scratch.train_batch_size": args.teacher_batch,
        "scratch.num_train_workers": args.workers,
        "scratch.num_val_workers": args.workers,
        "trainer.seed_value": args.seed,
    })
    write("teacher_adaptation", teacher)
    pseudo = read("pseudo_labeling")
    pseudo["sam3"]["device"] = "cuda"
    pseudo["filtering"]["hard_score_cutoff"] = args.pseudo_cutoff
    write("pseudo_labeling", pseudo)
    student = read("student_training")
    student["model"]["architecture"] = args.student_model
    student["training"].update(epochs=args.student_epochs, batch=args.student_batch,
                               imgsz=args.student_imgsz, workers=args.workers, device="0", seed=args.seed,
                               amp=args.student_amp)
    # Disable optional AMP calibration downloads when training a random-initialized model.
    if args.student_model.endswith((".yaml", ".yml")):
        student["training"].update(pretrained=False, amp=False)
    student["experiments"]["gt_pl_perbox"]["zerogate"] = args.zerogate
    student["evaluation"].update(device="0", imgsz=args.student_imgsz, batch=args.student_batch)
    write("student_training", student)
    return config


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def stop_process(process):
    # Descendants may outlive the leader, including after an otherwise normal exit.
    # Always address the owned process group, not just the direct child's state.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_stage(command, log, env, timeout):
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            stop_process(process)
    if code:
        raise RuntimeError(f"Stage exited {code}; see {log}")


def terminate_requested(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        splits = validate_inputs(args)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    plan = make_plan(args, splits)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not (ROOT / "SAM3/sam3/model_builder.py").is_file():
        p.error("SAM3 is not installed. Run the environment installer or scripts/setup_sam3.py first.")
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        p.error(f"CUDA device {args.device} is unavailable; check PyTorch and your GPU runtime.")
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[args.device] if visible else str(args.device)
    env["PYTHONPATH"] = str(ROOT / "SAM3") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(WANDB_MODE="disabled", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", YOLO_CONFIG_DIR=str(args.output / "cache/ultralytics"),
               TORCH_HOME=str(args.output / "cache/torch"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    status = {"status": "preparing", "stages": [], "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "status.json", status)
    previous_sigterm = signal.signal(signal.SIGTERM, terminate_requested)
    try:
        plan["inputs"]["base_checkpoint_sha256"] = sha256(args.base_checkpoint)
        plan["inputs"]["classes_sha256"] = sha256(args.classes_file)
        plan["environment"] = {"python": sys.version, "torch": torch.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device)}
        try:
            plan["git_commit"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            plan["git_commit"] = None
        plan["source_sha256"] = {str(f.relative_to(ROOT)): sha256(f)
                                for folder in ("pipeline", "scripts")
                                for f in (ROOT / folder).rglob("*.py")}
        plan["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                            if d.metadata["Name"]}
        write_json(args.output / "run.json", plan)
        config = configure(args)
        plan["config_sha256"] = {f.name: sha256(f) for f in config.glob("*.yaml")}
        write_json(args.output / "run.json", plan)
        commands = [
            ("adapt", [sys.executable, str(ROOT / "pipeline/teacher_adaptation/run_teacher_adaptation.py"),
                       "--config", str(config / "teacher_adaptation.yaml"), "--run-name", "run"]),
            ("label", [sys.executable, str(ROOT / "pipeline/run_pipeline.py"),
                       "--config", str(config / "pipeline.yaml"), "--run-name", "run", "--stages", "pseudo_labeling"]),
            ("distill", [sys.executable, str(ROOT / "pipeline/run_pipeline.py"),
                         "--config", str(config / "pipeline.yaml"), "--run-name", "run", "--stages", "student_training"]),
        ]
        for name, command in commands:
            record = {"stage": name, "status": "running", "command": command,
                      "log": str(args.output / "logs" / (name + ".log")),
                      "started_at": datetime.now(timezone.utc).isoformat()}
            status["status"] = "running"
            status["stages"].append(record)
            write_json(args.output / "status.json", status)
            print(f"Starting {name}; log: {record['log']}", flush=True)
            start = time.monotonic()
            run_stage(command, Path(record["log"]), env, args.stage_timeout)
            record.update(status="passed", seconds=time.monotonic() - start)
            write_json(args.output / "status.json", status)
        weights = list((args.output / "artifacts/run/weights").glob("*/best_standard.pt"))
        if len(weights) != 1:
            raise RuntimeError("Expected exactly one promoted student checkpoint")
        metrics = weights[0].parent / "coco_metrics.json"
        if not metrics.is_file():
            raise RuntimeError("Student checkpoint exists but held-out evaluation is missing")
        write_json(args.output / "result.json", {
            "student_checkpoint": str(weights[0]), "sha256": sha256(weights[0]),
            "metrics_file": str(metrics), "metrics": json.loads(metrics.read_text()),
            "teacher_checkpoint": plan["outputs"]["teacher"],
        })
        status["status"] = "completed"
        print(f"Completed. Deployable student: {weights[0]}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1]["status"] = status["status"]
        print(status["error"], file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "status.json", status)


if __name__ == "__main__":
    raise SystemExit(main())
