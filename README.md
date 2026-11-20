# ALD — Adapt, Label, Distill

> [!IMPORTANT]
> ## TL;DR
> **Train a compact object detector for deployment in your domain—even with limited labeled data.**
>
> Adapt a SAM3 teacher on a small labeled set, use it to label additional images, then train a lightweight YOLO11 student. Deploy only the student.

1. **Adapt:** fine-tune a SAM3 teacher on labeled examples.
2. **Label:** generate and filter pseudo-labels with the adapted teacher.
3. **Distill:** train a YOLO11 student with ground-truth labels and confidence-weighted pseudo-labels.

Only the student is needed at inference time.

## Getting started

Use **Linux x86_64, Python 3.12 and an NVIDIA GPU**. Install into a fresh environment:

```bash
python3.12 scripts/install_environment.py --backend cu130
source .venv-ald/bin/activate
```

Prepare four disjoint COCO splits (`train/`, `label/`, `val/`, `eval/`), a class-definition YAML and a trusted local SAM3 checkpoint. See the [data guide](docs/SETUP.md#configure-a-dataset). Weights and datasets are not included.

### Train with one command

```bash
python scripts/run_ald.py \
  --split-root /data/my_dataset \
  --classes-file /data/classes.yaml \
  --base-checkpoint /weights/sam3.pt \
  --output "$PWD/runs/my_dataset" \
  --teacher-epochs 20 --student-epochs 100
```

This runs **SAM3 adaptation → pseudo-labeling → YOLO training → evaluation**. The default student is pretrained YOLO11n (downloaded on first use); pass `--student-model /weights/yolo11n.pt` to use local weights.

- Add `--dry-run` to check inputs and inspect the plan without writing files or starting training.
- Choose epochs and batch sizes for your compute budget. Existing output directories are never overwritten.
- Find the final student checkpoint in `result.json`, progress in `status.json` and stage logs under `logs/` inside the output directory.

For Docker, CPU setup and moving the runtime to another machine, see [portable environments](docs/ENVIRONMENT.md). For separate stage commands and advanced settings, see [installation and usage](docs/SETUP.md).

## Evaluation and validation

Use the [evaluation protocol](pipeline/EVALUATION.md) for comparable metrics.

The repository has passed 47 regression tests and a bounded GPU Adapt → Label → Distill smoke test on RTX 4090. These checks validate execution, not full reproduction of paper results. See the [GPU acceptance report](docs/GPU_ACCEPTANCE_20261120.md) for environment details and limitations.

After installation, run:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output /tmp/ald-smoke-01
```

Use a fresh output directory. This CPU smoke test uses generated images and cached synthetic teacher detections.

## Citation

If you use ALD in your research, please cite our paper. The BibTeX below is a **placeholder**: the title follows the current manuscript; authors and publication details will be finalized before release.

```bibtex
@misc{ald,
  title  = {Adapt, Label, Distill: Adapting {SAM3} as an Offline Teacher for Compact Object Detection},
  author = {TODO: confirmed author list},
  year   = {TODO},
  note   = {TODO: publication or preprint details},
  url    = {TODO}
}
```

## License

> [!NOTE]
> **Project license: TBD.** A license for the original ALD code will be added before public release. This placeholder does not grant a license.

Third-party components retain their respective licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md) and [license texts](LICENSES/). Dataset and model-weight access remains subject to the respective providers' terms.

## Acknowledgments

ALD builds on [SAM3](https://github.com/facebookresearch/sam3) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Source provenance is documented in the [integration notes](docs/PRIVATE_INTEGRATION.md).
