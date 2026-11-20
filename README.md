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

Use Linux and Python 3.12. Teacher adaptation requires a CUDA-capable GPU. Obtain SAM3 weights through the [official SAM3 project](https://github.com/facebookresearch/sam3) and prepare your dataset separately; weights and datasets are not included.

Follow the [installation and usage guide](docs/SETUP.md) to install dependencies and the pinned, patched SAM3 source.

Prepare four disjoint COCO splits (`train/`, `label/`, `val/`, `eval/`), each containing images and `_annotations.coco.json`, plus a class-definition YAML. Then generate a local configuration:

```bash
python scripts/configure_dataset.py \
  --split-root /data/my_dataset \
  --classes-file /data/classes.yaml \
  --output "$PWD/configs/my_dataset" \
  --work-root /work/ald/my_dataset \
  --teacher-checkpoint /work/ald/my_dataset/teacher-01/teacher/merged_checkpoint.pt
```

Adapt the teacher once, then run Label and Distill:

```bash
python pipeline/teacher_adaptation/run_teacher_adaptation.py \
  --config "$PWD/configs/my_dataset/teacher_adaptation.yaml" \
  --run-name teacher-01

python pipeline/run_pipeline.py \
  --config "$PWD/configs/my_dataset/pipeline.yaml" \
  --run-name ald-01
```

Review the generated training settings before launching. For offline merging, set `sam3.base_checkpoint_path` in the teacher configuration to your local full SAM3 checkpoint. Online experiment tracking is disabled by default.

## Evaluation and validation

Use the [evaluation protocol](pipeline/EVALUATION.md) for comparable metrics.

The repository has passed 28 regression tests and a bounded GPU Adapt → Label → Distill smoke test on RTX 4090. These checks validate execution, not full reproduction of paper results. See the [GPU acceptance report](docs/GPU_ACCEPTANCE_20261120.md) for environment details and limitations.

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
