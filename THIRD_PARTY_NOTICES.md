# Third-party components and license scope

Original ALD material is licensed under the [MIT License](LICENSE), except for
the components listed below. MIT does not replace upstream licenses or grant
rights in third-party code, models or datasets. Copyright remains with the
respective contributors and upstream rights holders.

| Component / repository path | Source | Applicable terms |
|---|---|---|
| Original ALD orchestration, configuration, evaluation, data preparation, documentation and tests outside the exceptions below | ALD contributors; see historical [source provenance](results/source_provenance.json) | MIT |
| All material in `pipeline/student_training/` and `pipeline/baselines/stac/` | ALD extensions to [Ultralytics](https://github.com/ultralytics/ultralytics); pinned dependency versions are in `environments/` | AGPL-3.0; each directory includes the full license text |
| `patches/sam3-image-training.patch` and resulting modified SAM3 source | [Meta SAM3](https://github.com/facebookresearch/sam3), base `f6e51f59500a87c576c2df2323ce56b9fd7a12de` | [SAM License](LICENSES/SAM-LICENSE.txt), including ALD modifications |
| SAM3-derived configuration templates: `experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml`, `pipeline/evaluation/sam3/configs/pcatcher_eval.yaml`, `pipeline/evaluation/sam3/configs/pcatcher_eval_pretrained.yaml` | Meta SAM3 training/evaluation templates, adapted by ALD contributors | [SAM License](LICENSES/SAM-LICENSE.txt) |
| Ultralytics dependency and YOLO models | [Ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](LICENSES/Ultralytics-AGPL-3.0.txt), unless separately licensed by its rights holder |
| faster-coco-eval and other installed dependencies | Respective upstream packages; versions in `environments/` | Their own licenses and notices; not relicensed by this repository |

## Ultralytics-derived extensions

Ultralytics upstream notice: **Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license**.
The student and STAC data, loss and trainer implementations adapt upstream
dataset/augmentation, detection-loss and training code. ALD modifications add
confidence weighting, pseudo-label handling, STAC-style weighting, validation
and orchestration. These directories are distributed under AGPL-3.0, not MIT.
This distribution records those modifications on 2026-11-23; source history
records individual changes. Full corresponding source for these extensions is
included here; the upstream dependency source is linked above.

The MIT license for original modules does not remove AGPL obligations when
those modules are combined with Ultralytics into a covered work, including
applicable source-availability obligations for distribution or remote network
interaction. This repository is a mixed-license distribution, not an MIT-only
license for the complete training stack.

## SAM3 modifications

The patch modifies six upstream files for image training, logging, CPU device
handling and resource loading. It is distributed under the SAM License together
with its full license text. `scripts/setup_sam3.py` records the immutable base
and applies the patch. SAM3 source and model weights are fetched separately.
The three SAM3-derived configuration templates listed above also retain the SAM License. Preserve the SAM License with any redistributed patch, template or modified SAM3 source.

## Research baseline, datasets and model weights

The STAC directory is a local STAC-style implementation built on Ultralytics,
not the official [Google Research STAC implementation](https://github.com/google-research/ssl_detection)
and not a claim of code or numerical equivalence.

Datasets, annotations and checkpoints are not bundled. Obtain them from their
providers and comply with their access and license terms. MIT does not grant
rights to external data or weights.

Historical private-integration reports and `results/source_provenance.json`
describe the earlier preparation stage. Their authorization and visibility
fields are historical, not the current release status.
