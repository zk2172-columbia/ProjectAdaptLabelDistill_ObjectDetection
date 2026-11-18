# teacher_adaptation

Stage 1 — finetune SAM 3 on the small labeled set so it becomes a target-domain annotator.

20 epochs at 1008×1008, batch 2, AdamW at 8e-5, vision and language backbones frozen and only the
detection transformer head and class embeddings trained. Three steps: generate a per-run SAM 3
Hydra config, run SAM 3's own trainer, then merge the finetuned weights into the HuggingFace base
checkpoint.

**In:** a split under `experiment_files/finetuning-data/<dataset>-splits/`
**Out:** `experiment_files/experiments/<run-name>/merged_checkpoint.pt`

This stage has its **own `--run-name`**, separate from a pipeline run: adapting the teacher costs
GPU-hours and is done rarely, while the later stages are re-run many times against one teacher.

## Usage

Set `dataset:` in `config.yaml` to a folder under `experiment_files/finetuning-data/`, then:

```bash
python run_teacher_adaptation.py --config config.yaml --run-name ft-sam3-<dataset>
```

Run a single step against an existing run:

```bash
python run_teacher_adaptation.py --config config.yaml --run-name ft-sam3-<dataset> \
    --steps merge_checkpoint
```

Copy the resulting `merged_checkpoint.pt` path into `pseudo_labeling/config.yaml`'s
`sam3.checkpoint_path`.

Paths for SAM 3 (`paths.bpe_path`, `paths.roboflow_vl_100_root`) are set in `config.yaml`'s
`overrides:` block, relative to this directory, and made absolute when the Hydra config is
generated.
