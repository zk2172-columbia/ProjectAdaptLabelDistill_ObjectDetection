# data

Dataset splitting, the exported ground truth, and the per-dataset class lists.

Splits a source dataset into four parts — `train` (the labeled set, ~5%), `label` (the unlabeled
pool the teacher annotates), `val` (checkpoint selection) and `eval` (the held-out test split) —
then exports `eval`/`train`/`val` into the fixed COCO + YOLO ground truth everything downstream is
scored against. Run once per dataset, before anything else.

| Path | Contents |
|---|---|
| `split_dataset.py` | Keeps a source dataset's preset val/test splits and re-partitions its train set |
| `prepare_ground_truth.py` | Exports finished splits into `ground_truths/<dataset>/` |
| `coco_to_yolo.py` | Standalone COCO → YOLO label converter |
| `classes/<dataset>.yaml` | Class list; the order here is the literal YOLO class index |
| `ground_truths/<dataset>/` | Exported GT: `{train,val,eval}/annotations.coco.json`, YOLO labels, `dataset.yaml`, `manifest.json` |

## Usage

Edit `split_config.yaml` — the three source annotation files, `out_dir`, `shots`, `train_frac`,
`seed` — then:

```bash
python split_dataset.py --config split_config.yaml
```

Its `ground_truth_export:` block runs the export as a post-step. To re-export on its own:

```bash
python prepare_ground_truth.py \
    --splits-root ../../experiment_files/finetuning-data/<dataset>-splits \
    --out-dir ground_truths \
    --classes-file classes/<dataset>.yaml
```

Add a new dataset's class list as `classes/<dataset>.yaml`:

```yaml
classes:
  - id: 0
    name: plastic
  - id: 1
    name: metal
```
