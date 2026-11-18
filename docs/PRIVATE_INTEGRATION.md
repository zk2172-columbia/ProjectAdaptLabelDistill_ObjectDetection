# Private integration candidate — 2026-11-18

## Source and scope

Anthony's code was identified on 5090 at ProjectPcatcher-NOAA, subtree
Object_detection/Student_Teacher_Pipeline. Parent revision:
`6489372b896a0fb494ba98a426ae7670a0c9eaf1`; recent authorship: `anrunwang`.
The audited working-tree changes were included in the local import. The original remote and
running research jobs were not modified. This candidate uses a fresh local Git history.

The source SAM3 gitlink was not publicly fetchable. Its verified public ancestor is pinned in
`scripts/setup_sam3.py`, with a reviewable patch preserving image-training/logging behavior and
adding CPU device handling and importlib resource loading. Reapplying the patch to pristine
upstream files reconstructs all six modified files exactly.

## Repairs

- Enforce pseudo-label identity, finite values, box counts and class/coordinate matching.
- Handle empty-target weighted loss; verify neutral loss and gradients against stock YOLO.
- Fail required evaluations visibly and reject stale inference outputs.
- Preserve raw checkpoints, reject unmatched merges, verify saved tensors and record hashes.
- Remove power-control side effects; default tracking to offline; validate run/config paths.
- Restore original paper defaults, disable optional masking, and provide dataset configuration.
- Pin a tested environment; repair CPU teacher assumptions and update vulnerable dependencies.

## Validation and limitations

See `results/private_validation.json` for checks and scope. Six original ALD prediction sets
were re-scored on 5090: all six matched their saved metrics exactly with unchanged GT hashes.
Real annotations and prediction payloads stayed on 5090. These are cached-result regressions,
not fresh reproduction of training or every main-table baseline.

CPU smoke tests use synthetic data and validate execution rather than scientific accuracy.
Full teacher fine-tuning, full-scale CUDA student training, optional research variants and video
SAM3 are not certified by these checks. The original paper's dataset split provenance and run
records must accompany any future numerical reproduction claim.

## Staged delivery

1. **Private candidate:** review the local patch/archive, install the pinned environment, run
   unit tests and synthetic smoke checks. No datasets, credentials or checkpoints are bundled.
2. **Private GPU acceptance:** in an isolated workspace, verify CUDA installation, perform a
   bounded Adapt→Label→Distill run, then compare selected original runs using fixed splits,
   hashes, seeds and evaluation settings. Refresh the compute policy and host occupancy first.
3. **Public release, only if later authorized:** choose the original-code license, reconcile
   third-party redistribution obligations, confirm dataset/model rights, finalize citation and
   paper links, review history, and approve publication separately. No public license was assumed.

The target GitHub repository could not be inspected through the connected credentials. Its
current contents and visibility are not certified. Nothing was pushed, merged, published or
made public. Private integration is delivered locally for review/import.
