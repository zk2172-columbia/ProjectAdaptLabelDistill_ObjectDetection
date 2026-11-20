# Private GPU acceptance — 2026-11-20

This dated record supplements the November 18 private candidate report. It certifies bounded engineering checks on RTX 4090, not paper accuracy reproduction or public release readiness.

## Code and execution identity

- Initial audited candidate: `c1c9cfb2e2b2c8b54151c2d3980ef26b84e4090e`.
- Runtime fixes: `cbc81a402ecb94b2ab144d54aa9fc0d04fa88e4f`.
- Adapt ran on the initial candidate; Label/Distill and the final 28-test regression used the runtime-fix commit.
- Runtime fixes add optional `sam3.base_checkpoint_path` for offline merging and remove environment values from upstream training logs. The training algorithm and paper defaults are unchanged.
- The seven-file SAM3 patch reconstructs the pinned upstream files byte-for-byte.
- User redirected execution to 4090 after the configured 5090 SSH endpoint timed out. No 5090 CUDA acceptance is claimed.

## Verified checks

| Check | Result |
|---|---|
| Device/environment | RTX 4090, driver 580.178.04; Python 3.12.3; PyTorch 2.10.0+cu128; torchvision 0.25.0+cu128; CUDA tensor forward/backward finite |
| Compute policy | Revision 1, maximum 3 GPU tasks per host; no compute tasks occupied GPU at launch; this task launched one at a time |
| Unit regressions | 28 passed, including local-base merge and no environment-value logging |
| Adapt | 2 synthetic training images, batch 1, resolution 1008, seed 42, one epoch/two optimizer steps; final validation completed |
| Trained weights | 354 changed tensors; all saved tensors finite |
| Merge | 1,106 keys replaced, zero skipped, saved tensors read back exactly; raw resumable checkpoint preserved |
| Label | Adapted checkpoint produced 400 real detections on 2 synthetic images; conversion retained 160 boxes |
| Distill | GPU YOLO11n from architecture, one epoch, 4 combined GT/PL images, seed 42, batch 2, image size 64; checkpoint export and held-out COCO evaluation completed |
| Weighted-path diagnostic | One additional student epoch, gate 0 and mosaic 0; finite nonzero box loss 0.02034 and DFL loss 0.01308 |
| Fixed-split regression | 1,045-image garbage evaluation fixture, 7,567 GT boxes / 313,500 detections; six saved metrics reproduced exactly; maximum absolute difference 0.0 |

The fixed-split fixture uses ET-control predictions with the same garbage evaluation GT hash recorded for the paper split. It is an evaluator regression, not ALD training or original-paper prediction reproduction. Ground-truth, prediction and reference-metric hashes were unchanged before/after.

The default gate 0.5 suppressed all synthetic pseudo-box weights (maximum confidence 0.055176). The additional gate-0 diagnostic deliberately exercises the positive-weight path; it is not a selected accuracy result or a change to the paper recipe. Both tiny student runs have mAP 0.0, which is not an accuracy acceptance test.

## Explicit deviations and stopped attempts

- A lower-resolution 504 Adapt harness failed at SAM3 rotary-position shape validation before an optimizer step; restoring required resolution 1008 resolved it. The failed log is retained.
- Initial environment inventory used unavailable pip; it was replaced by importlib metadata collection before training.
- A relative pipeline-config argument failed before a stage started; the absolute path was used next.
- The existing CUDA environment lacked declared dependency ijson. Version 3.5.0 was installed into an isolated acceptance overlay; the existing shared environment was not modified.
- Teacher inference was preserved and hash-checked when resuming conversion after the missing-dependency error; no synthetic detections replaced its output.
- Teacher stage timeout: 900 seconds; student stages: at most 900 seconds; extra weighted diagnostic: 300 seconds. No full-training run or multi-seed sweep was launched.

## Artifact identity and retained evidence

- Base SAM3 SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Adapted raw SHA256: `149d3029d16e03b30a2b897eada494e94b2b2a966d908736c2979efea2dc7f08`.
- Merged SHA256: `2e1328e52353fc3a9a00b833e91d37a284e4360fff3453c24da97820b741e0b9`.
- Teacher prediction SHA256: `53266e670157d49909425e0fb821023b691a9f6ed14e69cd0f48b2a90bc22164`.
- Fixed-split GT SHA256: `24c232a13f528dbff8a80194480f1f563b2bab8238992cadd98d8069e8991e62`.
- Fixed-split prediction SHA256: `e76511abf0d4d9c201dc79065ce89e0abe199a4fefdd742634059a4be1327ebc`.

Full commands/job IDs, failed and successful logs, configurations, input hashes, package snapshot, checkpoints and metrics remain in sibling `audit-20261120-gpu/` on the execution host. Raw logs and artifacts are private local material, excluded from Git. No datasets, model weights or credentials are included in this source delivery.

## Private delivery scope

The host's existing GitHub credentials authenticate as zylearncoding and have admin/push permission on the target organization repository. The GitHub App connector's narrower installation scope returned 404; this did not mean the user lacked repository access. The target was verified private and initially had no refs. No visibility change is authorized or performed.

The candidate's reachable history was scanned for forbidden binary/data/credential artifacts and common secret patterns, with no real credential detected. SAM and Ultralytics license texts and attribution remain included; no blanket public license is assigned.

The prior CPU dependency audit does not certify this older diagnostic CUDA environment. This run is not a clean CUDA installation or CUDA vulnerability certification. Fresh paper reproduction, complete dataset/run provenance and public distribution approval remain separate work.
