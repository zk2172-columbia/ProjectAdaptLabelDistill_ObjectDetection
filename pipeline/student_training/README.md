# Distill

Use the root [README](../../README.md) for setup and the supported dataset configuration helper.
The default original ALD arm is `gt_pl_perbox`, combined GT/PL loader, lambda1 0.5 and gate 0.5.
GT boxes receive weight 1; PL boxes use validated class-and-coordinate confidence matches.
Missing, duplicate or ambiguous metadata fails instead of silently reverting to full supervision.

Gated boxes remain in assignment; anchors assigned to them receive zero weighted loss. Other
background anchors still contribute classification loss. This does not mask every pixel in a box.
Mosaic/affine confidence alignment is regression tested. Unsupported mixing augmentations fail.
Required held-out evaluation failure returns nonzero and writes evaluation status.
Balanced and pure-mosaic variants are optional research paths, not separately certified here.
