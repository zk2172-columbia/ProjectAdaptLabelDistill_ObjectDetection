# Label

Use the root [README](../../README.md) to configure paths and classes. Run SAM3 over the `label`
partition, then filter and convert detections to YOLO labels plus confidence metadata.
The original ALD defaults are within-class NMS IoU 0.3 and confidence cutoff 0.25.
Image masking is disabled by default. Missing predictions, no retained labels, or duplicate
image stems fail explicitly. Keep validation and evaluation images outside the labeling pool.
