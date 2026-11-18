# Adapt

Use the root [README](../../README.md) and dataset configuration helper for supported commands.
The phase generates a SAM3 Hydra configuration, fine-tunes the teacher, and merges its saved
head into the base checkpoint. Training defaults are recorded in `config.yaml` and the generated
configuration; retain those files with each run.

Checkpoint merging rejects empty or unmatched updates, verifies every saved tensor after reload,
and records SHA256 hashes. It always preserves the raw checkpoint and refuses to overwrite a
merged output. Use trusted checkpoints and configurations only. Training requires a separately
prepared compatible GPU environment; the private candidate's CPU tests do not certify full adaptation.
