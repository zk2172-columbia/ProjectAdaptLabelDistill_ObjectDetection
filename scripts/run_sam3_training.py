"""Launch the unmodified SAM3 trainer with a config outside its source package."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


def run(config: Path, sam3_root: Path, *, num_gpus: int = 1, use_cluster: bool = False) -> None:
    config = config.resolve()
    sam3_root = sam3_root.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"SAM3 training config not found: {config}")
    if not (sam3_root / "sam3/train/train.py").is_file():
        raise FileNotFoundError(f"SAM3 training source not found under {sam3_root}")
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sam3_root))
    from hydra import initialize_config_dir
    from sam3.train import train as upstream
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    args = SimpleNamespace(config=config.name, use_cluster=use_cluster,
                           num_gpus=num_gpus, num_nodes=None,
                           partition=None, account=None, qos=None)
    register_omegaconf_resolvers()
    with initialize_config_dir(config_dir=str(config.parent), version_base="1.2"):
        upstream.main(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sam3-root", required=True, type=Path)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--use-cluster", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    run(args.config, args.sam3_root, num_gpus=args.num_gpus,
        use_cluster=bool(args.use_cluster))


if __name__ == "__main__":
    main()
