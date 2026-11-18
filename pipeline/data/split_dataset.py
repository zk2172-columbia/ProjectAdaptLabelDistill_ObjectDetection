#!/usr/bin/env python3
"""split_dataset.py — re-split a dataset that already ships train/valid/test splits.

Keeps the source partition's val and eval sets as-is and re-partitions only its train set into
our smaller `train` and the `label` pool the teacher pseudo-labels:

    val   = preset val set, unchanged
    eval  = preset eval set, unchanged
    train = few-shot core over the preset train set (`shots` images per category, greedy) topped
            up with uniformly random preset-train images to `train_frac` of the OVERALL dataset
    label = preset train set minus train

Both stages are driven by one `seed`. Run this before teacher adaptation, once per dataset. A dataset with no
usable pre-existing partition must be merged into one unified COCO json first.

Output: four directories under `out_dir`, each holding `_annotations.coco.json` plus relative
symlinks to only the images it references (no image bytes are copied). Image and annotation ids
are renumbered into one global contiguous space, since Roboflow-style exports restart ids at 0 per
split; every image keeps `source_image_id` and `source_split`. Re-running into an existing output
directory purges images the new split no longer claims.

Usage
-----
    python split_dataset.py --config split_config.yaml

    # Config-free, or overriding individual config values:
    python split_dataset.py --train-ann /data/ds/train/_annotations.coco.json \
                            --val-ann   /data/ds/valid/_annotations.coco.json \
                            --eval-ann  /data/ds/test/_annotations.coco.json \
                            --out-dir   /data/ds-splits
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_ground_truth import export_from_config  # noqa: E402
from split_common import (  # noqa: E402
    SPLITS,
    build_subset,
    category_ids,
    fewshot_sample,
    index_categories,
    load_coco,
    print_category_warnings,
    print_split_table,
    renumber_subset,
    write_split,
)

# Source splits carried through untouched, named by the OUTPUT split they become. Whatever the
# export calls them on disk (valid/test, val/eval, ...) is a matter of the path in the config.
PASSTHROUGH = ("val", "eval")


# ── Loading ───────────────────────────────────────────────────────────────────


def load_sources(paths: dict) -> dict:
    """Load the three source jsons and require them to agree on their categories.

    All four output splits carry ONE `categories` list, so the three inputs must define the
    same one. What is compared is the id -> name mapping: the same set of category ids, each
    bound to the same name, in all three sources. A category present in one source and missing
    from another, or an id bound to different names, is a hard error -- category id 3 has to
    mean the same class in train, val, and eval or every annotation downstream is mislabelled.

    Declaration order and `supercategory` are NOT compared, and the output splits reuse the
    train source's `categories` list verbatim. A category that is declared everywhere but
    annotated nowhere is fine here; it is reported later as an unused category.
    """
    sources = {}
    for name in ("train", *PASSTHROUGH):
        sources[name] = load_coco(
            paths[name],
            hint=f"This is the preset {name} source; fix it in the export rather than here.")

    reference = {c["id"]: c["name"] for c in sources["train"]["categories"]}
    for name in PASSTHROUGH:
        other = {c["id"]: c["name"] for c in sources[name]["categories"]}
        if other != reference:
            only_train = sorted(set(reference.items()) - set(other.items()))[:5]
            only_other = sorted(set(other.items()) - set(reference.items()))[:5]
            raise ValueError(
                f"category mismatch between the preset train source and the preset {name} "
                f"source: train-only {only_train}, {name}-only {only_other}. All three sources "
                "must share one id -> name mapping.")
    return sources


def check_cross_split_leakage(sources: dict) -> None:
    """Refuse to run if a file name appears in more than one source split.

    An image present in both the preset train set and the preset val/eval set reaches
    evaluation through train or label, silently inflating every metric measured afterwards.
    There is no safe repair -- dropping it would reshape the preset partition, and keeping it
    would poison the results -- so this is a hard error the export has to fix.
    """
    seen: dict = {}
    for name, coco in sources.items():
        for im in coco["images"]:
            seen.setdefault(im["file_name"], []).append(name)
    shared = {n: s for n, s in seen.items() if len(s) > 1}
    if shared:
        example = ", ".join(f"{n!r} in {'+'.join(s)}" for n, s in list(shared.items())[:5])
        raise ValueError(
            f"{len(shared)} image file name(s) appear in more than one source split, so the "
            "same image would land in more than one output split and leak between training "
            f"and evaluation: {example}"
            f"{' ...' if len(shared) > 5 else ''}. Deduplicate the preset partition at the "
            "source -- this step will not silently drop images to repair it, because which "
            "split keeps the image is your decision, not the splitter's.")


# ── Splitting ─────────────────────────────────────────────────────────────────


def partition(sources: dict, *, shots: int, seed: int, train_frac: float, min_per_split: int):
    train_src = sources["train"]
    train_pool = sorted(im["id"] for im in train_src["images"])
    n_pool = len(train_pool)
    overall = sum(len(sources[n]["images"]) for n in ("train", *PASSTHROUGH))

    per_image = index_categories(train_src)
    core = fewshot_sample(train_pool, per_image, category_ids(train_src), shots, seed)

    target = max(min_per_split, round(overall * train_frac))
    train_size = max(len(core), target)

    if n_pool - train_size < min_per_split:
        driver = ("the few-shot core" if len(core) >= target
                  else f"train_frac ({train_frac:g} of {overall} overall images)")
        raise ValueError(
            f"the preset train set holds {n_pool} image(s), but {driver} wants {train_size} of "
            f"them for train, leaving {n_pool - train_size} for label -- below the "
            f"{min_per_split}-image minimum. This strategy never takes images back out of "
            "train to fix that, because the preset val/eval partition is yours to own. Lower "
            "shots or train_frac, raise the preset train set's size, or use the "
            "merge them into one unified COCO json first.")

    train_ids = set(core)
    rng = random.Random(seed)
    topup_n = 0
    if len(train_ids) < target:
        candidates = [i for i in train_pool if i not in train_ids]
        rng.shuffle(candidates)
        added = candidates[:target - len(train_ids)]
        train_ids.update(added)
        topup_n = len(added)

    label_ids = set(train_pool) - train_ids
    assert train_ids | label_ids == set(train_pool), "train+label must reconstruct the preset train set"
    assert not train_ids & label_ids, "train/label overlap"

    stats = {
        "overall": overall,
        "pool": n_pool,
        "core": len(core),
        "target": target,
        "topup": topup_n,
        "frozen": topup_n == 0 and len(core) > target,
        "source_counts": {n: len(sources[n]["images"]) for n in ("train", *PASSTHROUGH)},
    }
    return {"train": train_ids, "label": label_ids}, stats


def build_subsets(sources: dict, split_ids: dict) -> dict:
    """Build the four output subsets, renumbered into one global contiguous id space.

    Source exports restart image ids at 0 per split, so the three inputs collide. Ids are
    assigned in output-split order (train, label, val, eval) with annotation ids running
    alongside; `source_image_id` / `source_split` preserve the original identity.
    """
    raw = {
        "train": build_subset(sources["train"], split_ids["train"]),
        "label": build_subset(sources["train"], split_ids["label"]),
        **{name: build_subset(sources[name], {im["id"] for im in sources[name]["images"]})
           for name in PASSTHROUGH},
    }
    origin = {"train": "train", "label": "train", "val": "val", "eval": "eval"}

    subsets, next_img, next_ann = {}, 0, 0
    for name in SPLITS:
        sub = dict(raw[name])
        sub["images"] = [{**im, "source_split": origin[name]} for im in sub["images"]]
        sub = renumber_subset(sub, first_image_id=next_img, first_ann_id=next_ann)
        next_img += len(sub["images"])
        next_ann += len(sub["annotations"])
        subsets[name] = sub
    return subsets


def verify(subsets: dict, sources: dict, split_ids: dict) -> None:
    """The output must reproduce the preset partition exactly, and use one clean id space."""
    for name in PASSTHROUGH:
        got = {im["source_image_id"] for im in subsets[name]["images"]}
        want = {im["id"] for im in sources[name]["images"]}
        assert got == want, f"{name} split does not match its preset source exactly"

    for name in ("train", "label"):
        got = {im["source_image_id"] for im in subsets[name]["images"]}
        assert got == split_ids[name], f"{name} split lost images during renumbering"

    all_img_ids = [im["id"] for s in subsets.values() for im in s["images"]]
    assert len(set(all_img_ids)) == len(all_img_ids), "global image ids are not unique"
    assert set(all_img_ids) == set(range(len(all_img_ids))), "global image ids are not contiguous"

    all_ann_ids = [a["id"] for s in subsets.values() for a in s["annotations"]]
    assert len(set(all_ann_ids)) == len(all_ann_ids), "global annotation ids are not unique"

    for name, sub in subsets.items():
        valid = {im["id"] for im in sub["images"]}
        assert all(a["image_id"] in valid for a in sub["annotations"]), \
            f"{name} has annotations pointing outside its own images"


def report(subsets: dict, categories: list, stats: dict, shots: int) -> None:
    counts = stats["source_counts"]
    print(f"\nsources: " + ", ".join(f"{k}={v}" for k, v in counts.items())
          + f"  ->  {stats['overall']} images overall")
    print(f"few-shot core ({shots}-shot, from the {stats['pool']}-image preset train set): "
          f"{stats['core']} images   |   train target: {stats['target']} "
          f"({100 * stats['target'] / stats['overall']:.1f}% of overall)   |   "
          f"top-up added: {stats['topup']}")
    if stats["frozen"]:
        print(f"  note: few-shot core exceeds the train target -- train frozen at "
              f"{stats['core']} ({100 * stats['core'] / stats['overall']:.1f}% of overall)")

    print_split_table(subsets)
    print_category_warnings(subsets, categories, shots)


# ── Config / CLI ──────────────────────────────────────────────────────────────

DEFAULTS = {
    "shots": 10,
    "seed": 42,
    "train_frac": 0.05,
    "min_images_per_split": 1,
    # Post-step: re-export the finished splits into pipeline/data/ground_truths/<dataset>/.
    # See prepare_ground_truth.py. Absent/disabled = the splits are written and nothing else.
    "ground_truth_export": None,
}
SOURCE_KEYS = ("train", "val", "eval")


def resolve_settings(args) -> dict:
    """Merge config file values over defaults, then CLI flags over those.

    Relative paths in the config resolve against the config file's own directory (matching the
    rest of the pipeline's configs); relative paths given on the CLI resolve against cwd.
    """
    settings = dict(DEFAULTS)
    settings["out_dir"] = None
    sources = {k: {"ann": None, "img_dir": None} for k in SOURCE_KEYS}

    if args.config:
        config_path = Path(args.config).resolve()
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - set(settings) - {"sources"}
        if unknown:
            raise SystemExit(f"{config_path}: unknown config key(s): {', '.join(sorted(unknown))}")

        raw_sources = cfg.pop("sources", None) or {}
        unknown_src = set(raw_sources) - set(SOURCE_KEYS)
        if unknown_src:
            raise SystemExit(f"{config_path}: unknown sources key(s): "
                             f"{', '.join(sorted(unknown_src))} (expected {', '.join(SOURCE_KEYS)})")
        for name, entry in raw_sources.items():
            entry = entry or {}
            unknown_field = set(entry) - {"ann", "img_dir"}
            if unknown_field:
                raise SystemExit(f"{config_path}: unknown field(s) under sources.{name}: "
                                 f"{', '.join(sorted(unknown_field))}")
            for field in ("ann", "img_dir"):
                if entry.get(field):
                    sources[name][field] = str((config_path.parent / entry[field]).resolve())

        if cfg.get("out_dir"):
            cfg["out_dir"] = str((config_path.parent / cfg["out_dir"]).resolve())
        # ground_truth_export:'s own paths resolve against the config too, so the export step
        # stays a config value rather than anything this strategy hardcodes.
        gte = cfg.get("ground_truth_export")
        if gte:
            for key in ("out_dir", "classes_file"):
                if gte.get(key):
                    gte[key] = str((config_path.parent / gte[key]).resolve())
        settings.update({k: v for k, v in cfg.items() if v is not None})

    for name in SOURCE_KEYS:
        for field in ("ann", "img_dir"):
            val = getattr(args, f"{name}_{field}", None)
            if val is not None:
                sources[name][field] = str(Path(val).resolve())
    for key in settings:
        val = getattr(args, key, None)
        if val is not None:
            settings[key] = str(Path(val).resolve()) if key == "out_dir" else val

    missing = [n for n in SOURCE_KEYS if not sources[n]["ann"]]
    if missing:
        raise SystemExit(
            f"no annotation file given for preset source(s): {', '.join(missing)}. All three "
            "are required -- set sources.<name>.ann in the config or pass --<name>-ann. This "
            "strategy carries the preset val/eval sets through unchanged, so it cannot invent "
            "them.")
    if not settings["out_dir"]:
        raise SystemExit("no output directory given -- set `out_dir` in the config or pass --out-dir")

    if settings["shots"] < 1:
        raise SystemExit("shots must be >= 1")
    if settings["min_images_per_split"] < 1:
        raise SystemExit("min_images_per_split must be >= 1")
    if not 0 < settings["train_frac"] <= 1:
        raise SystemExit("train_frac must be in (0, 1]")

    for name in SOURCE_KEYS:
        if not sources[name]["img_dir"]:
            sources[name]["img_dir"] = str(Path(sources[name]["ann"]).parent)
    settings["sources"] = sources
    return settings


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="data: keep a dataset's preset "
                    "val/eval splits and re-split its preset train set into train/label.",
        epilog="Every value can come from --config or be overridden by the matching flag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML config (see split_config.yaml)")
    for name in SOURCE_KEYS:
        parser.add_argument(f"--{name}-ann", default=None, dest=f"{name}_ann",
                            help=f"COCO json of the preset {name} source")
        parser.add_argument(f"--{name}-img-dir", default=None, dest=f"{name}_img_dir",
                            help=f"image root for the preset {name} source "
                                 f"(default: the directory containing its annotation file)")
    parser.add_argument("--out-dir", default=None, dest="out_dir",
                        help="directory to write the train/label/val/eval split dirs into")
    parser.add_argument("--shots", type=int, default=None,
                        help="images per category for the few-shot core (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for both stages (default: 42)")
    parser.add_argument("--train-frac", type=float, default=None, dest="train_frac",
                        help="target train share of the OVERALL dataset, preset val/eval "
                             "included; a larger few-shot core overrides it (default: 0.05)")
    parser.add_argument("--min-images-per-split", type=int, default=None,
                        dest="min_images_per_split",
                        help="floor for the label split left behind by train (default: 1)")
    args = parser.parse_args(argv)

    s = resolve_settings(args)
    out_root = Path(s["out_dir"])

    sources = load_sources({n: Path(s["sources"][n]["ann"]) for n in SOURCE_KEYS})
    check_cross_split_leakage(sources)

    split_ids, stats = partition(sources, shots=s["shots"], seed=s["seed"],
                                 train_frac=s["train_frac"],
                                 min_per_split=s["min_images_per_split"])
    subsets = build_subsets(sources, split_ids)
    verify(subsets, sources, split_ids)

    img_roots = {"train": s["sources"]["train"]["img_dir"],
                 "label": s["sources"]["train"]["img_dir"],
                 "val": s["sources"]["val"]["img_dir"],
                 "eval": s["sources"]["eval"]["img_dir"]}
    for name in SPLITS:
        write_split(subsets[name], out_root / name, Path(img_roots[name]))

    report(subsets, sources["train"]["categories"], stats, s["shots"])
    print(f"\nwrote {', '.join(SPLITS)} to {out_root}")

    # Convert the splits just written into the formats everything downstream consumes — YOLO
    # labels for training, COCO for scoring — under pipeline/data/ground_truths/<dataset>/. Off unless
    # the config asks for it, and every path in that block is a config value, so this strategy
    # stays dataset-agnostic and still imports nothing from the rest of the pipeline.
    export_from_config(s, None, out_root)


if __name__ == "__main__":
    main()
