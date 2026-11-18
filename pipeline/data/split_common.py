"""split_common.py — pieces shared by every data strategy.

Each strategy under data/ decides *which* images land in `train` / `label` / `val` /
`eval`. Everything else — loading and validating COCO input, the few-shot sampler, building
subset jsons, writing split directories, and the console report — is the same regardless of
that decision and lives here, so a fix to the sampler or the symlink writer applies to every
strategy at once.

Strategies import this by putting data/ on sys.path:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from split_common import fewshot_sample, load_coco, write_split

matching how run_teacher_adaptation.py reaches pipeline_common.py one level up.

All strategies share one output contract: four directories named by SPLITS, each holding
OUT_ANN_NAME plus relative symlinks to only the images that split references. Later phases
consume any strategy's output identically.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Filename every generated split's annotation json is written under, inside that split's own
# directory -- the split is identified by the directory, not the filename. Deliberately the
# Roboflow/COCO-export convention rather than "<split>.json": SAM3's finetuning Hydra template
# (experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml) hardcodes
# ".../<split>/_annotations.coco.json" as a literal in its ann_file interpolation, so a dataset
# is only usable for SAM3 finetuning if its splits use this exact name.
OUT_ANN_NAME = "_annotations.coco.json"

SPLITS = ("train", "label", "val", "eval")

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


# ── Loading and validation ────────────────────────────────────────────────────


def load_coco(path: Path, *, hint: str = "") -> dict:
    """Load one COCO json and reject anything the split layout cannot represent.

    `hint` is appended to id/file_name uniqueness errors so a strategy can point at the
    specific input that needs fixing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"annotation file not found: {path}")
    with open(path) as f:
        coco = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{path}: missing required COCO key {key!r}")
    if not coco["images"]:
        raise ValueError(f"{path}: contains no images")

    suffix = f" {hint}" if hint else ""

    ids = [im["id"] for im in coco["images"]]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    if dup_ids:
        raise ValueError(
            f"{path}: {len(dup_ids)} duplicate image id(s), e.g. {dup_ids[:5]}.{suffix}")

    # Each split is a flat directory of symlinks named by file_name, so two images sharing a
    # file_name inside one output split cannot both be represented -- and the stale cleanup in
    # write_split() would delete a link the split still needs.
    names = [im["file_name"] for im in coco["images"]]
    dup_names = [n for n, c in Counter(names).items() if c > 1]
    if dup_names:
        raise ValueError(
            f"{path}: {len(dup_names)} duplicate file_name(s), e.g. {dup_names[:5]}. Split "
            f"directories are flat, so file names must be unique.{suffix}")

    for name in names:
        if os.path.isabs(name) or os.path.normpath(name).startswith(".."):
            raise ValueError(
                f"{path}: file_name {name!r} is absolute or escapes the image root; "
                "file_name must be a relative path inside the image directory.")

    known = set(ids)
    unknown = {a["image_id"] for a in coco["annotations"]} - known
    if unknown:
        raise ValueError(
            f"{path}: {len(unknown)} annotation(s) reference image ids not in 'images', "
            f"e.g. {sorted(unknown)[:5]}")

    return coco


def index_categories(coco: dict) -> dict:
    """image id -> set of category ids present in it (empty set if unannotated)."""
    per_image = {im["id"]: set() for im in coco["images"]}
    for ann in coco["annotations"]:
        per_image[ann["image_id"]].add(ann["category_id"])
    return per_image


def category_ids(coco: dict) -> list:
    return [c["id"] for c in coco["categories"]]


# ── The few-shot sampler ──────────────────────────────────────────────────────


def fewshot_sample(image_ids, per_image: dict, cat_ids, shots: int, seed: int) -> set:
    """Greedy image-level quota fill over a seeded shuffle of `image_ids`.

    Sorts the candidate ids, shuffles them with `random.Random(seed)`, then makes one pass
    keeping any image that still helps an under-quota category and spending one shot from
    *every* category that image contains.

    The quota is image-level, not instance-level: an image holding 14 boxes of one category
    spends exactly one shot for it. Categories overshoot whenever they co-occur with a
    still-hungry category. Images with no annotations are never selected (empty category set).

    This reproduces the sampler behind the few-shot annotation files shipped with the ODinW
    benchmark -- verified byte-for-byte against 423 of them, across shots 1/3/5/10 and seeds
    3/30/300 -- so a train split's core stays comparable to published few-shot numbers.
    Nothing about the implementation is ODinW-specific.

    Only the *relative order* of `image_ids` matters, so sampling over a set of ids and over
    any order-isomorphic renumbering of them selects the same images.
    """
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)

    # Keyed by category id rather than the upstream `[shots] * n_cats` list indexed at c-1,
    # which silently misbehaves on 0-based (e.g. a `background` category) or non-contiguous
    # category ids. Equivalent for contiguous 1..N ids.
    quota = {c: shots for c in cat_ids}

    keep = []
    for img_id in ids:
        cats = per_image[img_id]
        if any(quota.get(c, 0) > 0 for c in cats):
            keep.append(img_id)
            for c in cats:
                # Decrements every category in the image, including already-exhausted ones
                # (they go negative) -- this is what makes co-occurring categories overshoot
                # rather than pulling in extra images.
                if c in quota:
                    quota[c] -= 1
    return set(keep)


def release_from_train(train_ids: set, core_ids: set, k: int, per_image: dict,
                       rng: random.Random) -> list:
    """Pick `k` images for train to give back, least-damaging first.

    Top-up images go first -- they are not part of the few-shot core, so releasing them costs
    nothing. Only if that is not enough are core images released, choosing at each step the
    image whose rarest category is best covered by the *rest* of train, recomputed after every
    removal. Unannotated images (which the core never contains) sort as maximally safe.
    """
    released: list = []

    topups = sorted(train_ids - core_ids)
    rng.shuffle(topups)
    released.extend(topups[:k])

    kept = set(train_ids) - set(released)
    while len(released) < k:
        coverage: Counter = Counter()
        for img_id in kept:
            for c in per_image[img_id]:
                coverage[c] += 1

        def safety(img_id):
            # Higher is safer: how well-covered this image's rarest category is elsewhere.
            return min((coverage[c] for c in per_image[img_id]), default=sys.maxsize)

        candidates = sorted(kept, key=lambda i: (-safety(i), i))
        if not candidates:
            raise AssertionError("ran out of train images to release")
        chosen = candidates[0]
        released.append(chosen)
        kept.discard(chosen)

    return released


# ── Building and writing splits ───────────────────────────────────────────────


def build_subset(coco: dict, keep_ids: set) -> dict:
    """A COCO dict holding only `keep_ids`, with ids untouched."""
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [im for im in coco["images"] if im["id"] in keep_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep_ids],
    }


def renumber_subset(subset: dict, first_image_id: int = 0, first_ann_id: int = 0) -> dict:
    """Renumber a subset's image and annotation ids into a contiguous range.

    Each image keeps its original identity in `source_image_id` (and whatever `source_split`
    the caller already set), so the mapping back to the input json is never lost. Images are
    renumbered in ascending original-id order, and annotations follow their image, so the
    result is a deterministic function of the subset's contents.
    """
    out = dict(subset)
    images, remap = [], {}
    for offset, im in enumerate(sorted(subset["images"], key=lambda x: x["id"])):
        new = dict(im)
        new["source_image_id"] = im["id"]
        new["id"] = first_image_id + offset
        remap[im["id"]] = new["id"]
        images.append(new)

    annotations = []
    for offset, ann in enumerate(sorted(subset["annotations"],
                                        key=lambda a: (remap[a["image_id"]], a["id"]))):
        new = dict(ann)
        new["id"] = first_ann_id + offset
        new["image_id"] = remap[ann["image_id"]]
        annotations.append(new)

    out["images"] = images
    out["annotations"] = annotations
    return out


def write_split(subset: dict, split_dir: Path, img_root: Path) -> None:
    """Write one split directory: its annotation json plus relative symlinks to its images."""
    split_dir, img_root = Path(split_dir), Path(img_root)
    split_dir.mkdir(parents=True, exist_ok=True)

    for im in subset["images"]:
        src = img_root / im["file_name"]
        if not src.is_file():
            raise FileNotFoundError(
                f"image referenced by the {split_dir.name} split not found on disk: {src}")

    # Drop images this split no longer claims. Membership shifts between runs whenever the
    # input, seed, or fractions change; without this, an image reassigned from train to val
    # stays symlinked in BOTH directories, and any consumer that enumerates the directory
    # instead of reading the json sees a split contradicting its own annotations.
    keep_names = {im["file_name"] for im in subset["images"]}
    for entry in os.scandir(split_dir):
        if entry.name not in keep_names and Path(entry.name).suffix.lower() in IMG_SUFFIXES:
            os.unlink(entry.path)

    for im in subset["images"]:
        link = split_dir / im["file_name"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(img_root / im["file_name"], start=link.parent))

    with open(split_dir / OUT_ANN_NAME, "w") as f:
        json.dump(subset, f)


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_split_table(subsets: dict) -> None:
    total = sum(len(s["images"]) for s in subsets.values())
    print(f"\n{'split':<8}{'images':>9}{'anns':>9}{'% of pool':>12}")
    for name in SPLITS:
        sub = subsets[name]
        n = len(sub["images"])
        pct = 100 * n / total if total else 0.0
        print(f"{name:<8}{n:>9}{len(sub['annotations']):>9}{pct:>11.1f}%")
        if not n:
            print("         (empty)")


def print_category_warnings(subsets: dict, categories: list, shots: int) -> None:
    """Report categories that fall short of the quota, are unused, or are missing from a split."""
    cat_names = {c["id"]: c["name"] for c in categories}

    images_with_cat: dict = {c: 0 for c in cat_names}
    present: dict = {name: set() for name in SPLITS}
    unannotated = {name: 0 for name in SPLITS}
    for name, sub in subsets.items():
        per_image = index_categories(sub)
        for cats in per_image.values():
            if not cats:
                unannotated[name] += 1
            for c in cats:
                if c in images_with_cat:
                    images_with_cat[c] += 1
                present[name].add(c)

    unused = sorted(c for c, n in images_with_cat.items() if not n)
    if unused:
        print(f"\nnote: {len(unused)} declared categor{'y' if len(unused) == 1 else 'ies'} have "
              f"no annotations anywhere: {', '.join(repr(cat_names[c]) for c in unused[:10])}"
              f"{' ...' if len(unused) > 10 else ''}")

    thin = sorted(c for c, n in images_with_cat.items() if 0 < n < shots)
    if thin:
        print(f"\nwarning: {len(thin)} categor{'y' if len(thin) == 1 else 'ies'} have fewer than "
              f"{shots} images in the whole dataset, so train cannot reach the quota:")
        for c in thin[:10]:
            print(f"  {cat_names[c]!r}: {images_with_cat[c]} image(s)")
        if len(thin) > 10:
            print(f"  ... and {len(thin) - 10} more")

    for name in SPLITS:
        if not subsets[name]["images"]:
            continue
        missing = [cat_names[c] for c in sorted(cat_names)
                   if images_with_cat[c] and c not in present[name]]
        if missing:
            print(f"\nwarning: {len(missing)} categor{'y' if len(missing) == 1 else 'ies'} absent "
                  f"from the {name} split: {', '.join(repr(m) for m in missing[:10])}"
                  f"{' ...' if len(missing) > 10 else ''}")

    total_unann = sum(unannotated.values())
    if total_unann:
        print(f"\nnote: {total_unann} image(s) carry no annotations (never selected by the "
              "few-shot stage); placed as "
              + ", ".join(f"{k}={unannotated[k]}" for k in SPLITS))


def assert_partition(subsets: dict, expected_ids: set, key: str = "id") -> None:
    """Every expected image appears in exactly one split."""
    seen: set = set()
    for i, a in enumerate(SPLITS):
        ids_a = {im[key] for im in subsets[a]["images"]}
        overlap = seen & ids_a
        assert not overlap, f"{a} overlaps an earlier split: {sorted(overlap)[:5]}"
        seen |= ids_a
    assert seen == expected_ids, "splits do not cover the dataset exactly"
