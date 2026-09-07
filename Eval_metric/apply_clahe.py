#!/usr/bin/env python3
"""Apply CLAHE to a generated-image run without changing the eval harness.

The output mirrors the input run layout (``descriptions.csv`` plus image paths),
so it can be passed to ``run_feature_eval.py --gen_dir <output>/samples``.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_run_root(path):
    path = Path(path).resolve()
    if (path / "descriptions.csv").is_file():
        return path
    if path.name == "samples" and (path.parent / "descriptions.csv").is_file():
        return path.parent
    raise FileNotFoundError(
        f"descriptions.csv not found in {path} or its parent; pass a generation run root or samples directory"
    )


def _listed_images(run_root):
    csv_path = run_root / "descriptions.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "file" not in rows[0]:
        raise ValueError(f"{csv_path} has no image rows or 'file' column")
    paths = []
    for row in rows:
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe image path in descriptions.csv: {relative}")
        source = run_root / relative
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.append((relative, source))
    return paths


def apply_clahe(input_dir, output_dir, clip_limit=2.0, tile_grid_size=8, overwrite=False):
    run_root = _resolve_run_root(input_dir)
    output_root = Path(output_dir).resolve()
    if run_root == output_root:
        raise ValueError("output_dir must differ from the input run; originals are never overwritten")
    if clip_limit <= 0 or tile_grid_size <= 0:
        raise ValueError("clip_limit and tile_grid_size must be positive")

    pairs = _listed_images(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size))
    )
    written = skipped = 0
    for relative, source in pairs:
        destination = output_root / relative
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode {source}")
        enhanced = clahe.apply(image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp" + destination.suffix)
        if not cv2.imwrite(str(temporary), enhanced):
            raise OSError(f"OpenCV could not write {temporary}")
        os.replace(temporary, destination)
        written += 1

    shutil.copy2(run_root / "descriptions.csv", output_root / "descriptions.csv")
    provenance = {
        "operation": "CLAHE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(run_root),
        "output_run": str(output_root),
        "clip_limit": float(clip_limit),
        "tile_grid_size": [int(tile_grid_size), int(tile_grid_size)],
        "image_count": len(pairs),
        "written": written,
        "skipped_existing": skipped,
        "descriptions_csv_sha256": _sha256(run_root / "descriptions.csv"),
    }
    with (output_root / "clahe_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="generation run root or its samples directory")
    parser.add_argument("--output_dir", required=True, help="new generation run root")
    parser.add_argument("--clip_limit", type=float, default=2.0)
    parser.add_argument("--tile_grid_size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = apply_clahe(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
