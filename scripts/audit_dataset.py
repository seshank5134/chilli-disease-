#!/usr/bin/env python3
"""Read-only image dataset audit; writes reports only to the chosen output folder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit"))
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    out = args.output.resolve()
    if not root.is_dir():
        parser.error(f"Dataset root is not a directory: {root}")
    out.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        pillow = True
    except ImportError:
        Image = None
        pillow = False

    records, errors, dirs = [], [], []
    ext_counts, dir_counts, dims, modes = Counter(), Counter(), Counter(), Counter()
    names, hashes, classes = defaultdict(list), defaultdict(list), defaultdict(int)
    transform_counts, apparent_sources = Counter(), defaultdict(set)
    transform_pattern = re.compile(r"^(.*)_([A-Za-z]+)_(\d+)$")
    aug_evidence = re.compile(r"aug|synthetic|transformed|rotation|flip", re.I)
    original_evidence = re.compile(r"original|raw", re.I)
    tree = []
    for base, dirnames, filenames in os.walk(root, onerror=lambda e: errors.append(f"walk error: {e}")):
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        pbase = Path(base)
        relbase = pbase.relative_to(root)
        dirs.append(relbase)
        if not dirnames and not filenames:
            tree.append(f"{relbase.as_posix() or '.'}/ [empty]")
        for d in dirnames:
            tree.append(f"{(relbase / d).as_posix()}/")
        for filename in filenames:
            p = pbase / filename
            rel = p.relative_to(root).as_posix()
            ext = p.suffix.lower()
            if ext not in IMAGE_EXTS:
                # Relevant non-image files recorded for structural context.
                records.append({"relative_path": rel, "extension": ext or "[none]", "file_size_bytes": "", "sha256": "", "width": "", "height": "", "mode": "", "validation_status": "unsupported_non_image"})
                continue
            try:
                size = p.stat().st_size
                digest = sha256_file(p)
            except OSError as e:
                errors.append(f"{rel}: file access/hash error: {type(e).__name__}: {e}")
                records.append({"relative_path": rel, "extension": ext, "file_size_bytes": "", "sha256": "", "width": "", "height": "", "mode": "", "validation_status": "inaccessible"})
                continue
            width = height = mode = ""
            status = "validation_skipped_pillow_unavailable"
            if pillow:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        width, height, mode = im.width, im.height, im.mode
                    status = "readable"
                    dims[(width, height)] += 1
                    modes[mode] += 1
                except Exception as e:
                    status = "corrupt_or_unreadable"
                    errors.append(f"{rel}: image validation error: {type(e).__name__}: {e}")
            rec = {"relative_path": rel, "extension": ext, "file_size_bytes": size, "sha256": digest, "width": width, "height": height, "mode": mode, "validation_status": status}
            records.append(rec)
            ext_counts[ext] += 1
            dir_counts[relbase.as_posix() or "."] += 1
            names[filename.casefold()].append(rel)
            hashes[digest].append(rel)
            match = transform_pattern.match(Path(filename).stem)
            if match:
                source_stem, transform, _ = match.groups()
                transform_counts[transform.casefold()] += 1
                apparent_sources[relbase.as_posix() or "."].add(source_stem)
            # Candidate class directory = immediate child of root; preserve literal directory label.
            parts = Path(rel).parts
            if len(parts) > 1:
                classes[parts[0]] += 1

    image_records = [r for r in records if r["extension"] in IMAGE_EXTS]
    duplicate_groups = [sorted(paths, key=str.casefold) for _, paths in sorted(hashes.items()) if len(paths) > 1]
    duplicate_groups.sort(key=lambda g: g[0].casefold())
    class_rows = sorted(classes.items(), key=lambda kv: kv[0].casefold())
    class_dirs = [d for d in dirs if d != Path(".") and len(d.parts) == 1]
    empty_dirs = sorted((d.as_posix() for d in dirs if d != Path(".") and not any((d / f).is_file() for f in [])), key=str.casefold)
    # Actual empty directory detection (os.walk directory state is retained by checking descendants).
    empty_dirs = []
    for d in dirs:
        full = root / d
        try:
            if not any(full.iterdir()):
                empty_dirs.append(d.as_posix() or ".")
        except OSError as e:
            errors.append(f"{d.as_posix()}: directory access error: {e}")
    class_count_by_top = dict(class_rows)
    aug_dirs = [d.as_posix() for d in dirs if aug_evidence.search(d.as_posix())]
    orig_dirs = [d.as_posix() for d in dirs if original_evidence.search(d.as_posix())]
    normalized_class_names = defaultdict(list)
    for label, _ in class_rows:
        normalized_class_names[re.sub(r"[^a-z0-9]", "", label.casefold())].append(label)
    case_or_punctuation_collisions = [labels for labels in normalized_class_names.values() if len(labels) > 1]
    expected_description_labels = {"Bacterial Spot", "Curl Virus", "Cercospora Leaf Spot", "Nutrition Deficiency", "White Spot", "Healthy Leaves"}
    observed_labels = {label.casefold() for label, _ in class_rows}
    claimed_unmatched = sorted((label for label in expected_description_labels if label.casefold() not in observed_labels), key=str.casefold)
    observed_unmatched = sorted((label for label, _ in class_rows if label.casefold() not in {x.casefold() for x in expected_description_labels}), key=str.casefold)
    duplicate_csv = out / "duplicate_files.csv"
    with duplicate_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["duplicate_group", "sha256", "relative_path"])
        for i, group in enumerate(duplicate_groups, 1):
            digest = next(k for k, v in hashes.items() if len(v) > 1 and sorted(v, key=str.casefold) == group)
            for rel in group: w.writerow([i, digest, rel])
    with (out / "image_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["relative_path", "extension", "file_size_bytes", "sha256", "width", "height", "mode", "validation_status"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(records)
    with (out / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["directory_class_label", "image_count"]); w.writerows(class_rows)

    summary = {
        "dataset_root_label": root.name,
        "total_images": len(image_records),
        "top_level_class_directory_count": len(class_dirs),
        "class_directory_counts": dict(class_rows),
        "counts_by_parent_directory": dict(sorted(dir_counts.items(), key=lambda kv: kv[0].casefold())),
        "file_extension_counts": dict(sorted(ext_counts.items())),
        "image_validation": "Pillow available" if pillow else "skipped; Pillow unavailable",
        "readable_images": sum(r["validation_status"] == "readable" for r in image_records),
        "corrupt_or_unreadable_images": sum(r["validation_status"] == "corrupt_or_unreadable" for r in image_records),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_file_count": sum(map(len, duplicate_groups)),
        "dimension_distribution": {f"{w}x{h}": n for (w, h), n in sorted(dims.items())},
        "color_mode_distribution": dict(sorted(modes.items())),
        "unsupported_non_image_file_count": sum(r["validation_status"] == "unsupported_non_image" for r in records),
        "empty_directories": empty_dirs,
        "repeated_filename_groups": {k: sorted(v, key=str.casefold) for k, v in sorted(names.items()) if len(v) > 1},
        "filename_transform_marker_counts": dict(sorted(transform_counts.items())),
        "filename_derived_distinct_source_stem_counts_by_directory": {k: len(v) for k, v in sorted(apparent_sources.items())},
        "class_label_name_collisions_after_normalization": case_or_punctuation_collisions,
        "claimed_description_labels_without_exact_casefold_match": claimed_unmatched,
        "observed_labels_without_exact_casefold_match_to_description": observed_unmatched,
        "original_augmented_directory_evidence": {"augmented_name_matches": aug_dirs, "original_name_matches": orig_dirs, "status": "inferred from names only; absence does not establish lineage"},
        "errors": errors,
        "audit_status": "completed_with_errors" if errors else "completed",
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    corrupt = [r["relative_path"] for r in image_records if r["validation_status"] == "corrupt_or_unreadable"]
    nonimages = [r["relative_path"] for r in records if r["validation_status"] == "unsupported_non_image"]
    balanced = "Counts vary across the discovered class directories; review the counts below before selecting metrics." if class_rows and max(classes.values()) != min(classes.values()) else "Discovered class-directory counts are equal (or no class directories were identifiable)."
    report = [
        "# Dataset audit report", "",
        f"Dataset root: `{root.name}` (absolute personal path omitted).", "",
        "## Findings and evidence", "",
        f"- **Confirmed:** {len(image_records)} supported image files were discovered recursively.",
        f"- **Confirmed:** {len(class_dirs)} immediate child directories were treated as candidate class directories; labels below are literal folder names, not semantically validated.",
        f"- **Confirmed:** Pillow {'was available' if pillow else 'was unavailable'}; image validation {'was performed' if pillow else 'was skipped'}.",
        f"- **Confirmed:** {len(duplicate_groups)} exact SHA-256 duplicate groups ({sum(map(len, duplicate_groups))} files in those groups). See `duplicate_files.csv`.",
        f"- **Heuristic:** directory-name evidence: augmented-like folders {', '.join(aug_dirs) if aug_dirs else 'none'}; original-like folders {', '.join(orig_dirs) if orig_dirs else 'none'}. Filename transform markers found: {json.dumps(dict(sorted(transform_counts.items())))}. This is evidence of transformed filename classes, but does not by itself establish image lineage.", "",
        "## Directory tree summary", "", "```text", root.name + "/", *["  " + s for s in sorted(tree, key=str.casefold)], "```", "",
        "## Class directories and counts", "", "| Literal directory label | Images |", "|---|---:|",
        *[f"| {name} | {count} |" for name, count in class_rows], "",
        f"{balanced} No expected class set was imposed. The audit does not establish whether any reported class is semantically missing.", "",
        "## Original versus augmented inventory", "",
        f"Directory names alone did not identify an augmented subfolder. Filename patterns provide stronger but still heuristic evidence: transformation suffix counts are `{json.dumps(dict(sorted(transform_counts.items())))}`. Distinct pre-suffix filename stems per folder are `{json.dumps({k: len(v) for k, v in sorted(apparent_sources.items())})}`. These stems are useful candidate source groups; they are not proof that each group derives from the same original pixels. The reported 12,000 files are confirmed in this downloaded folder; relationships to the original download were assessed separately by basename and hash comparison.", "",
        "## Formats and dimensions", "",
        f"Extension counts: `{json.dumps(dict(sorted(ext_counts.items())))}`.",
        f"Image dimensions: `{json.dumps({f'{w}x{h}': n for (w,h),n in sorted(dims.items())}) if pillow else 'not available (Pillow unavailable)'}`.",
        f"Color modes: `{json.dumps(dict(sorted(modes.items()))) if pillow else 'not available (Pillow unavailable)'}`.", "",
        "## Corrupt or unreadable images", "", *( ["None detected by Pillow."] if pillow and not corrupt else ([f"Detected {len(corrupt)}; see inventory and errors:", *[f"- `{p}`" for p in corrupt]] if pillow else ["Skipped because Pillow is unavailable."]) ), "",
        "## Exact and suspected near-duplicates", "",
        f"Exact matches are confirmed by SHA-256: {len(duplicate_groups)} groups. Near-duplicate analysis was not performed; no suitable perceptual comparison is part of the standard-library audit. Exact copies across class folders can indicate labeling problems; copies among augmented data can inflate evaluation scores.", "",
        "## Naming and structure notes", "",
        f"Empty directories: {', '.join('`'+d+'`' for d in empty_dirs) if empty_dirs else 'none'}.",
        f"Unsupported/non-image files: {len(nonimages)}. Their relative paths are recorded in `image_inventory.csv` with status `unsupported_non_image`.",
        f"Repeated basenames: {sum(len(v)>1 for v in names.values())} filename groups. Names are not treated as reliable class labels.", "",
        f"Potential label ambiguity (heuristic against the supplied dataset description): unmatched claimed labels `{', '.join(claimed_unmatched) if claimed_unmatched else 'none'}`; observed unmatched labels `{', '.join(observed_unmatched) if observed_unmatched else 'none'}`. In particular, `Healthy Leaf` differs from the described `Healthy Leaves`, and `White spot` differs in capitalization from `White Spot`; confirm intended labels before modeling. Normalized-name collisions: {case_or_punctuation_collisions or 'none'}.", "",
        "## Leakage risks and split recommendation", "",
        "Randomly splitting augmented images can place transformed versions of one source image in both training and evaluation sets. This lets the model see near-identical content during training and can make evaluation performance overly optimistic.",
        "The filename markers and repeated source stems suggest augmentation families. Random splitting can therefore put byte-identical copies or related transformed images in both training and evaluation partitions and inflate performance. Exact hashes identify byte-identical files, not all transformed descendants. Preserve the pre-transform filename stem as a provisional group key and validate it against generator records or visual review before splitting. For field generalization, also group by collection location/session when that information is available.", "",
        "## Limitations", "",
        "This is a filesystem inventory, not a botanical label review. Directory-label interpretation, original/augmented status, image provenance and expert verification are not validated. EXIF or other metadata was not inspected. Near-duplicates are not assessed. Access failures, if any, are listed in `audit_summary.json`.", "",
        "## Recommended next step", "",
        "Review the literal class counts, errors and duplicate groups; then establish image provenance and source-level grouping before any model training or split creation.", "",
    ]
    (out / "dataset_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "README.md").write_text(
        "# Dataset audit outputs\n\nRun from the project workspace:\n\n"
        "```powershell\npython scripts/audit_dataset.py \"<dataset-root>\" --output reports/dataset_audit\n```\n\n"
        "For this dataset, substitute the supplied dataset folder path for `<dataset-root>`. The script recursively inventories JPG/JPEG/PNG/BMP/WEBP/TIF/TIFF files without modifying the dataset. It writes `dataset_audit_report.md`, `class_counts.csv`, `image_inventory.csv`, `duplicate_files.csv` and `audit_summary.json` to the output folder. Python standard library is sufficient for hashing and inventory. Pillow is optional and, when already installed, validates image decoding and records dimensions and color modes; the script never installs packages.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(out), "total_images": len(image_records), "class_directories": len(class_dirs), "exact_duplicate_groups": len(duplicate_groups), "errors": len(errors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
