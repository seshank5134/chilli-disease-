# Original and augmented dataset comparison

This comparison uses the two downloaded folders in the Desktop `ml  project` directory. It only reads their files; audit outputs are kept in this workspace.

## Inventory

| Literal class folder | Original images | Augmented images |
|---|---:|---:|
| Bacterial Spot | 156 | 2,000 |
| Cercospora Leaf Spot | 180 | 2,000 |
| Curl Virus | 423 | 2,000 |
| Healthy Leaf | 458 | 2,000 |
| Nutrition Deficiency | 444 | 2,000 |
| White spot | 195 | 2,000 |
| **Total** | **1,856** | **12,000** |

## What the file evidence shows

- **Confirmed:** the augmented folder contains 12,000 JPG files, with 2,000 in each of six literal class folders.
- **Confirmed:** the augmented names end in transformation labels (`bright`, `flipped`, `rotated`, `sheared`, `shifted`, or `zoomed`) plus an index. The inventory has 1,988 bright, 1,976 flipped, 1,970 rotated, 2,017 sheared, 2,061 shifted, and 1,988 zoomed filenames.
- **Strong filename-based lineage evidence, not pixel-level proof:** removing the transformation suffix yields 1,845 distinct class-local filename stems; all 1,845 match original image filename stems in the same class folder. The other 11 original filenames have no matching stem in the augmented folder. This strongly suggests the augmented set was generated from these originals, but the filenames alone cannot guarantee the actual image contents.
- **Confirmed:** no SHA-256 file hashes match between original and augmented folders. That is expected if every augmented image was re-encoded or transformed, and it does not contradict the filename-based relationship evidence.
- **Confirmed:** the augmented folder has 1,751 exact duplicate hash groups covering 4,083 files. These are byte-identical duplicates among the augmented files; check `reports/dataset_audit_augmented/duplicate_files.csv` for affected paths.
- **Label review needed:** 15 augmented exact-duplicate groups span different class folders; the original folder also has cross-class exact duplicates. Identical image bytes carrying different folder labels are ambiguous training examples. The training script excludes every file whose exact hash appears under more than one label and marks those rows in the split manifest.
- **Confirmed:** Pillow decoded all 1,856 original and 12,000 augmented files; no corrupt/unreadable images or audit access errors were found. All are RGB. Original images are all 1000×1000; 10,013 augmented images are 1000×1000, with 1,987 images spread across other dimensions.

## Training and evaluation implication

Do not randomly split all 13,856 original and augmented images. The augmented filename pattern exposes candidate family groups, so keep every image sharing the pre-transformation filename stem together in one partition. Ideally, reserve original images for evaluation and keep their related augmented variants out of training. Verify the grouping rule and resolve the 11 originals without corresponding augmented names before creating splits. The exact duplicates should also stay in one partition, although these are already contained in the apparent source families.

Class labels differ slightly from the dataset description: the folders say `Healthy Leaf` and `White spot`, while the description says `Healthy Leaves` and `White Spot`. Confirm these are intended labels before training or exposing them in the prediction app.

## Audit output folders

- Original dataset: `reports/dataset_audit/`
- Augmented dataset: `reports/dataset_audit_augmented/`
- Reusable audit script: `scripts/audit_dataset.py`
