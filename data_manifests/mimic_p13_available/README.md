# MIMIC-CXR p13 available JPG/TXT pairing

This directory contains a read-only audit and loader-ready manifest for the p13
files currently available on disk. It preserves the official MIMIC-CXR split;
it does not reassign patients by prefix and does not downsample No Finding.

## Result

- Expected: 37,857 images, 22,945 studies, 6,548 patients
- Paired: 37,809 images, 22,923 studies, 6,539 patients
- Missing JPG: 48
- Missing TXT for an available JPG: 0
- Unreadable JPG: 0
- Reports without any available JPG: 22
- Official split: train 36,683 / validate 374 / test 752

Use `p13_paired_manifest.csv.gz` with `MIMICCXRDataset` or
`LDM_dp_finetune.py --manifest_csv`. The other CSV files retain the exact
missing-image and report-only audit rows.

## Rebuild

```bash
docker exec CXR_dp_medclip sh -lc '
  cd /workspace/PrivaText-CXR &&
  /opt/conda/bin/python -B src/Data/build_mimic_prefix_manifest.py \
    --prefix p13 \
    --report-root /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
    --jpg-root /storage/hjchoi/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --out-dir data_manifests/mimic_p13_available \
    --verify-images'
```
