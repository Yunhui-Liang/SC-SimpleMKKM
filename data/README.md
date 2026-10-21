# Included data

## UCI Multiple Features (MFeat)

- Upstream: UCI Multiple Features, https://archive.ics.uci.edu/dataset/72/multiple%2Bfeatures, DOI 10.24432/C5HC70, dataset creator Robert Duin.
- Upstream license: CC BY 4.0; retain attribution when sharing.
- Included form: six aligned float32 feature matrices and class labels (2,000 rows); each view is row-L2 normalized and original sample order is retained.
- The processed matrices and manifest are included. The original upstream ZIP is omitted to keep this compact package small. Download it from UCI if you need to regenerate the processed matrices.
- Checksums for all included MFeat files are listed in data/SHA256SUMS.

Other paper datasets are intentionally omitted. The installed package provides dataset preparation recipes for downloading or generating them; review each source manifest and license before redistribution.

## Bundled small sample datasets

The package also includes row-L2-normalized arrays made from scikit-learn's
bundled copies of [Iris](https://archive.ics.uci.edu/dataset/53/iris),
[Wine](https://archive.ics.uci.edu/dataset/109/wine),
[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), and the test set of
[Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
These UCI datasets are licensed CC BY 4.0. The `synth3k` sample is generated
synthetically. Per-file details are in `src/scsimplemkkm/sampledata/`.
