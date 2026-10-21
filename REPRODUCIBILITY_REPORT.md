# Reproducibility verification

This compact package preserves the 0.8.4.post2 code snapshot and a complete small real-data reproduction using UCI MFeat.

## MFeat seed-0 reference

- Input: 2,000 rows, six aligned views, ten classes.
- Configuration: configs/mfeat_rbf_self_tuned_seed0.json; seed 0, one self-tuned RBF kernel per view, local scale rank 7, support size 10.
- Output: 2,000 predicted labels and ten nonempty clusters.
- Quality: ACC 0.872500000, NMI 0.879435470, ARI 0.826279590.
- Archived v0.8.4 reference: the same three values exactly.
- The original run used Python 3.12.15, NumPy 2.5.3, SciPy 1.18.1, psutil 7.2.2, threadpoolctl 3.7.0, h5py 3.16.0, FAISS 1.15.1, and Numba 0.68.0. See requirements-reference-windows-py312.txt; it is a reference set, not a full transitive lock file.

Small numeric differences can occur across numerical libraries and hardware. Runtime and memory are host-specific. Use scripts/verify_data.py to verify the included data, then run the commands in README.md or README_CN.md.