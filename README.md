# SC-SimpleMKKM (Compact Reproduction Package)

This compact package contains the 0.8.4.post2 main method source, the small six-view MFeat dataset, a seed-0 configuration, and a reference check. Large sensor data, synthetic scale data, paper-scale inputs, and run outputs are omitted.

## Quick reproduction

Use Python 3.10 or newer; Python 3.12 is recommended. From the repository root, run:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[ann-faiss,parallel]"
    python scripts/verify_data.py
    python -m scsimplemkkm.cli doctor
    python -m scsimplemkkm.cli plan --config configs/mfeat_rbf_self_tuned_seed0.json
    python -m scsimplemkkm.cli run-mv --config configs/mfeat_rbf_self_tuned_seed0.json
    python scripts/check_mfeat_reference.py

Outputs are written to outputs/. On Linux/macOS, activate the environment with source .venv/bin/activate; the remaining python commands are the same.

The configuration uses seed 0, one locally self-tuned RBF kernel per view, local_scale_k=7, sparse support size support_k=10, the SimpleMKKM solver, and batched Lloyd discretization. MFeat contains 2,000 samples, six views, and ten classes.

Archived reference metrics are ACC 0.872500, NMI 0.8794354695958252, and ARI 0.8262795901539949. Small numerical differences may occur across NumPy, FAISS, BLAS, and CPU builds. Runtime and memory depend on the host.

MFeat source attribution, CC BY 4.0 terms, preprocessing notes, and SHA-256 values are in data/README.md and data/SHA256SUMS. The original UCI archive is omitted to reduce package size; the processed six-view matrices and labels are included.
