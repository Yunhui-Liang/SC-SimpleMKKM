"""Verify the checked-in dataset hashes and multi-view manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from scsimplemkkm.prepared_catalog import inspect_manifest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    checksum_file = root / "data" / "SHA256SUMS"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"hash mismatch: {relative}")

    sample_root = root / "src" / "scsimplemkkm" / "sampledata"
    sample_checksums = sample_root / "SHA256SUMS"
    for line in sample_checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = sample_root / relative
        if not path.is_file():
            failures.append(f"missing package sample: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"sample hash mismatch: {relative}")

    for dataset in ("mfeat_mvc",):
        manifest = root / "data" / "examples" / dataset / "manifest.json"
        result = inspect_manifest(manifest)
        if not result.get("ready"):
            failures.append(f"manifest invalid: {dataset}: {result.get('message')}")
        else:
            dims = ",".join(str(v["d"]) for v in result["views"])
            print(f"{dataset}: ready n={result['n']} views={len(result['views'])} dims={dims}")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("All dataset hashes and manifests are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
