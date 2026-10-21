"""Compare the latest MFeat run summary with the archived seed-0 reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = root / "outputs" / "mfeat_rbf_self_tuned_seed000" / "result.json"
    expected = root / "expected" / "mfeat_seed0_v084.json"
    if not result.is_file():
        print(f"Missing run result: {result}", file=sys.stderr)
        return 2
    actual = json.loads(result.read_text(encoding="utf-8"))
    reference = json.loads(expected.read_text(encoding="utf-8"))
    external = (actual.get("evaluation") or {}).get("external") or {}
    tolerance = float(reference["tolerance_for_quality_metrics"])
    failed = False
    print(f"Run software version: {actual.get('software_version')}")
    print(f"Reference software version: {reference['source_software_version']}")
    for key in ("ACC", "NMI", "ARI"):
        value = external.get(key)
        target = float(reference[key])
        if value is None:
            print(f"{key}: missing")
            failed = True
            continue
        delta = abs(float(value) - target)
        ok = delta <= tolerance
        print(f"{key}: actual={float(value):.9f} reference={target:.9f} abs_diff={delta:.9f} {'OK' if ok else 'OUT_OF_TOLERANCE'}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
