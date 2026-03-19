from __future__ import annotations

import argparse
import json
import os

from verifier import BoxVerifier



def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a solution.json file under strict physics rules")
    parser.add_argument("solution", type=str, help="Path to solution.json")
    args = parser.parse_args()

    with open(args.solution, "r", encoding="utf-8") as f:
        solution = json.load(f)

    verifier = BoxVerifier(solution["bin_size"])
    ok, msg, report = verifier.verify(solution["items"], solution["placement"], return_report=True)
    print(msg)
    report_path = os.path.splitext(args.solution)[0] + "_verification.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved verification report to {report_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
