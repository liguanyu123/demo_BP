from __future__ import annotations

import json
import os
from collections import Counter

RESULTS_DIR = "results"



def main() -> None:
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    summaries = []
    status_counter = Counter()
    error_counter = Counter()
    fallback_counter = Counter()

    for name in sorted(os.listdir(RESULTS_DIR)):
        exp_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(exp_dir):
            continue
        summary_path = os.path.join(exp_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        status = summary.get("status", "UNKNOWN")
        status_counter[status] += 1
        if summary.get("last_error"):
            error_counter[summary["last_error"]] += 1
        if summary.get("used_reference_fallback"):
            fallback_counter[status] += 1
        summaries.append(summary)

    total = len(summaries)
    verified_success = status_counter.get("SUCCESS", 0) + status_counter.get("REFERENCE_FALLBACK_SUCCESS", 0)
    report = {
        "total_experiments": total,
        "verified_success_count": verified_success,
        "verified_success_rate": round(verified_success / total, 4) if total else 0.0,
        "status_breakdown": dict(status_counter),
        "fallback_breakdown": dict(fallback_counter),
        "top_failure_reasons": dict(error_counter.most_common(10)),
        "experiments": summaries,
    }

    output_path = os.path.join(RESULTS_DIR, "experiment_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Experiment Report ===")
    print(f"Total experiments: {total}")
    print(f"Verified success count: {verified_success}")
    print(f"Verified success rate: {report['verified_success_rate']:.2%}")
    print(f"Status breakdown: {dict(status_counter)}")
    if fallback_counter:
        print(f"Fallback breakdown: {dict(fallback_counter)}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
