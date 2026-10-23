"""Sequential public experiment: independent processes, no biometric data in summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def run(*arguments):
        subprocess.run([sys.executable, "-m", "face_lite", *arguments], cwd=root, check=True)

    run("download-models")
    run("prepare-orl")
    for variant in ("fp32", "int8"):
        output = f"outputs/orl-{variant}"
        run("experiment", "--manifest", "data/orl/manifest.json", "--variant", variant, "--output", output)
        run("benchmark", "--manifest", "data/orl/manifest.json", "--variant", variant,
            "--gallery", f"{output}/gallery.json", "--calibration", f"{output}/calibration.json",
            "--repeats", str(args.repeats), "--output", f"{output}/benchmark.json")
    run("compare", "--baseline", "outputs/orl-fp32", "--candidate", "outputs/orl-int8",
        "--output", "outputs/comparison.json")
    variants = {}
    for variant in ("fp32", "int8"):
        directory = root / f"outputs/orl-{variant}"
        report = json.loads((directory / "report.json").read_text())
        bench = json.loads((directory / "benchmark.json").read_text())
        variants[variant] = {
            "model_version": report["model_version"], "calibration_id": report["calibration_id"],
            "dataset_digest": report["dataset_digest"], "enrolled_people": report["enrolled_people"],
            "validation_queries": report["validation"]["queries"],
            "test": {k: v for k, v in report["test"].items() if k not in ("per_person", "comparator_only")},
            "numerical_quality_target_met": report["numerical_quality_target_met"],
            "minimum_acceptance_coverage_met": report["minimum_acceptance_coverage_met"],
            "benchmark": bench,
        }
    comparison = json.loads((root / "outputs/comparison.json").read_text())
    summary = {"schema": 1, "scope": "public-functional-only", "robot_validated": False,
               "acceptance": False, "source": "AT&T Laboratories Cambridge / ORL",
               "variants": variants,
               "candidate_eligible_locally": comparison["candidate_eligible_locally"],
               "limitations": ["20 known test queries, not 100", "No verified session/camera separation",
                               "No own data or robot available", "No threshold retuning on test",
                               "CPU-only Mac; capture/network excluded; samples are tiny grayscale images"]}
    (root / "outputs/local-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("Aggregate summary: outputs/local-summary.json (no images or embeddings)")


if __name__ == "__main__":
    main()
