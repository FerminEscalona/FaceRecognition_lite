"""Validation-only joint calibration, frozen test, and reproducible measurements."""
from __future__ import annotations

import math
import json
import platform
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import psutil

from .artifacts import ASSETS, digest_file, fingerprint
from .core import FaceEngine
from .data import load_manifest, write_json
from .matching import Calibration, Gallery, decide


def collect(engine: FaceEngine, root: Path, samples: list[dict], gallery: Gallery) -> list[dict]:
    records = []
    for sample in samples:
        start = time.perf_counter()
        item = engine.extract(root / sample["image"])
        scores = gallery.scores(item["embedding"]) if item["status"] == "ok" else []
        records.append({"truth": sample["person_id"], "subject_id": sample["subject_id"],
                        "status": item["status"], "reason": item.get("reason"), "scores": scores,
                        "timings_ms": item["timings_ms"], "elapsed_ms": (time.perf_counter() - start) * 1000})
    return records


def calibrate(records: list[dict], model_version: str, provenance: str, consistency: float) -> Calibration:
    if not any(r["truth"] is not None for r in records) or not any(r["truth"] is None for r in records):
        raise ValueError("Calibration requires known and unknown queries")
    valid = [r for r in records if r["status"] == "ok" and r["scores"]]
    if not valid:
        raise ValueError("No valid validation faces; cannot calibrate")
    top = np.array([r["scores"][0][1] for r in valid])
    gap = np.array([r["scores"][0][1] - r["scores"][1][1] if len(r["scores"]) > 1 else 3.0 for r in valid])
    correct = np.array([r["truth"] is not None and r["scores"][0][0] == r["truth"] for r in valid])
    thresholds = np.unique(np.r_[-1.0, top, np.nextafter(top, np.inf), 1.000001])
    gaps = gap[gap <= 2]
    margins = np.unique(np.r_[0.0, gaps, np.nextafter(gaps, np.inf), 2.000001])
    best = None
    for threshold in thresholds:
        for margin in margins:
            accepted = (top >= threshold) & (gap >= margin) & (gap > 1e-7)
            wrong = int(np.count_nonzero(accepted & ~correct))
            if wrong:
                continue
            key = (int(np.count_nonzero(accepted & correct)), float(threshold), float(margin))
            if best is None or key > best:
                best = key
    assert best is not None  # Reject-all threshold is always available.
    return Calibration(model_version, best[1], best[2], provenance, consistency)


def wilson(successes: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, center-half), min(1, center+half)]


def summarize(records: list[dict], policy: Calibration, include_comparator: bool = True) -> dict:
    known = sum(r["truth"] is not None for r in records)
    unknown = len(records) - known
    correct = wrong_known = false_unknown = rejected_known = rejected_unknown = 0
    statuses: Counter = Counter()
    by_person: dict[str, Counter] = {}
    for r in records:
        decision = decide(r["scores"], policy) if r["status"] == "ok" else {"status": r["status"], "person_id": None}
        statuses[decision["status"]] += 1
        if r["truth"] is not None:
            person = by_person.setdefault(r["truth"], Counter(total=0, correct=0, wrong=0, rejected=0))
            person["total"] += 1
            if decision["person_id"] == r["truth"]:
                correct += 1
                person["correct"] += 1
            elif decision["person_id"] is not None:
                wrong_known += 1
                person["wrong"] += 1
            else:
                rejected_known += 1
                person["rejected"] += 1
        elif decision["person_id"] is not None:
            false_unknown += 1
        else:
            rejected_unknown += 1
    comparator_records = sum(r["status"] == "ok" for r in records)
    result = {
        "queries": len(records), "known_queries": known, "unknown_queries": unknown,
        "correct_known": correct, "wrong_known": wrong_known, "false_unknown_identifications": false_unknown,
        "incorrect_assignments": wrong_known + false_unknown, "rejected_known": rejected_known,
        "rejected_unknown": rejected_unknown,
        "known_identification_rate": correct/known if known else None,
        "known_identification_wilson95": wilson(correct, known),
        "unknown_false_identification_rate": false_unknown/unknown if unknown else None,
        "unknown_false_identification_wilson95": wilson(false_unknown, unknown),
        "status_counts": dict(statuses), "per_person": by_person,
        "comparator_valid_queries": comparator_records,
        "quality_rejections_are_not_identity_evidence": True,
        "latency": latency([r["elapsed_ms"] for r in records]),
    }
    if include_comparator:
        result["comparator_only"] = summarize([r for r in records if r["status"] == "ok"], policy, False)
    return result


def latency(values: list[float]) -> dict:
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": max(values), "over_2000_ms": sum(t > 2000 for t in values), "samples": len(values)} if values else {}


def evaluate(engine: FaceEngine, manifest_path: Path, output: Path,
             mode: str = "experiment", calibration_path: Path | None = None) -> dict:
    if mode not in ("experiment", "calibrate", "evaluate"):
        raise ValueError("Invalid evaluation mode")
    supplied_policy = Calibration(**json.loads(calibration_path.read_text())) if calibration_path else None
    if mode == "evaluate" and supplied_policy is None:
        raise ValueError("Evaluation requires an explicit frozen calibration")
    manifest, dataset_digest = load_manifest(manifest_path)
    root = manifest_path.parent
    output.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "calibration.json", "gallery.json", "model-manifest.json"):
        (output / name).unlink(missing_ok=True)
    rows, enrollments = [], []
    for row in manifest["enrollment"]:
        result = engine.enroll([root / p for p in row["images"]], row["person_id"],
                               [f"{row['person_id']}-{p}" for p in ("front", "left", "right")])
        rows.extend(result["templates"])
        enrollments.append({k: v for k, v in result.items() if k != "templates"} | {"person_id": row["person_id"]})
    write_json(output / "gallery.json", rows)
    write_json(output / "model-manifest.json", engine.manifest)
    gallery = Gallery(rows, engine.model_version)
    report = {"schema": 1, "status": "prototipo-local", "scope": manifest["scope"],
              "dataset_digest": dataset_digest, "model_version": engine.model_version,
              "environment": environment(), "cold_start_ms": engine.cold_start_ms,
              "enrollment": enrollments, "enrolled_people": len(gallery.person_ids),
              "requested_people": len(manifest["enrollment"]), "robot_validated": False,
              "limitations": ["Public images do not validate phone-to-robot recognition",
                              "Wilson intervals assume independent queries; ORL sessions are not verified",
                              "Enrollment and quality limits are provisional until domain evaluation"]}
    if not gallery.person_ids:
        report.update(status="no-valid-enrollments", acceptance=False)
        write_json(output / "report.json", report)
        return report
    validation = collect(engine, root, manifest["validation"], gallery)
    # Bind the policy to data, not runtime scores/timing: numerical roundoff on
    # another CPU must not invalidate a frozen policy for the same corpus.
    validation_content = [digest_file(root / path) for row in manifest["enrollment"] for path in row["images"]]
    validation_content.extend(digest_file(root / row["image"]) for row in manifest["validation"])
    provenance = fingerprint({"split": "validation", "enrollment": manifest["enrollment"],
                              "samples": manifest["validation"], "content": validation_content})
    try:
        if supplied_policy:
            if (supplied_policy.model_version != engine.model_version or supplied_policy.validation_digest != provenance
                    or supplied_policy.enrollment_min_similarity != engine.config.enrollment_min_similarity):
                raise ValueError("Frozen calibration does not match this model/validation corpus")
            policy = supplied_policy
        else:
            policy = calibrate(validation, engine.model_version, provenance, engine.config.enrollment_min_similarity)
    except ValueError as error:
        report.update(status="calibration-failed", error=str(error), acceptance=False)
        write_json(output / "report.json", report)
        return report
    policy = Calibration(**(policy.to_dict() | {"scope": manifest["scope"]}))
    write_json(output / "calibration.json", policy.to_dict())
    if mode == "calibrate":
        report.update(calibration=policy.to_dict(), calibration_id=policy.calibration_id,
                      validation=summarize(validation, policy), acceptance=False)
        write_json(output / "report.json", report)
        return report
    # Test is first extracted only after threshold/margin are frozen.
    test = collect(engine, root, manifest["test"], gallery)
    report.update(calibration=policy.to_dict(), calibration_id=policy.calibration_id,
                  validation=summarize(validation, policy), test=summarize(test, policy))
    per_person = report["test"]["per_person"]
    coverage = (len(gallery.person_ids) >= 5 and len(rows) == 3 * len(manifest["enrollment"])
                and len(per_person) >= 5 and all(c["total"] >= 20 for c in per_person.values())
                and report["test"]["unknown_queries"] >= 100
                and len({s["subject_id"] for s in manifest["test"] if s["person_id"] is None}) >= 5)
    numerical = (report["test"]["known_identification_rate"] >= .95
                 and report["test"]["incorrect_assignments"] == 0)
    report["numerical_quality_target_met"] = numerical
    report["minimum_acceptance_coverage_met"] = coverage
    report["acceptance"] = False  # Hardware/capture/resource acceptance needs real robot evidence.
    report["domain_quality_candidate"] = coverage and numerical and manifest["scope"] == "phone-robot"
    write_json(output / "report.json", report)
    return report


def environment() -> dict:
    return {"python": platform.python_version(), "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "opencv": cv2.__version__, "numpy": np.__version__,
            "runtime": "OpenCV DNN CPU", "capture_latency_included": False}


def benchmark(engine: FaceEngine, images: list[Path], gallery: Gallery, policy: Calibration,
              repeats: int = 100) -> dict:
    if not images or repeats < 10:
        raise ValueError("Benchmark requires input images and at least ten repetitions")
    process = psutil.Process()
    for i in range(5):
        engine.identify(images[i % len(images)], gallery, policy)
    samples, statuses, stages = [], Counter(), {}
    rss = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(.01):
            rss.append(process.memory_info().rss)

    sampler = threading.Thread(target=monitor, daemon=True)
    cpu_start = process.cpu_times()
    start = time.perf_counter()
    sampler.start()
    checkpoints = []
    try:
        for i in range(repeats):
            item = engine.identify(images[i % len(images)], gallery, policy)
            samples.append(item["elapsed_ms"])
            statuses[item["status"]] += 1
            for key, value in item.get("timings_ms", {}).items():
                stages.setdefault(key, []).append(value)
            checkpoints.append(process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    elapsed = time.perf_counter() - start
    cpu_end = process.cpu_times()
    cpu_seconds = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
    middle = len(checkpoints)//2
    drift = float(np.median(checkpoints[middle:]) - np.median(checkpoints[:middle]))
    return {"environment": environment(), "model_version": engine.model_version,
            "calibration_id": policy.calibration_id, "threads": engine.config.threads,
            "gallery_people": len(gallery.person_ids), "warmup": 5, "cold_start_ms": engine.cold_start_ms,
            "latency": latency(samples), "stage_latency": {k: latency(v) for k, v in stages.items()},
            "status_counts": dict(statuses), "peak_sampled_rss_bytes": max(rss + checkpoints),
            "process_lifetime_peak_rss_bytes": peak_rss(), "rss_median_drift_bytes": drift,
            "memory_stability_observed": drift <= 10 * 1024**2,
            "memory_stability_note": "Median second-half minus first-half RSS; <=10 MiB is a local heuristic, not a leak proof",
            "cpu_seconds": cpu_seconds, "wall_seconds": elapsed,
            "mean_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
            "model_bytes": ASSETS["yunet"]["bytes"] + ASSETS[engine.config.variant]["bytes"],
            "capture_and_robot_load_validated": False,
            "latency_target_locally_met": max(samples) <= 2000}


def peak_rss() -> int:
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024
