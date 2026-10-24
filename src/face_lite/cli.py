"""Local tools. Structured outputs go to explicit files, never embeddings in logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .artifacts import ASSETS, download_models
from .core import FaceEngine, PipelineConfig
from .data import load_manifest, prepare_orl, read_json, write_json
from .evaluation import benchmark, environment, evaluate
from .matching import Calibration, Gallery


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="face-lite")
    commands = p.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-models", help="Explicit pinned SHA-256-verified download")
    download.add_argument("--models", type=Path, default=Path("artifacts/models"))
    download.add_argument("--variants", choices=("fp32", "int8"), nargs="+", default=["fp32", "int8"])
    data = commands.add_parser("prepare-orl", help="Download ORL and persist a fixed public functional split")
    data.add_argument("--directory", type=Path, default=Path("data/orl"))
    compare = commands.add_parser("compare", help="Compare independently evaluated FP32/INT8 runs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    for name in ("supabase-check", "camera"):
        c = commands.add_parser(name, help="Read-only Supabase integration / on-demand photo demo")
        c.add_argument("--env-file", type=Path, default=Path(".env"))
        c.add_argument("--models", type=Path, default=Path("artifacts/models"))
        c.add_argument("--config", type=Path)
        c.add_argument("--variant", choices=("fp32", "int8"))
        c.add_argument("--threads", type=int)
        if name == "supabase-check":
            c.add_argument("--output", type=Path, help="Optional aggregate diagnostics; no biometric vectors")
        else:
            c.add_argument("--camera", type=int, default=0, help="Camera index, default 0")
            c.add_argument("--calibration", type=Path, required=True)
    for name in ("doctor", "template", "enroll", "identify", "calibrate", "evaluate", "experiment", "benchmark"):
        c = commands.add_parser(name)
        c.add_argument("--models", type=Path, default=Path("artifacts/models"))
        c.add_argument("--config", type=Path, help="PipelineConfig JSON, defaults are provisional")
        c.add_argument("--variant", choices=("fp32", "int8"), help="Overrides config variant")
        c.add_argument("--threads", type=int, help="Overrides process-wide OpenCV threads")
        c.add_argument("--output", type=Path, required=True)
        if name == "template":
            c.add_argument("--image", type=Path, required=True)
            c.add_argument("--person-id", required=True)
            c.add_argument("--template-id", required=True)
        if name == "enroll":
            c.add_argument("--images", type=Path, nargs=3, required=True, metavar=("FRONT", "LEFT", "RIGHT"))
            c.add_argument("--person-id", required=True)
            c.add_argument("--template-ids", nargs=3, required=True)
        if name in ("identify", "benchmark"):
            c.add_argument("--gallery", type=Path, required=True)
            c.add_argument("--calibration", type=Path, required=True)
        if name == "identify":
            c.add_argument("--image", type=Path, required=True)
        if name in ("calibrate", "evaluate", "experiment", "benchmark"):
            c.add_argument("--manifest", type=Path, required=True)
        if name == "evaluate":
            c.add_argument("--calibration", type=Path, required=True)
        if name == "benchmark":
            c.add_argument("--repeats", type=int, default=100)
    return p


def comparison(baseline: Path, candidate: Path) -> dict:
    a, b = read_json(baseline / "report.json"), read_json(candidate / "report.json")
    ba, bb = read_json(baseline / "benchmark.json"), read_json(candidate / "benchmark.json")
    if a["dataset_digest"] != b["dataset_digest"] or a["scope"] != b["scope"]:
        raise ValueError("Comparison requires the same dataset and split")
    for report, bench in ((a, ba), (b, bb)):
        if report["model_version"] != bench["model_version"] or report["calibration_id"] != bench["calibration_id"]:
            raise ValueError("Benchmark and evaluation versions differ")
        if bench["dataset_digest"] != report["dataset_digest"]:
            raise ValueError("Benchmark and evaluation datasets differ")
    same_work = (ba["gallery_people"] == bb["gallery_people"] and ba["threads"] == bb["threads"]
                 and ba["latency"]["samples"] == bb["latency"]["samples"])
    same_work = same_work and ba["environment"] == bb["environment"]
    if not same_work:
        raise ValueError("Benchmark environment, gallery size and repetitions must match")
    quality = (a["numerical_quality_target_met"] and b["numerical_quality_target_met"]
               and b["test"]["known_identification_rate"] >= a["test"]["known_identification_rate"]
               and b["test"]["incorrect_assignments"] <= a["test"]["incorrect_assignments"])
    resource = (bb["model_bytes"] < ba["model_bytes"] or
                bb["latency"]["p95_ms"] < ba["latency"]["p95_ms"] * .95 or
                bb["peak_sampled_rss_bytes"] < ba["peak_sampled_rss_bytes"] * .95)
    return {"scope": a["scope"], "baseline": a["model_version"], "candidate": b["model_version"],
            "quality_non_degradation_and_target": quality, "resource_improvement": resource,
            "candidate_eligible_locally": bool(quality and resource and bb["latency_target_locally_met"]),
            "robot_approved": False, "automatic_promotion": False,
            "baseline_metrics": {"test": a["test"], "benchmark": ba},
            "candidate_metrics": {"test": b["test"], "benchmark": bb}}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "download-models":
            download_models(args.models, args.variants)
            print(json.dumps({"status": "verified", "models": ["yunet", *args.variants]}))
            return 0
        if args.command == "prepare-orl":
            path = prepare_orl(args.directory)
            load_manifest(path)
            print(json.dumps({"status": "prepared", "manifest": str(path), "scope": "public-functional-only"}))
            return 0
        if args.command == "compare":
            write_json(args.output, comparison(args.baseline, args.candidate))
            print(json.dumps({"status": "compared", "output": str(args.output)}))
            return 0
        config = read_json(args.config) if args.config else {}
        for key in ("variant", "threads"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        engine = FaceEngine(args.models, PipelineConfig.from_dict(config))
        if args.command in ("supabase-check", "camera"):
            from .supabase_gallery import IntegrationError, SupabaseGallerySource, SupabaseSettings
            try:
                settings = SupabaseSettings.load(args.env_file)
                with SupabaseGallerySource(settings) as source:
                    if args.command == "supabase-check":
                        try:
                            _, result = source.fetch(engine.model_version, engine.dimension,
                                                     engine.config.enrollment_min_similarity)
                        except IntegrationError as error:
                            result = error.report()
                        if args.output:
                            write_json(args.output, result)
                        print(json.dumps(result, ensure_ascii=False))
                        return 0 if result["status"] == "compatible" else 2
                    from .camera import run_camera
                    policy = Calibration(**read_json(args.calibration))
                    run_camera(engine, source, policy, camera_index=args.camera)
                    return 0
            except IntegrationError as error:
                print(json.dumps(error.report(), ensure_ascii=False), file=sys.stderr)
                return 2
        if args.command == "doctor":
            result = {"status": "ready", "environment": environment(), "manifest": engine.manifest,
                      "cold_start_ms": engine.cold_start_ms,
                      "model_bytes": ASSETS["yunet"]["bytes"] + ASSETS[engine.config.variant]["bytes"]}
        elif args.command == "template":
            result = engine.template(args.image, args.person_id, args.template_id)
        elif args.command == "enroll":
            result = engine.enroll(args.images, args.person_id, args.template_ids)
        elif args.command in ("calibrate", "evaluate", "experiment"):
            result = evaluate(engine, args.manifest, args.output, args.command, getattr(args, "calibration", None))
            print(json.dumps({"status": result["status"], "output": str(args.output),
                              "enrolled_people": result["enrolled_people"], "robot_validated": False}))
            return 0 if "calibration" in result else 2
        else:
            policy = Calibration(**read_json(args.calibration))
            gallery = Gallery(read_json(args.gallery), engine.model_version, engine.dimension)
            if args.command == "identify":
                result = engine.identify(args.image, gallery, policy)
            else:
                if policy.model_version != engine.model_version:
                    raise ValueError("Calibration incompatible with this model")
                manifest, dataset_digest = load_manifest(args.manifest)
                result = benchmark(engine, [args.manifest.parent / s["image"] for s in manifest["test"]],
                                   gallery, policy, args.repeats)
                result["dataset_digest"] = dataset_digest
        write_json(args.output, result)
        # Never print vectors or personal identifiers to stdout.
        print(json.dumps({"status": result.get("status", "measured"), "output": str(args.output)}))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, cv2.error) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
