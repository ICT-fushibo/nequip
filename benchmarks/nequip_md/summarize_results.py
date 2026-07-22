#!/usr/bin/env python3
"""Summarize benchmark JSON files and calculate the E0/E1/B0/B1 speedups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


MODES = ("E0", "E1", "B0", "B1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="directory containing json/*.json")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    json_dir = results_dir / "json" if (results_dir / "json").is_dir() else results_dir
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON result files found in {json_dir}")

    records: list[dict[str, Any]] = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") in MODES and "milliseconds_per_step" in record:
            records.append(record)
    if not records:
        raise ValueError(f"No E0/E1/B0/B1 benchmark records found in {json_dir}")

    systems = sorted({str(record["system_label"]) for record in records})
    report: dict[str, Any] = {"results_dir": str(results_dir), "systems": {}}
    csv_rows = []
    warnings: list[str] = []

    for system in systems:
        system_records = [r for r in records if str(r["system_label"]) == system]
        mode_summaries: dict[str, Any] = {}
        for mode in MODES:
            mode_records = [r for r in system_records if r["mode"] == mode]
            if not mode_records:
                warnings.append(f"{system}: missing mode {mode}")
                continue
            milliseconds = [float(r["milliseconds_per_step"]) for r in mode_records]
            mode_summary = summary(milliseconds)
            mode_summary["n_atoms"] = int(mode_records[0]["n_atoms"])
            mode_summary["model_hashes"] = sorted(
                {
                    value
                    for value in (
                        r.get("selected_model_sha256") for r in mode_records
                    )
                    if value is not None
                }
            )
            validation_statuses = sorted(
                {
                    r.get("numerical_validation", {}).get("status", "not_run")
                    for r in mode_records
                }
            )
            mode_summary["numerical_validation_statuses"] = validation_statuses
            validation_passed = all(
                r.get("numerical_validation", {}).get("validation_passed") is True
                for r in mode_records
            )
            mode_summary["numerical_validation_passed"] = validation_passed
            checkpoint_errors = mode_records[0].get("numerical_validation", {}).get(
                "checkpoint_potential_energy_abs_error_ev"
            )
            mode_summary["checkpoint_potential_energy_abs_error_ev"] = (
                checkpoint_errors
            )
            mode_summaries[mode] = mode_summary
            csv_rows.append(
                {
                    "system": system,
                    "n_atoms": mode_summary["n_atoms"],
                    "mode": mode,
                    "count": mode_summary["count"],
                    "median_ms_per_step": mode_summary["median"],
                    "mean_ms_per_step": mode_summary["mean"],
                    "std_ms_per_step": mode_summary["std"],
                    "q25_ms_per_step": mode_summary["q25"],
                    "q75_ms_per_step": mode_summary["q75"],
                    "numerical_validation_statuses": ";".join(validation_statuses),
                    "numerical_validation_passed": validation_passed,
                    "step1_potential_energy_abs_error_ev": (
                        checkpoint_errors.get("1") if checkpoint_errors else None
                    ),
                    "step50_potential_energy_abs_error_ev": (
                        checkpoint_errors.get("50") if checkpoint_errors else None
                    ),
                    "step100_potential_energy_abs_error_ev": (
                        checkpoint_errors.get("100") if checkpoint_errors else None
                    ),
                    "step1000_potential_energy_abs_error_ev": (
                        checkpoint_errors.get("1000") if checkpoint_errors else None
                    ),
                }
            )
            if mode != "E0" and not validation_passed:
                warnings.append(
                    f"{system}: mode {mode} failed or lacks E0 trajectory validation"
                )

        medians = {
            mode: float(values["median"]) for mode, values in mode_summaries.items()
        }
        speedups = {}
        comparisons = {
            "E1_vs_E0": ("E0", "E1"),
            "B0_vs_E0": ("E0", "B0"),
            "B1_vs_B0": ("B0", "B1"),
            "B1_vs_E0": ("E0", "B1"),
        }
        for name, (reference, optimized) in comparisons.items():
            if reference in medians and optimized in medians:
                speedups[name] = medians[reference] / medians[optimized]

        if "B0" in mode_summaries and "B1" in mode_summaries:
            if mode_summaries["B0"]["model_hashes"] != mode_summaries["B1"]["model_hashes"]:
                warnings.append(f"{system}: B0 and B1 did not use identical model hashes")

        report["systems"][system] = {
            "modes": mode_summaries,
            "speedups": speedups,
        }

    report["warnings"] = warnings
    output_prefix = (
        Path(args.output_prefix).expanduser()
        if args.output_prefix
        else results_dir / "summary"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_output = output_prefix.with_suffix(".json")
    csv_output = output_prefix.with_suffix(".csv")
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {json_output}")
    print(f"Wrote {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
