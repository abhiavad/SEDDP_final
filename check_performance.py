#!/usr/bin/env python3
"""
ADCS CSV PERFORMANCE SCORECARD

PURPOSE
-------
Evaluate ADCS performance from Basilisk-exported CSV logs.

This script evaluates three independent performance channels:

1) Detumbling
2) Nadir pointing
3) Velocity alignment

The script computes:

- acquisition time
- post-lock mean
- post-lock standard deviation
- post-lock median
- post-lock maximum
- post-lock in-spec percentage
- post-lock escape count
- post-lock escape rate
- conservative 1σ envelope check

The framework is intended for:
- controller tuning
- controller comparison
- requirement verification
"""

from __future__ import annotations

import io

from dataclasses import dataclass

from pathlib import Path

from simulation_config import OUTPUT_FOLDER_NAME

from typing import Optional

import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

OUTPUT_DIR = Path(
    OUTPUT_FOLDER_NAME
)

CSV_FILE = (
    OUTPUT_DIR
    / "adcs_log.csv"
)

# Base filename for generated reports.
#
# Produces:
#
#   <REPORT_NAME>.csv
#   <REPORT_NAME>.txt

REPORT_NAME = OUTPUT_FOLDER_NAME

# Continuous in-spec hold requirement.

HOLD_TIME_SEC = 600.0


# ==========================================================
# METRIC SPECIFICATION
# ==========================================================

@dataclass(frozen=True)
class MetricSpec:

    name: str

    signal_col: str

    threshold_native: float

    threshold_display: float

    native_unit: str

    display_unit: str

    display_scale: float


# ==========================================================
# REQUIREMENTS
# ==========================================================

METRICS: dict[str, MetricSpec] = {

    "detumbling": MetricSpec(

        name="detumbling",

        signal_col="omega_mag_rad_s",

        threshold_native=np.deg2rad(0.5),

        threshold_display=0.5,

        native_unit="rad/s",

        display_unit="deg/s",

        display_scale=180.0 / np.pi
    ),

    "nadir": MetricSpec(

        name="nadir",

        signal_col="angle_x_nadir_deg",

        threshold_native=30.0,

        threshold_display=30.0,

        native_unit="deg",

        display_unit="deg",

        display_scale=1.0
    ),

    "velocity": MetricSpec(

        name="velocity",

        signal_col="angle_z_vel_deg",

        threshold_native=10.0,

        threshold_display=10.0,

        native_unit="deg",

        display_unit="deg",

        display_scale=1.0
    )
}


# ==========================================================
# CSV LOADING
# ==========================================================

def load_adcs_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load Basilisk ADCS CSV.

    Handles:
    - metadata comment blocks
    - comma-separated formatting
    - whitespace-separated formatting
    - mixed formatting drift
    """

    raw_text = csv_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    data_lines = []

    for line in raw_text.splitlines():

        if not line.strip():
            continue

        if line.lstrip().startswith("#"):
            continue

        data_lines.append(line)

    if not data_lines:

        raise ValueError(
            f"No usable data found in CSV:\n{csv_path}"
        )

    df = pd.read_csv(

        io.StringIO("\n".join(data_lines)),

        sep=r"\s*,\s*|\s+",

        engine="python"
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


# ==========================================================
# HELPERS
# ==========================================================

def finite_mask(
    t: np.ndarray,
    x: np.ndarray
) -> np.ndarray:

    return np.isfinite(t) & np.isfinite(x)


def format_value(
    value,
    unit=""
):

    try:
        v = float(value)

    except Exception:
        return str(value)

    if not np.isfinite(v):
        return "nan"

    if unit:
        return f"{v:.6g} {unit}"

    return f"{v:.6g}"


def pass_fail(flag: bool) -> str:

    return "PASS" if flag else "FAIL"


def compliance_status(pct: float) -> str:

    if not np.isfinite(pct):
        return "UNKNOWN"

    if pct >= 95.0:
        return "EXCELLENT"

    if pct >= 80.0:
        return "ACCEPTABLE"

    if pct >= 50.0:
        return "POOR"

    return "FAILING"


# ==========================================================
# ACQUISITION DETECTION
# ==========================================================

def find_acquisition_window(

    t: np.ndarray,

    inside: np.ndarray,

    hold_time_sec: float

) -> tuple[Optional[int], Optional[int]]:

    n = len(t)

    if n == 0:
        return None, None

    outside_prefix = np.concatenate((
        [0],
        np.cumsum(~inside).astype(int)
    ))

    for i in range(n):

        target_time = t[i] + hold_time_sec

        j = int(np.searchsorted(
            t,
            target_time,
            side="left"
        ))

        if j >= n:
            break

        outside_count = int(
            outside_prefix[j + 1]
            - outside_prefix[i]
        )

        if outside_count == 0:
            return i, j

    return None, None


# ==========================================================
# IN-SPEC PERCENTAGE
# ==========================================================

def weighted_in_spec_percentage(

    t_post: np.ndarray,

    x_post: np.ndarray,

    threshold_native: float

) -> float:

    if len(t_post) < 2:
        return float("nan")

    dt = np.diff(t_post)

    if np.any(dt < 0):

        raise ValueError(
            "Time vector is not monotonic."
        )

    total_time = float(np.sum(dt))

    if total_time <= 0:
        return float("nan")

    inside = x_post[:-1] < threshold_native

    compliant_time = float(
        np.sum(dt[inside])
    )

    return 100.0 * compliant_time / total_time


# ==========================================================
# ESCAPE COUNT
# ==========================================================

def escape_count_from_sequence(

    inside_sequence: np.ndarray

) -> int:

    if len(inside_sequence) < 2:
        return 0

    transitions = np.diff(
        inside_sequence.astype(int)
    )

    return int(
        np.sum(transitions == -1)
    )


# ==========================================================
# METRIC EVALUATION
# ==========================================================

def evaluate_metric(

    df: pd.DataFrame,

    spec: MetricSpec,

    hold_time_sec: float

) -> dict[str, object]:

    required_cols = {
        "time_sec",
        spec.signal_col
    }

    missing = required_cols - set(df.columns)

    if missing:

        raise KeyError(
            f"Missing columns for {spec.name}: {sorted(missing)}"
        )

    t_all = df["time_sec"].to_numpy(dtype=float)

    x_native = df[spec.signal_col].to_numpy(dtype=float)

    mask = finite_mask(
        t_all,
        x_native
    )

    t_all = t_all[mask]

    x_native = x_native[mask]

    if len(t_all) == 0:

        raise ValueError(
            f"No valid samples for {spec.signal_col}"
        )

    order = np.argsort(t_all)

    t_all = t_all[order]

    x_native = x_native[order]

    inside = x_native < spec.threshold_native

    start_idx, end_idx = find_acquisition_window(

        t_all,

        inside,

        hold_time_sec
    )

    result = {

        "mode": spec.name,

        "signal_col": spec.signal_col,

        "threshold_display": spec.threshold_display,

        "threshold_unit": spec.display_unit,

        "hold_time_sec": hold_time_sec,

        "acquired": False,

        "acquisition_time_sec": float("nan"),

        "acquisition_time_min": float("nan"),

        "post_lock_duration_sec": float("nan"),

        "post_lock_samples": 0,

        "post_mean": float("nan"),

        "post_sigma": float("nan"),

        "post_median": float("nan"),

        "post_max": float("nan"),

        "post_1sigma_upper": float("nan"),

        "margin_to_requirement": float("nan"),

        "in_spec_percentage": float("nan"),

        "escape_count": float("nan"),

        "escape_rate_per_hour": float("nan"),

        "pass_1sigma": False,

        "compliance_status": "UNKNOWN",

        "notes": ""
    }

    if start_idx is None or end_idx is None:

        result["notes"] = (
            "No persistent hold window found."
        )

        return result

    acquisition_time_sec = float(
        t_all[start_idx]
    )

    post_start = end_idx + 1

    result["acquired"] = True

    result["acquisition_time_sec"] = acquisition_time_sec

    result["acquisition_time_min"] = (
        acquisition_time_sec / 60.0
    )

    inside_from_acquisition = inside[start_idx:]

    escape_count = escape_count_from_sequence(
        inside_from_acquisition
    )

    if post_start >= len(t_all):

        result["notes"] = (
            "Acquisition succeeded but "
            "no post-lock samples exist."
        )

        result["escape_count"] = int(escape_count)

        return result

    t_post = t_all[post_start:]

    x_post_native = x_native[post_start:]

    x_post_display = (
        x_post_native
        * spec.display_scale
    )

    if len(t_post) >= 2:

        post_duration_sec = float(
            t_post[-1] - t_post[0]
        )

    else:

        post_duration_sec = 0.0

    mean_val = float(
        np.mean(x_post_display)
    )

    sigma_val = float(
        np.std(
            x_post_display,
            ddof=0
        )
    )

    median_val = float(
        np.median(x_post_display)
    )

    max_val = float(
        np.max(x_post_display)
    )

    upper_1sigma = (
        abs(mean_val) + sigma_val
    )

    margin_to_requirement = (
        spec.threshold_display
        - upper_1sigma
    )

    in_spec_pct = weighted_in_spec_percentage(

        t_post,

        x_post_native,

        spec.threshold_native
    )

    if post_duration_sec > 0:

        escape_rate = (
            escape_count
            / (post_duration_sec / 3600.0)
        )

    else:

        escape_rate = float("nan")

    result.update({

        "post_lock_duration_sec":
            post_duration_sec,

        "post_lock_samples":
            int(len(x_post_display)),

        "post_mean":
            mean_val,

        "post_sigma":
            sigma_val,

        "post_median":
            median_val,

        "post_max":
            max_val,

        "post_1sigma_upper":
            upper_1sigma,

        "margin_to_requirement":
            margin_to_requirement,

        "in_spec_percentage":
            in_spec_pct,

        "escape_count":
            int(escape_count),

        "escape_rate_per_hour":
            escape_rate,

        "pass_1sigma":
            bool(
                np.isfinite(upper_1sigma)
                and upper_1sigma
                <= spec.threshold_display
            ),

        "compliance_status":
            compliance_status(in_spec_pct)
    })

    return result


# ==========================================================
# REPORTING
# ==========================================================

def print_result(
    row: dict[str, object]
):

    unit = row["threshold_unit"]

    print("\n=====================================================")

    print(row["mode"].upper())

    print("=====================================================")

    print("\nRequirement")
    print("-----------")

    print(
        f"Threshold                 : "
        f"{format_value(row['threshold_display'], unit)}"
    )

    print(
        f"Hold Duration             : "
        f"{format_value(row['hold_time_sec'], 's')}"
    )

    print("\nAcquisition")
    print("-----------")

    print(
        f"Acquired                  : "
        f"{pass_fail(row['acquired'])}"
    )

    if not row["acquired"]:

        print(
            f"Notes                     : "
            f"{row['notes']}"
        )

        return

    print(
        f"Acquisition Time          : "
        f"{format_value(row['acquisition_time_sec'], 's')} "
        f"({format_value(row['acquisition_time_min'], 'min')})"
    )

    print("\nPost-Lock Statistics")
    print("--------------------")

    print(
        f"Duration                  : "
        f"{format_value(row['post_lock_duration_sec'], 's')}"
    )

    print(
        f"Samples                   : "
        f"{row['post_lock_samples']}"
    )

    print(
        f"Mean                      : "
        f"{format_value(row['post_mean'], unit)}"
    )

    print(
        f"Sigma                     : "
        f"{format_value(row['post_sigma'], unit)}"
    )

    print(
        f"Median                    : "
        f"{format_value(row['post_median'], unit)}"
    )

    print(
        f"Maximum                   : "
        f"{format_value(row['post_max'], unit)}"
    )

    print("\nCompliance")
    print("----------")

    print(
        f"In-Spec Percentage        : "
        f"{format_value(row['in_spec_percentage'], '%')}"
    )

    print(
        f"Compliance Status         : "
        f"{row['compliance_status']}"
    )

    print(
        f"Escape Count              : "
        f"{row['escape_count']}"
    )

    print(
        f"Escape Rate               : "
        f"{format_value(row['escape_rate_per_hour'], 'escapes/hour')}"
    )

    print("\nStatistical Margin")
    print("------------------")

    print(
        f"1σ Envelope               : "
        f"{format_value(row['post_1sigma_upper'], unit)}"
    )

    print(
        f"Requirement Margin        : "
        f"{format_value(row['margin_to_requirement'], unit)}"
    )

    print(
        f"Pass 1σ Requirement       : "
        f"{pass_fail(row['pass_1sigma'])}"
    )

    if row["notes"]:

        print(
            f"\nNotes                     : "
            f"{row['notes']}"
        )


def write_text_report(

    rows: list[dict[str, object]],

    output_path: Path
):

    lines = []

    lines.append(
        "====================================================="
    )

    lines.append(
        "ADCS PERFORMANCE REPORT"
    )

    lines.append(
        "====================================================="
    )

    lines.append("")

    lines.append(
        "OVERALL SUMMARY"
    )

    lines.append(
        "====================================================="
    )

    for row in rows:

        lines.append("")

        lines.append(
            f"{row['mode'].upper()}"
        )

        lines.append(
            f"    Acquisition         : "
            f"{pass_fail(row['acquired'])}"
        )

        lines.append(
            f"    1σ Requirement      : "
            f"{pass_fail(row['pass_1sigma'])}"
        )

        lines.append(
            f"    Compliance          : "
            f"{format_value(row['in_spec_percentage'], '%')}"
        )

        lines.append(
            f"    Status              : "
            f"{row['compliance_status']}"
        )

    for row in rows:

        unit = row["threshold_unit"]

        lines.append("")
        lines.append(
            "====================================================="
        )

        lines.append(
            row["mode"].upper()
        )

        lines.append(
            "====================================================="
        )

        lines.append("")
        lines.append("Requirement")
        lines.append("-----------")

        lines.append(
            f"Threshold                 : "
            f"{format_value(row['threshold_display'], unit)}"
        )

        lines.append(
            f"Hold Duration             : "
            f"{format_value(row['hold_time_sec'], 's')}"
        )

        lines.append("")
        lines.append("Acquisition")
        lines.append("-----------")

        lines.append(
            f"Acquired                  : "
            f"{pass_fail(row['acquired'])}"
        )

        if row["acquired"]:

            lines.append(
                f"Acquisition Time          : "
                f"{format_value(row['acquisition_time_sec'], 's')} "
                f"({format_value(row['acquisition_time_min'], 'min')})"
            )

        lines.append("")
        lines.append("Post-Lock Statistics")
        lines.append("--------------------")

        lines.append(
            f"Duration                  : "
            f"{format_value(row['post_lock_duration_sec'], 's')}"
        )

        lines.append(
            f"Samples                   : "
            f"{row['post_lock_samples']}"
        )

        lines.append(
            f"Mean                      : "
            f"{format_value(row['post_mean'], unit)}"
        )

        lines.append(
            f"Sigma                     : "
            f"{format_value(row['post_sigma'], unit)}"
        )

        lines.append(
            f"Median                    : "
            f"{format_value(row['post_median'], unit)}"
        )

        lines.append(
            f"Maximum                   : "
            f"{format_value(row['post_max'], unit)}"
        )

        lines.append("")
        lines.append("Compliance")
        lines.append("----------")

        lines.append(
            f"In-Spec Percentage        : "
            f"{format_value(row['in_spec_percentage'], '%')}"
        )

        lines.append(
            f"Compliance Status         : "
            f"{row['compliance_status']}"
        )

        lines.append(
            f"Escape Count              : "
            f"{row['escape_count']}"
        )

        lines.append(
            f"Escape Rate               : "
            f"{format_value(row['escape_rate_per_hour'], 'escapes/hour')}"
        )

        lines.append("")
        lines.append("Statistical Margin")
        lines.append("------------------")

        lines.append(
            f"1σ Envelope               : "
            f"{format_value(row['post_1sigma_upper'], unit)}"
        )

        lines.append(
            f"Requirement Margin        : "
            f"{format_value(row['margin_to_requirement'], unit)}"
        )

        lines.append(
            f"Pass 1σ Requirement       : "
            f"{pass_fail(row['pass_1sigma'])}"
        )

        if row["notes"]:

            lines.append("")
            lines.append(
                f"Notes                     : "
                f"{row['notes']}"
            )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ==========================================================
# SUMMARY TABLE
# ==========================================================

def build_summary_dataframe(

    rows: list[dict[str, object]]

) -> pd.DataFrame:

    summary_df = pd.DataFrame(rows)

    preferred_order = [

        "mode",

        "acquired",

        "pass_1sigma",

        "compliance_status",

        "in_spec_percentage",

        "acquisition_time_sec",

        "post_mean",

        "post_sigma",

        "post_1sigma_upper",

        "margin_to_requirement",

        "escape_count",

        "escape_rate_per_hour",

        "post_max"
    ]

    existing = [
        c for c in preferred_order
        if c in summary_df.columns
    ]

    remaining = [
        c for c in summary_df.columns
        if c not in existing
    ]

    return summary_df[
        existing + remaining
    ]


# ==========================================================
# MAIN
# ==========================================================

def main():

    csv_path = CSV_FILE

    output_dir = OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    df = load_adcs_csv(csv_path)

    rows = []

    for metric_name in (

        "detumbling",

        "nadir",

        "velocity"
    ):

        spec = METRICS[metric_name]

        if spec.signal_col not in df.columns:

            print(
                f"Skipping {metric_name}: "
                f"missing column {spec.signal_col}"
            )

            continue

        row = evaluate_metric(

            df,

            spec,

            HOLD_TIME_SEC
        )

        rows.append(row)

        print_result(row)

    if not rows:

        raise RuntimeError(
            "No metrics evaluated."
        )

    summary_df = build_summary_dataframe(
        rows
    )

    csv_output = (
        output_dir
        / f"{REPORT_NAME}.csv"
    )

    summary_df.to_csv(
        csv_output,
        index=False
    )

    txt_output = (
        output_dir
        / f"{REPORT_NAME}.txt"
    )

    write_text_report(

        rows,

        txt_output
    )

    print("\n=====================================================")

    print("SUMMARY TABLE")

    print("=====================================================")

    with pd.option_context(

        "display.max_columns", None,

        "display.width", 200
    ):

        print(
            summary_df.to_string(index=False)
        )

    print("\nSaved CSV summary:")
    print(csv_output)

    print("\nSaved TXT report:")
    print(txt_output)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    main()