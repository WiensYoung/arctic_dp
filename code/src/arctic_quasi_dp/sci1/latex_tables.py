"""Generate LaTeX booktabs tables from experiment summary CSVs.

Generates publication-quality LaTeX tables for:
- Table I: Scenario Matrix
- Table II: Controller Ablation Summary
- Table III: Main Performance Metrics
- Table IV: Runtime Feasibility
- Table V: Data Provenance
- Table VI: HOCBF Diagnostics

Reference:
- IEEE/ACM table formatting guidelines
- booktabs LaTeX package documentation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    s = str(s)
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    return s


def _format_number(val: float, decimals: int = 2, pct: bool = False) -> str:
    """Format a number for LaTeX table."""
    if pd.isna(val):
        return "--"
    if pct:
        return f"{val * 100:.{decimals}f}\\%"
    return f"{val:.{decimals}f}"


def _format_mean_ci(mean: float, ci_low: float, ci_high: float, decimals: int = 2) -> str:
    """Format mean [CI_low, CI_high] for LaTeX."""
    if pd.isna(mean):
        return "--"
    return f"{mean:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def generate_latex_table(
    df: pd.DataFrame,
    columns: List[Dict[str, str]],
    caption: str,
    label: str,
    font_size: str = "\\small",
    resize_to_width: Optional[str] = None,
) -> str:
    """Generate a LaTeX booktabs table from a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of column specs, each with:
            - "key": DataFrame column name
            - "header": LaTeX column header
            - "format": "float", "int", "pct", "str", "mean_ci"
            - "decimals": (optional) number of decimal places
        caption: Table caption
        label: Table label
        font_size: Font size command (e.g., \\small, \\footnotesize)
        resize_to_width: If set, wrap in \\resizebox{width}{!}{...}

    Returns:
        LaTeX table string
    """
    n_cols = len(columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    if font_size:
        lines.append(font_size)

    if resize_to_width:
        lines.append(f"\\resizebox{{{resize_to_width}}}{{!}}{{%")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    headers = [c.get("header", c["key"]) for c in columns]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            key = col["key"]
            fmt = col.get("format", "str")
            decimals = col.get("decimals", 2)

            if key not in row:
                cells.append("--")
                continue

            val = row[key]
            if fmt == "float":
                cells.append(_format_number(float(val), decimals))
            elif fmt == "int":
                cells.append(str(int(val)) if not pd.isna(val) else "--")
            elif fmt == "pct":
                cells.append(_format_number(float(val), decimals, pct=True))
            elif fmt == "str":
                cells.append(_escape_latex(str(val)))
            elif fmt == "mean_ci":
                ci_low = row.get(f"{key}_ci_low", row.get(f"{key}_ci_lo", val))
                ci_high = row.get(f"{key}_ci_high", row.get(f"{key}_ci_hi", val))
                cells.append(_format_mean_ci(float(val), float(ci_low), float(ci_high), decimals))
            else:
                cells.append(_escape_latex(str(val)))

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if resize_to_width:
        lines.append("}%")  # close resizebox

    lines.append("\\end{table}")

    return "\n".join(lines)


def csv_to_latex(
    csv_path: Path,
    output_path: Path,
    columns: List[Dict[str, str]],
    caption: str,
    label: str,
    **kwargs,
) -> Path:
    """Convert a CSV file to a LaTeX table.

    Args:
        csv_path: Input CSV file
        output_path: Output .tex file
        columns: Column specifications
        caption: Table caption
        label: Table label
        **kwargs: Additional arguments to generate_latex_table

    Returns:
        Path to output .tex file
    """
    df = pd.read_csv(csv_path)
    latex_str = generate_latex_table(df, columns, caption, label, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")

    return output_path


# Pre-defined column specs for standard tables

MAIN_METRICS_COLUMNS = [
    {"key": "controller", "header": "Controller", "format": "str"},
    {"key": "rms_position_error_m_mean", "header": "RMS Error (m)", "format": "mean_ci", "decimals": 1},
    {"key": "p95_position_error_m_mean", "header": "P95 Error (m)", "format": "mean_ci", "decimals": 1},
    {"key": "safety_violation_time_s_mean", "header": "Safety Viol. (s)", "format": "mean_ci", "decimals": 1},
    {"key": "tail_position_cvar95_m_mean", "header": "CVaR$_{95}$ (m)", "format": "mean_ci", "decimals": 1},
    {"key": "thrust_saturation_ratio_mean", "header": "Thrust Sat.", "format": "mean_ci", "decimals": 3},
    {"key": "failure_mean", "header": "Failure Rate", "format": "mean_ci", "decimals": 3},
]

RUNTIME_COLUMNS = [
    {"key": "controller", "header": "Controller", "format": "str"},
    {"key": "solver_time_p95_ms", "header": "Solver P95 (ms)", "format": "mean_ci", "decimals": 1},
    {"key": "solver_time_mean_ms", "header": "Solver Mean (ms)", "format": "mean_ci", "decimals": 1},
    {"key": "infeasible_rate", "header": "Infeasible Rate (\\%)", "format": "mean_ci", "decimals": 1},
]

ABLATION_COLUMNS = [
    {"key": "controller", "header": "Controller", "format": "str"},
    {"key": "rms_position_error_m", "header": "RMS Error (m)", "format": "float", "decimals": 3},
    {"key": "safety_violation_rate", "header": "Safety Viol. (\\%)", "format": "float", "decimals": 1},
    {"key": "delta_rms_pct", "header": "$\\Delta$ RMS (\\%)", "format": "float", "decimals": 1},
]

SCENARIO_MATRIX_COLUMNS = [
    {"key": "scenario_id", "header": "ID", "format": "str"},
    {"key": "group", "header": "Group", "format": "str"},
    {"key": "description", "header": "Description", "format": "str"},
    {"key": "ice_concentration", "header": "SIC", "format": "float", "decimals": 2},
    {"key": "ice_thickness_m", "header": "SIT (m)", "format": "float", "decimals": 2},
    {"key": "ice_drift_speed_mps", "header": "Drift (m/s)", "format": "float", "decimals": 2},
]


def generate_all_latex_tables(
    summary_dir: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate all LaTeX tables from summary CSVs.

    Args:
        summary_dir: Directory containing summary CSVs
        output_dir: Directory for .tex output

    Returns:
        List of paths to generated .tex files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Table III: Main metrics
    main_csv = summary_dir / "aggregate_metrics_ci95.csv"
    if main_csv.exists():
        p = csv_to_latex(
            main_csv,
            output_dir / "table3_main_metrics.tex",
            MAIN_METRICS_COLUMNS,
            caption="Main performance metrics (mean $\\pm$ 95\\% CI across seeds).",
            label="tab:main_metrics",
        )
        generated.append(p)

    # Table IV: Runtime
    runtime_csv = summary_dir / "runtime_summary.csv"
    if runtime_csv.exists():
        p = csv_to_latex(
            runtime_csv,
            output_dir / "table4_runtime.tex",
            RUNTIME_COLUMNS,
            caption="Solver runtime feasibility (mean $\\pm$ 95\\% CI).",
            label="tab:runtime",
        )
        generated.append(p)

    # Table VI: HOCBF diagnostics (aggregated by controller)
    hocbf_csv = summary_dir / "hocbf_diagnostics.csv"
    if hocbf_csv.exists():
        hocbf_df = pd.read_csv(hocbf_csv)
        if "controller" in hocbf_df.columns and len(hocbf_df) > 0:
            # 按 controller 聚合 HOCBF 诊断数据
            num_cols = [c for c in hocbf_df.columns
                       if c != "controller" and pd.api.types.is_numeric_dtype(hocbf_df[c])]
            if num_cols:
                hocbf_agg = hocbf_df.groupby("controller")[num_cols].mean().reset_index()
            else:
                hocbf_agg = hocbf_df.drop_duplicates(subset=["controller"])
        else:
            hocbf_agg = hocbf_df
        hocbf_cols = [
            {"key": "controller", "header": "Controller", "format": "str"},
            {"key": "safety_filter_hocbf_margin_min", "header": "HOCBF Margin", "format": "mean_ci", "decimals": 4},
            {"key": "safety_filter_slack_mean", "header": "Mean Slack", "format": "mean_ci", "decimals": 4},
            {"key": "hocbf_soft_certificate_rate", "header": "Certificate Rate", "format": "mean_ci", "decimals": 3},
            {"key": "safety_set_h_min", "header": "Safety h(x)", "format": "mean_ci", "decimals": 2},
            {"key": "mode_switch_count", "header": "Mode Switches", "format": "mean_ci", "decimals": 1},
        ]
        # 写入聚合后的 CSV 供 LaTeX 使用
        agg_csv = summary_dir / "hocbf_diagnostics_aggregated.csv"
        hocbf_agg.to_csv(agg_csv, index=False)
        p = csv_to_latex(
            agg_csv,
            output_dir / "table6_hocbf_diagnostics.tex",
            hocbf_cols,
            caption="HOCBF safety filter diagnostics (mean across scenarios).",
            label="tab:hocbf",
        )
        generated.append(p)

    return generated
