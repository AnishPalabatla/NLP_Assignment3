"""Tiny table helpers (no tabulate/jinja2 dependency) for building writeup tables from result rows."""

from __future__ import annotations

from typing import Any, Sequence


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def rows_to_markdown(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None, digits: int = 2) -> str:
    if not rows:
        return ""
    columns = list(columns or rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c), digits) for c in columns) + " |")
    return "\n".join(lines)


def rows_to_latex(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None, digits: int = 2) -> str:
    if not rows:
        return ""
    columns = list(columns or rows[0].keys())
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(esc(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(esc(_fmt(r.get(c), digits)) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
