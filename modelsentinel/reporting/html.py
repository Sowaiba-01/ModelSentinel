"""Self-contained HTML report generator (no external template dependencies)."""
from __future__ import annotations

import datetime as _dt
import html


def _bar(score: float, grade: str) -> str:
    colors = {"EXCELLENT": "#16a34a", "GOOD": "#65a30d",
              "WARNING": "#d97706", "CRITICAL": "#dc2626"}
    color = colors.get(grade, "#6b7280")
    pct = max(0.0, min(100.0, float(score)))
    return (
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{pct:.1f}%;background:{color}"></div></div>'
    )


def _kv_table(data: dict[str, object]) -> str:
    rows = ""
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            continue
        rows += f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
    return f"<table class='kv'>{rows}</table>" if rows else "<p class='muted'>No data.</p>"


def generate_html_report(
    output_path: str,
    model_name: str = "model",
    health: dict | None = None,
    evaluation: dict | None = None,
    data_quality: dict | None = None,
    drift: dict | None = None,
    schema: dict | None = None,
) -> str:
    """Render a standalone HTML health report and write it to ``output_path``."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall = health.get("overall") if health else None
    grade = health.get("grade", "N/A") if health else "N/A"

    comp_html = ""
    if health:
        for name, c in health.get("components", {}).items():
            comp_html += (
                f"<div class='comp'><div class='comp-head'>"
                f"<span>{html.escape(name.replace('_', ' ').title())}</span>"
                f"<b>{c['score']:.0f} · {c['grade']}</b></div>"
                f"{_bar(c['score'], c['grade'])}</div>"
            )

    issues_html = ""
    if data_quality and data_quality.get("issues"):
        items = "".join(f"<li>{html.escape(str(i))}</li>" for i in data_quality["issues"])
        issues_html = f"<ul class='issues'>{items}</ul>"

    drift_rows = ""
    if drift:
        for feat, d in drift.get("features", {}).items():
            flag = "⚠ drift" if d.get("drift") else "ok"
            detail = d.get("psi", d.get("js_divergence", ""))
            drift_rows += (
                f"<tr><td>{html.escape(str(feat))}</td><td>{html.escape(str(d.get('type','')))}</td>"
                f"<td>{detail}</td><td>{d.get('p_value','')}</td><td>{flag}</td></tr>"
            )
    drift_table = (
        "<table class='kv'><tr><th>feature</th><th>type</th><th>psi/js</th>"
        f"<th>p-value</th><th>status</th></tr>{drift_rows}</table>"
        if drift_rows else "<p class='muted'>No drift analysis provided.</p>"
    )

    schema_html = ""
    if schema:
        ok = "valid ✓" if schema.get("valid") else "issues found ✗"
        schema_html = (
            f"<p>Status: <b>{ok}</b></p>"
            f"<p>Missing columns: {html.escape(str(schema.get('missing_columns', [])))}</p>"
            f"<p>Unexpected columns: {html.escape(str(schema.get('unexpected_columns', [])))}</p>"
            f"<p>Dtype mismatches: {html.escape(str(schema.get('dtype_mismatch', {})))}</p>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModelSentinel Report — {html.escape(model_name)}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
margin:0;background:#0f172a;color:#e2e8f0}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:26px;margin:0 0 4px}}
.muted{{color:#94a3b8}}
.hero{{display:flex;align-items:center;gap:28px;background:#1e293b;
border-radius:16px;padding:24px;margin:20px 0}}
.gauge{{font-size:56px;font-weight:800;line-height:1}}
.pill{{display:inline-block;padding:4px 12px;border-radius:999px;
font-size:13px;font-weight:700;background:#334155}}
.card{{background:#1e293b;border-radius:14px;padding:20px;margin:16px 0}}
.card h2{{font-size:16px;margin:0 0 14px;color:#cbd5e1;text-transform:uppercase;
letter-spacing:.05em}}
.bar-track{{height:9px;background:#334155;border-radius:6px;overflow:hidden;margin-top:6px}}
.bar-fill{{height:100%}}
.comp{{margin-bottom:14px}}
.comp-head{{display:flex;justify-content:space-between;font-size:14px}}
table.kv{{width:100%;border-collapse:collapse;font-size:14px}}
table.kv td,table.kv th{{padding:7px 10px;border-bottom:1px solid #334155;text-align:left}}
table.kv th{{color:#94a3b8;font-weight:600}}
ul.issues{{margin:0;padding-left:18px}} ul.issues li{{margin:4px 0;color:#fca5a5}}
.foot{{margin-top:28px;font-size:12px;color:#64748b;text-align:center}}
</style></head><body><div class="wrap">
<h1>ModelSentinel — Health Report</h1>
<div class="muted">Model: <b>{html.escape(model_name)}</b> · Generated {ts}</div>
<div class="hero">
  <div><div class="gauge">{overall if overall is not None else '—'}</div>
  <div class="muted">/ 100 overall</div></div>
  <div><span class="pill">{html.escape(str(grade))}</span>
  <div class="muted" style="margin-top:8px">Weighted across performance, data quality, drift & reliability.</div></div>
</div>
<div class="card"><h2>Component Scores</h2>{comp_html or "<p class='muted'>No components.</p>"}</div>
<div class="card"><h2>Evaluation</h2>{_kv_table(evaluation) if evaluation else "<p class='muted'>Not run.</p>"}</div>
<div class="card"><h2>Data Quality</h2>{_kv_table({k:v for k,v in (data_quality or {}).items() if k in ('n_rows','n_cols','missing_total','duplicate_rows','score')})}{issues_html}</div>
<div class="card"><h2>Drift</h2>{drift_table}</div>
<div class="card"><h2>Schema</h2>{schema_html or "<p class='muted'>Not checked.</p>"}</div>
<div class="foot">Generated by ModelSentinel · AI Reliability &amp; Observability Toolkit</div>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return output_path
