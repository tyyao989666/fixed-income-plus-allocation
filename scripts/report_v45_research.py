"""Read-only final v45 evaluation and offline Excel/HTML research reporting."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("low_vol", "balanced", "high_vol")
LAYERS = ("A0", "A1", "A2", "A3")
KEYS = ["profile", "layer", "date"]
DISCLAIMER = (
    "Historical walk-forward research on repeatedly reused history, not untouched OOS. "
    "No guarantee of future returns. Annualization=252; risk-free rate=0. "
    "Solver-stage effective-anchor evidence is not full v45 completion. "
    "Training, checkpoint selection, component gate passage and deployment are distinct. "
    "Cost stress reprices the fixed executed path; it does not retrain policies or rewards. "
    "Close-to-close T+1 accounting is not verified opening-price execution."
)


class ReportInputError(ValueError):
    """Required source evidence is missing or inconsistent."""


def require(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.empty:
        raise ReportInputError(f"{label}: empty table or missing columns {missing}")


def performance(returns):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ReportInputError("Performance needs >=2 finite daily returns above -100%")
    nav = np.r_[1., np.cumprod(1 + values)]
    vol = values.std(ddof=1)
    return dict(days=len(values), total_return=float(nav[-1]-1),
        cagr=float(np.expm1(np.log1p(values).mean()*252)),
        annual_volatility=float(vol*np.sqrt(252)),
        sharpe=float(values.mean()/vol*np.sqrt(252)) if vol > 1e-15 else np.nan,
        max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()))


def validate_daily(frame, profiles=PROFILES):
    require(frame, KEYS + ["net_return", "gross_return", "cost", "turnover"], "daily")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ReportInputError("Daily keys are null/duplicated")
    columns = ["net_return", "gross_return", "cost", "turnover"]
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ReportInputError("Nonfinite daily return/cost/turnover")
    groups = frame.groupby(["profile", "layer"])
    if set(groups.groups) != {(p, a) for p in profiles for a in LAYERS}:
        raise ReportInputError("Need all three profiles and A0/A1/A2/A3")
    calendars = [tuple(g.date.sort_values()) for _, g in groups]
    if len(calendars[0]) < 2 or any(c != calendars[0] for c in calendars):
        raise ReportInputError("Unequal comparison calendars; never silently intersect/drop days")
    if (frame[["cost", "turnover"]] < -1e-12).any().any():
        raise ReportInputError("Negative cost/actual L1 turnover")
    if not np.allclose(frame.cost, frame.turnover*.0005, rtol=0, atol=1e-10):
        raise ReportInputError("Costs must equal actual executed L1 turnover * 5bp")
    if not np.allclose(frame.net_return, frame.gross_return-frame.cost, rtol=0, atol=1e-10):
        raise ReportInputError("Gross minus costs does not reconcile to net return")
    if (frame.net_return <= -1).any():
        raise ReportInputError("Nonpositive wealth")
    return frame.sort_values(KEYS).reset_index(drop=True)


def attribution(daily, holdings):
    """Use upstream actual executed-weight contributions, never mean weights.

    The v45 replay emits return_contribution = post-trade weight * same-day
    return. Prior net wealth links those daily contributions to compounded P&L.
    """
    require(holdings, KEYS + ["asset_code", "module", "weight", "return_contribution"], "holdings")
    h = holdings.copy()
    h["date"] = pd.to_datetime(h.date, errors="raise")
    if h[KEYS+["asset_code", "module"]].isna().any().any() or h.duplicated(KEYS+["asset_code"]).any():
        raise ReportInputError("Holdings keys are null/duplicated")
    if not np.isfinite(h[["weight", "return_contribution"]].to_numpy(float)).all() or (h.weight < -1e-10).any():
        raise ReportInputError("Invalid actual holding/contribution")
    actual = h.groupby(KEYS).agg(weight=("weight", "sum"), gross=("return_contribution", "sum"))
    expected = daily.set_index(KEYS)
    if set(actual.index) != set(expected.index):
        raise ReportInputError("Holdings/daily calendars do not match")
    actual = actual.reindex(expected.index)
    if not np.allclose(actual.weight, 1, atol=1e-8, rtol=0):
        raise ReportInputError("Holdings weights do not sum to one")
    if not np.allclose(actual.gross, expected.gross_return, atol=1e-9, rtol=0):
        raise ReportInputError("Executed-weight contributions do not reconcile to gross")
    wealth = daily.sort_values(KEYS).copy()
    wealth["prior_wealth"] = wealth.groupby(["profile", "layer"]).net_return.transform(
        lambda s: (1+s).cumprod().shift(1, fill_value=1))
    h = h.merge(wealth[KEYS+["prior_wealth"]], on=KEYS, validate="many_to_one")
    h["linked_contribution"] = h.return_contribution*h.prior_wealth
    costs = wealth[KEYS+["cost", "prior_wealth"]].copy()
    costs["asset_code"], costs["module"] = "TRANSACTION_COST", "cost"
    costs["return_contribution"] = -costs.cost
    costs["linked_contribution"] = -costs.cost*costs.prior_wealth
    cols = KEYS+["asset_code", "module", "return_contribution", "linked_contribution"]
    return pd.concat([h[cols], costs[cols]], ignore_index=True)


def paired_bootstrap(candidate, baseline, block=20, draws=3000, seed=45):
    x, y = np.asarray(candidate, float), np.asarray(baseline, float)
    performance(x)
    performance(y)
    if x.shape != y.shape or block < 1 or draws < 1:
        raise ReportInputError("Bootstrap requires paired arrays and positive block/draws")
    if len(x) < block:
        return dict(block_days=block, draws=0, status="unavailable: sample shorter than block")
    rng = np.random.default_rng(seed)
    annual, sharpes = [], []
    for left in range(0, draws, 100):
        count = min(100, draws-left)
        starts = rng.integers(0, len(x), (count, int(np.ceil(len(x)/block))))
        ix = ((starts[..., None]+np.arange(block)) % len(x)).reshape(count, -1)[:, :len(x)]
        a, b = x[ix], y[ix]
        annual.extend(np.expm1(np.log1p(a).mean(1)*252)-np.expm1(np.log1p(b).mean(1)*252))
        sa, sb = a.std(1, ddof=1), b.std(1, ddof=1)
        valid = (sa > 1e-15) & (sb > 1e-15)
        sharpes.extend(((a[valid].mean(1)/sa[valid]-b[valid].mean(1)/sb[valid])*np.sqrt(252)).tolist())
    result = dict(block_days=block, draws=draws, seed=seed,
        status="ok" if len(sharpes) == draws else "unavailable: degenerate Sharpe draws",
        cagr_positive_fraction=float(np.mean(np.asarray(annual)>0)),
        cagr_delta_low=float(np.quantile(annual, .025)), cagr_delta_high=float(np.quantile(annual, .975)))
    if len(sharpes) == draws:
        result.update(sharpe_positive_fraction=float(np.mean(np.asarray(sharpes)>0)),
            sharpe_delta_low=float(np.quantile(sharpes, .025)), sharpe_delta_high=float(np.quantile(sharpes, .975)))
    return result


def evaluate(daily, draws=3000):
    metrics, stress, bootstrap = [], [], []
    for (profile, layer), group in daily.groupby(["profile", "layer"]):
        periods = {"full": group, "2024+": group[group.date >= "2024-01-01"]}
        periods.update({str(y): g for y, g in group.groupby(group.date.dt.year)})
        for period, part in periods.items():
            if len(part) >= 2:
                metrics.append(dict(profile=profile, layer=layer, period=period,
                    start=part.date.min(), end=part.date.max(), **performance(part.net_return)))
        for bps in (5, 10, 20):
            stress.append(dict(profile=profile, layer=layer, cost_bps=bps,
                scope="fixed executed path, not retrained rewards/policy",
                **performance(group.gross_return-group.turnover*bps/10000)))
    for profile, group in daily.groupby("profile"):
        p = group.pivot(index="date", columns="layer", values="net_return").sort_index()
        for period, part in {"full": p, "2024+": p.loc["2024-01-01":]}.items():
            if len(part) < 2:
                continue
            for a, b in (("A2", "A1"), ("A3", "A2"), ("A3", "A1")):
                for block in (20, 60):
                    bootstrap.append(dict(profile=profile, period=period, comparison=f"{a}-{b}",
                        **paired_bootstrap(part[a], part[b], block, draws)))
    return pd.DataFrame(metrics), pd.DataFrame(stress), pd.DataFrame(bootstrap)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return [r for k, v in value.items() for r in flatten(v, f"{prefix}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [r for i, v in enumerate(value) for r in flatten(v, f"{prefix}[{i}]" )]
    return [dict(field=prefix, value=value)]


def read(path):
    if not path.is_file():
        raise ReportInputError(f"Required input missing: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.DataFrame(flatten(json.loads(path.read_text(encoding="utf-8-sig"))))


def training_summary(log):
    require(log, ["epoch"], "training log")
    groups = [c for c in ("model", "stage", "profile", "year", "seed", "run_id") if c in log]
    work = log.copy()
    if not groups:
        work["model"] = "unspecified"
        groups = ["model"]
    rows = []
    for key, part in work.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row.update(last_logged_epoch=part.epoch.max(), logged_epochs=part.epoch.nunique(),
            actual_trained_epochs=None, selected_epoch=None)
        # Source logs may update the selected checkpoint as training proceeds.
        for source, dest in (("trained_epochs", "actual_trained_epochs"), ("selected_epoch", "selected_epoch")):
            if source in part and part[source].notna().any():
                row[dest] = part.sort_values("epoch")[source].dropna().iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def write_workbook(path, tables):
    # Raw million-row predictions stay intact in parquet; Excel is the readable
    # research index, with full source paths/hashes rather than truncated rows.
    selected = {key: tables[key] for key in ("overview", "metrics", "PPO", "stress",
        "bootstrap", "failures", "io", "data", "splits", "training_summary") if key in tables}
    headline = tables["metrics"].query("period == '2024+' and layer == 'A3'").copy()
    headline = headline[["profile", "cagr", "annual_volatility", "sharpe", "max_drawdown", "start", "end"]]
    selected = {"results": headline, **selected}
    selected["attribution"] = tables["attribution"].groupby(
        ["profile", "layer", "module"], as_index=False).linked_contribution.sum()
    selected["prediction_coverage"] = tables["prediction"].groupby(
        ["task", "module", "horizon_days", "model", "seed"], as_index=False).agg(
            rows=("prediction", "size"), finite_predictions=("prediction", "count"),
            active_rows=("active_component", "sum"))
    selected["nav"] = tables["daily"].query("layer == 'A3'").pivot(index="date", columns="profile", values="nav").reset_index()
    gates_path = ROOT / "output/reports/v45_forecast_gates.csv"
    if gates_path.exists():
        gate = pd.read_csv(gates_path)
        fields = [c for c in ("year", "task", "module", "horizon_days", "candidate", "active", "reason", "rank_ic", "delta_rank_ic", "bootstrap_lower_90") if c in gate]
        selected["forecast_gate"] = gate[fields]
    parameter_rows = []
    for name in ("v45_forecast_resolved_config.json", "v45_resolved_portfolio_config.json"):
        source = ROOT / "output/reports" / name
        if source.exists():
            rows = flatten(json.loads(source.read_text(encoding="utf-8")))
            parameter_rows.extend(dict(source=name, **row) for row in rows)
    selected["parameters"] = pd.DataFrame(parameter_rows)
    controller_gates = []
    for source in sorted((ROOT / "output/reports").glob("v45_*_20*_ppo_gate.json")):
        gate = json.loads(source.read_text(encoding="utf-8"))
        controller_gates.append(dict(profile=source.name.split("_20")[0].removeprefix("v45_"),
            year=gate["year"], qualified_seeds=gate.get("qualified_seeds"),
            deployed_gate=gate.get("deployment_gate_passed"), selection_edge=gate.get("selection_edge"),
            reason=gate.get("reason"), source=source.name))
    selected["PPO_gates"] = pd.DataFrame(controller_gates)
    control_source = ROOT / "output/reports/v45_ablation_summary.csv"
    if control_source.exists():
        selected["confidence_controls"] = pd.read_csv(control_source).query("layer in ['R0','R1']")
    temp = ROOT / "output/builders/v45_report"
    temp.mkdir(parents=True, exist_ok=True)
    bundle = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = bundle / "bin/node.exe"
    if not node.exists() or not (bundle / "node_modules/@oai/artifact-tool").exists():
        raise ReportInputError("Bundled artifact-tool runtime unavailable; Excel export was not substituted silently")
    junction = temp / "node_modules"
    if not junction.exists():
        subprocess.run(["powershell", "-NoProfile", "-Command", f"New-Item -ItemType Junction -Path '{junction}' -Target '{bundle / 'node_modules'}' | Out-Null"], check=True)
    payload = {key: json.loads(frame.to_json(orient="split", date_format="iso", default_handler=str)) for key, frame in selected.items()}
    data = temp / "tables.json"
    data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    builder = temp / "build.mjs"
    shutil.copyfile(ROOT / "scripts/report_v45_workbook.mjs", builder)
    subprocess.run([str(node), str(builder), str(data), str(path)], check=True)


def charts(daily, holdings, contrib, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC") if f in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    figures = {}
    for profile, group in daily.groupby("profile"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
        for layer, part in group.groupby("layer"):
            nav = (1+part.net_return).cumprod()
            axes[0, 0].plot(part.date, nav, label=layer, linewidth=1.2)
            peak = np.maximum.accumulate(np.r_[1., nav])[1:]
            axes[0, 1].plot(part.date, (nav/peak-1)*100, label=layer, linewidth=1)
        axes[0, 0].set_title(f"{profile}: cumulative net wealth (initial capital=1)")
        axes[0, 0].legend(ncol=4)
        axes[0, 1].set_title("Drawdown (%)")
        h = holdings[(holdings.profile == profile) & (holdings.layer == "A3")]
        sleeve = h.groupby(["date", "module"]).weight.sum().unstack().sort_index()
        axes[1, 0].stackplot(sleeve.index, sleeve.to_numpy().T, labels=sleeve.columns, alpha=.8)
        axes[1, 0].set_title("A3 actual executed sleeve exposures")
        axes[1, 0].legend(fontsize=7, ncol=3, loc="upper left")
        c = contrib[(contrib.profile == profile) & (contrib.layer == "A3")].groupby("module").linked_contribution.sum()
        axes[1, 1].barh(c.index, c.values*100, color=["#B34747" if x < 0 else "#21867A" for x in c])
        axes[1, 1].set_title("A3 linked P&L contribution (% initial capital)")
        figures[profile] = fig
        fig2, ax = plt.subplots(figsize=(13, max(4, h.asset_code.nunique()*.18)), layout="constrained")
        wide = h.pivot(index="date", columns="asset_code", values="weight").sort_index()
        ax.imshow(wide.to_numpy().T, aspect="auto", cmap="YlGnBu", interpolation="nearest")
        ax.set_yticks(range(len(wide.columns)), wide.columns, fontsize=7)
        ax.set_title(f"{profile}: every investable asset, actual executed weights")
        ticks = np.unique(np.linspace(0, len(wide)-1, min(8, len(wide))).astype(int))
        ax.set_xticks(ticks, [str(wide.index[t].date()) for t in ticks])
        figures[profile+"_assets"] = fig2
    if not training.empty and "epoch" in training:
        loss_cols = [c for c in ("train_loss", "validation_score", "validation_loss") if c in training]
        if loss_cols:
            fig, axes = plt.subplots(1, len(loss_cols), figsize=(13, 3.4), squeeze=False, layout="constrained")
            keys = [c for c in ("stage", "year", "seed", "profile", "run_id") if c in training]
            series = training.groupby(keys, dropna=False) if keys else [("training", training)]
            for key, part in series:
                for ax, col in zip(axes[0], loss_cols):
                    valid = part.dropna(subset=["epoch", col]).sort_values("epoch")
                    ax.plot(valid.epoch, valid[col], linewidth=.7, alpha=.6)
                    ax.set_title(col)
                    ax.set_xlabel("Observed epoch (not selected checkpoint)")
            figures["training"] = fig
    blobs = {}
    import io
    for name, fig in figures.items():
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        blobs[name] = buffer.getvalue()
        plt.close(fig)
    return blobs


def render_report(output, tables, figures):
    output.mkdir(parents=True, exist_ok=True)
    for name, blob in figures.items():
        (output/f"v45_{name}.png").write_bytes(blob)
    write_workbook(output/"v45_research_workbook.xlsx", tables)
    deployed = int(tables["PPO"].get("model_used_days", pd.Series(dtype=float)).sum())
    learned = int(tables["PPO"].get("active_components_days", pd.Series(dtype=float)).sum())
    conclusion = (f"实际部署：预测活跃记录 {learned}，PPO接管记录 {deployed}。"
        + ("本轮 A1=A2=A3，收益不能归功于 iTransformer/PPO；研究晋升未通过。" if learned == 0 and deployed == 0 else "训练、预测门与执行门需分别判断，不能只看最终收益。"))
    headline = tables["metrics"].query("period == '2024+' and layer == 'A3'").copy()
    sections = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>v45 Research</title><style>body{font:13px/1.5 Calibri,'Microsoft YaHei',sans-serif;color:#202829;background:#fff;margin:24px auto;max-width:1480px;padding:0 18px}h1{font-size:25px}h2{font-size:17px;border-bottom:2px solid #18746b;padding-top:18px}table{border-collapse:collapse;width:100%;font-size:11px;font-variant-numeric:tabular-nums}th,td{padding:5px 8px;border-bottom:1px solid #dde4e3;text-align:right}th{background:#e7efed}tr:nth-child(even){background:#f5f7f7}img{width:100%;height:auto}.table{overflow:auto}p{max-width:125ch}@media print{body{margin:0}img{break-inside:avoid}}</style></head><body><h1>v45 Research Evaluation</h1>",
        f"<p>{html.escape(DISCLAIMER)}</p><h2>核心结论</h2><p>{html.escape(conclusion)}</p>",
        "<p>2024-01-02 至 2026-07-15；收益为复合年化、成本为实际成交 L1×5bp。工程检查通过不代表源数据PIT、真实成交与未来表现已获认证。</p>",
        headline.to_html(index=False, float_format=lambda x:f"{x:.5f}")]
    for name in ("overview", "metrics", "PPO", "stress", "bootstrap", "failures"):
        sections.append(f"<h2>{html.escape(name)}</h2><div class='table'>"+tables[name].to_html(index=False, escape=True, na_rep="unavailable", float_format=lambda x:f"{x:.5f}")+"</div>")
    for name, blob in figures.items():
        sections.append(f"<h2>{html.escape(name)}</h2><img alt='{html.escape(name)}' src='data:image/png;base64,{base64.b64encode(blob).decode()}'>")
    sections.append("</body></html>")
    (output/"v45_research_report.html").write_text("\n".join(sections), encoding="utf-8")
    text = ["# v45 研究结果", "", conclusion, "", DISCLAIMER,
        "", "详见 ../../docs/v45_results_and_failure_diagnosis_20261015.md。",
        "", "## 2024年以来三产品", "", "```text", headline.to_string(index=False), "```",
        "", "## 全口径比较", "", "```text", tables["metrics"].to_string(index=False), "```",
        "", "## Training And Deployment", "", "```text", tables["PPO"].to_string(index=False), "```",
        "", "No formal promotion is inferred by this report. See original forecast/PPO/portfolio gates in Excel.",
        "Bootstrap positive fractions are historical resampling frequencies, not future winning probabilities.",
        "10/20bp scenarios reprice a fixed executed path, not capacity or impact estimates.",
        "", "## Missing Evidence And Limitations", "", tables["failures"].to_string(index=False)]
    (output/"v45_research_report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    args = parser.parse_args()
    reports = args.root/"output/reports"
    output = args.output or reports
    sources = []

    def load(path):
        frame = read(path)
        sources.append(path)
        return frame

    tables = {"data": load(reports/"v45_field_audit.csv"),
        "splits": load(reports/"v45_split_dates.csv"),
        "metadata": load(reports/"v45_metadata.json"),
        "prediction": load(args.root/"data_processed/v45_predictions.parquet"),
        "forecast_gate": load(reports/"v45_forecast_report.json"),
        "portfolio_gate": load(reports/"v45_portfolio_gate.json")}
    dailies, holdings = [], []
    for profile in PROFILES:
        manifest_path = reports/f"v45_{profile}_target_manifest.json"
        load(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("a2_a3_same_target") is not True:
            raise ReportInputError(f"{profile}: A2/A3 shared target evidence missing")
        tables[f"targets_{profile}"] = pd.DataFrame(flatten(manifest))
        for layer in LAYERS:
            daily = load(reports/f"v45_{profile}_{layer}_daily.csv")
            daily = daily.rename(columns={"strategy_return": "net_return"})
            if "cost" not in daily:
                daily = daily.rename(columns={"transaction_cost": "cost"})
            daily["profile"], daily["layer"] = profile, layer
            dailies.append(daily)
            h = load(args.root/f"data_processed/v45_{profile}_{layer}_holdings.parquet")
            h["profile"], h["layer"] = profile, layer
            holdings.append(h)
    daily = validate_daily(pd.concat(dailies, ignore_index=True))
    h = pd.concat(holdings, ignore_index=True)
    h["date"] = pd.to_datetime(h.date)
    contrib = attribution(daily, h)
    metrics, stress, boot = evaluate(daily, args.bootstrap_draws)
    logs = []
    for pattern in ("*v45*training*.csv", "v45_*_ppo_training.jsonl"):
        for path in sorted(reports.glob(pattern)):
            log = load(path)
            log["source_log"] = path.name
            logs.append(log)
    training = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    neural = []
    for path in sorted((args.root/"output/models/v45_forecast").glob("*/*.csv")):
        log = load(path)
        log["source_log"] = str(path.relative_to(args.root))
        log["run_id"] = str(path.relative_to(args.root))
        neural.append(log)
    if not training.empty:
        training["run_id"] = training.apply(lambda r: f"{r.get('profile')}/{r.get('year')}/{r.get('seed')}/{r.get('lr')}/{r.get('entropy_weight')}", axis=1)
    if neural:
        training = pd.concat([training, *neural], ignore_index=True)
    usage = daily.groupby(["profile", "layer"]).agg(days=("date", "size"),
        actual_trade_days=("turnover", lambda s: int((s>1e-8).sum())))
    for field in ("model_used", "active_components"):
        if field in daily:
            usage[field+"_days"] = daily.groupby(["profile", "layer"])[field].apply(lambda s:int((s>0).sum()))
    failures = [{"item": "Formal promotion", "status": "not established by reporting; inspect original gates"},
        {"item": "Unexecuted sensitivities", "status": "extra delay/proxy exclusion/retrained reward stress unavailable unless upstream artifacts supplied"}]
    if training.empty:
        failures.append({"item": "Training logs", "status": "unavailable; cannot infer actual/selected epochs from deployment"})
    # Include all v45 tabular/JSON evidence, including failed and zero-gate experiments.
    for path in sorted(reports.glob("v45_*")):
        if path in sources or path.suffix not in (".csv", ".json", ".jsonl") or path.name.startswith("v45_research"):
            continue
        name = "input_"+str(len(tables))
        tables[name] = load(path)
    io_rows = []
    for path in dict.fromkeys(sources):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        io_rows.append(dict(role="input", path=str(path), bytes=path.stat().st_size, sha256=digest))
    for name in ("v45_research_workbook.xlsx", "v45_research_report.html", "v45_research_report.md"):
        io_rows.append(dict(role="output", path=str(output/name), bytes=None, sha256="not self-hashed"))
    overview = pd.DataFrame([dict(item="Research boundary", value=DISCLAIMER),
        dict(item="Common start", value=str(daily.date.min().date())),
        dict(item="Common end", value=str(daily.date.max().date())),
        dict(item="Standard Sharpe", value="mean(daily net return)/sample std * sqrt(252), rf=0"),
        dict(item="CAGR", value="exp(mean(log(1+daily net return))*252)-1"),
        dict(item="Attribution", value="actual executed-weight contributions linked by prior net wealth; costs separate")])
    ordered = dict(overview=overview, metrics=metrics, PPO=usage.reset_index(), stress=stress,
        bootstrap=boot, failures=pd.DataFrame(failures), io=pd.DataFrame(io_rows),
        holdings=h, attribution=contrib, daily=daily, training=training)
    if not training.empty and "epoch" in training:
        ordered["training_summary"] = training_summary(training)
    ordered.update(tables)
    render_report(output, ordered, charts(daily, h, contrib, training))
    print(f"v45 research report written to {output}; no upstream source files modified.")


if __name__ == "__main__":
    try:
        main()
    except ReportInputError as exc:
        raise SystemExit(f"REPORT BLOCKED: {exc}") from exc
