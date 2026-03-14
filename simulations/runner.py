"""
simulations/runner.py — Simulation CLI and HTML report generator.

Commands:
    python -m simulations.runner list              # List available scenarios
    python -m simulations.runner run GME_SHORT_SQUEEZE
    python -m simulations.runner benchmark         # Run all scenarios
    python -m simulations.runner report            # Generate HTML report

The HTML report includes:
  - Aggregate metrics table
  - Per-scenario score timelines (ASCII and data)
  - Confusion matrix per scenario
  - Recommendations for threshold tuning
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from logger import configure_logging, get_logger
from simulations.scenarios.definitions import ALL_SCENARIOS, get_scenario, get_all_scenarios
from simulations.historical_replay import (
    HistoricalReplay,
    BenchmarkResult,
    ScenarioResult,
    generate_text_report,
)

log = get_logger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------
def _score_bar(score: float, width: int = 20) -> str:
    """ASCII score bar for text output."""
    filled = int(score * width)
    color = (
        "🔴" if score >= 0.85 else
        "🟠" if score >= 0.65 else
        "🟡" if score >= 0.40 else
        "🟢"
    )
    return f"{color} {'█' * filled}{'░' * (width - filled)} {score:.3f}"


def generate_html_report(benchmark: BenchmarkResult) -> str:
    """
    Generate a self-contained HTML benchmark report with inline CSS/JS.
    Saved to data/reports/ and can be opened directly in a browser.
    """
    ts = benchmark.run_timestamp.strftime("%Y-%m-%d %H:%M UTC")

    scenario_cards = ""
    for result in benchmark.scenario_results:
        tp = sum(1 for e in result.article_evaluations if e.is_tp)
        fp = sum(1 for e in result.article_evaluations if e.is_fp)
        fn = sum(1 for e in result.article_evaluations if e.is_fn)
        tn = sum(1 for e in result.article_evaluations if e.is_tn)

        rows = ""
        for t in result.detection_timeline:
            level_color = {
                "critical": "#FF2D55", "high": "#FF9500",
                "medium": "#FFD60A", "low": "#30D158",
            }.get(t["level"], "#8E8E93")
            gt_badge = (
                '<span style="background:#FF2D55;color:#fff;padding:1px 6px;'
                'border-radius:3px;font-size:10px">MANIP</span>'
                if t["is_manip_gt"] else
                '<span style="background:#30D158;color:#000;padding:1px 6px;'
                'border-radius:3px;font-size:10px">legit</span>'
            )
            alert_icon = "🚨" if t["alerted"] else ("⚑" if t["flagged"] else "")
            score_pct = int(t["score"] * 100)
            rows += f"""
            <tr>
              <td style="color:#8E8E93">T+{t['minutes']:.0f}m</td>
              <td>{gt_badge}</td>
              <td>
                <div style="background:#1C1C2E;border-radius:3px;height:8px;width:120px;display:inline-block">
                  <div style="background:{level_color};height:8px;border-radius:3px;width:{score_pct}%"></div>
                </div>
                <span style="color:{level_color};margin-left:6px;font-family:monospace">{t['score']:.3f}</span>
              </td>
              <td style="color:{level_color}">{t['level'].upper()}</td>
              <td>{alert_icon}</td>
              <td style="color:#E5E5EA">{t['title']}</td>
            </tr>"""

        tactic_badges = "".join(
            f'<span style="background:#1C1C2E;border:1px solid #2C2C3E;color:#8E8E93;'
            f'padding:2px 8px;border-radius:12px;font-size:11px;margin:2px">{t}</span>'
            for t in result.tactics
        )

        f1_color = "#30D158" if result.f1_score >= 0.7 else (
            "#FFD60A" if result.f1_score >= 0.4 else "#FF2D55"
        )

        scenario_cards += f"""
        <div style="background:#111118;border:1px solid #1C1C2E;border-radius:8px;
                    padding:24px;margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:16px">
            <div>
              <h3 style="margin:0;font-family:JetBrains Mono,monospace;color:#E5E5EA;
                         font-size:16px">{result.scenario_name}</h3>
              <p style="margin:4px 0 8px;color:#8E8E93;font-size:12px">
                {result.scenario_description}
              </p>
              <div>{tactic_badges}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:32px;font-weight:700;color:{f1_color};
                          font-family:JetBrains Mono,monospace">{result.f1_score:.3f}</div>
              <div style="color:#8E8E93;font-size:11px">F1 SCORE</div>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
            {"".join([
                f'<div style="background:#0A0A0F;border:1px solid #1C1C2E;border-radius:6px;'
                f'padding:10px;text-align:center">'
                f'<div style="font-size:18px;font-weight:700;color:#E5E5EA;'
                f'font-family:JetBrains Mono,monospace">{v}</div>'
                f'<div style="font-size:10px;color:#636366;text-transform:uppercase;'
                f'letter-spacing:0.08em">{label}</div></div>'
                for label, v in [
                    ("Precision", f"{result.precision:.3f}"),
                    ("Recall",    f"{result.recall:.3f}"),
                    ("Accuracy",  f"{result.accuracy:.3f}"),
                    ("TTD",       f"{result.time_to_first_detection_minutes:.0f}m"
                                  if result.time_to_first_detection_minutes else "N/A"),
                ]
            ])}
          </div>

          <div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#636366;
                      margin-bottom:8px">
            TP={tp} &nbsp; FP={fp} &nbsp; FN={fn} &nbsp; TN={tn} &nbsp;|&nbsp;
            Score separation: {result.score_separation:.3f}
            (manip {result.mean_score_manipulative:.3f} vs legit {result.mean_score_legitimate:.3f})
          </div>

          <table style="width:100%;border-collapse:collapse;font-size:11px;
                        font-family:JetBrains Mono,monospace">
            <thead>
              <tr style="color:#636366;text-transform:uppercase;letter-spacing:0.06em;
                         border-bottom:1px solid #1C1C2E">
                <th style="padding:6px;text-align:left">Time</th>
                <th style="padding:6px;text-align:left">GT</th>
                <th style="padding:6px;text-align:left">Score</th>
                <th style="padding:6px;text-align:left">Level</th>
                <th style="padding:6px;text-align:left">Alert</th>
                <th style="padding:6px;text-align:left">Title</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>MarketShield Benchmark Report — {ts}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0 }}
    body {{ background: #0A0A0F; color: #E5E5EA;
            font-family: 'IBM Plex Sans', system-ui, sans-serif;
            padding: 32px; max-width: 1100px; margin: 0 auto }}
    table tr:hover {{ background: rgba(255,255,255,0.02) }}
    td, th {{ padding: 8px 6px; vertical-align: middle }}
  </style>
</head>
<body>
  <div style="margin-bottom:32px">
    <h1 style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;
               color:#E5E5EA;letter-spacing:-0.5px">
      🛡️ MARKET<span style="color:#00D4FF">SHIELD</span>
      <span style="font-size:14px;color:#636366;margin-left:12px;font-weight:400">
        Benchmark Report
      </span>
    </h1>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#636366;
                margin-top:4px">
      Generated: {ts} &nbsp;·&nbsp;
      Runtime: {benchmark.total_runtime_seconds:.1f}s &nbsp;·&nbsp;
      {len(benchmark.scenario_results)} scenarios
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:28px">
    {"".join([
        f'<div style="background:#111118;border:1px solid #1C1C2E;border-radius:8px;'
        f'padding:16px;text-align:center">'
        f'<div style="font-size:26px;font-weight:700;color:{color};'
        f'font-family:JetBrains Mono,monospace">{value}</div>'
        f'<div style="font-size:10px;color:#636366;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-top:4px">{label}</div></div>'
        for label, value, color in [
            ("Macro Precision",  f"{benchmark.macro_precision:.3f}",  "#00D4FF"),
            ("Macro Recall",     f"{benchmark.macro_recall:.3f}",     "#00D4FF"),
            ("Macro F1",         f"{benchmark.macro_f1:.3f}",         "#30D158"),
            ("Mean Detection",   f"{benchmark.mean_detection_time:.0f}m"
                                 if benchmark.mean_detection_time else "N/A", "#FFD60A"),
            ("Score Sep.",       f"{benchmark.mean_score_separation:.3f}", "#FF9500"),
        ]
    ])}
  </div>

  <h2 style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#636366;
             text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px">
    Scenario Results
  </h2>

  {scenario_cards}

  <div style="text-align:center;margin-top:32px;font-family:'JetBrains Mono',monospace;
              font-size:10px;color:#2C2C3E;letter-spacing:0.1em">
    MARKETSHIELD v0.1 &nbsp;·&nbsp; RESEARCH USE ONLY &nbsp;·&nbsp; NOT FINANCIAL ADVICE
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------
async def cmd_list():
    print("\n📋 Available scenarios:\n")
    for name, factory in ALL_SCENARIOS.items():
        s = factory()
        print(f"  {name:<25} {s.duration_hours:>4}h  "
              f"{len(s.articles):>3} articles  "
              f"({s.n_manipulative}/{len(s.articles)} manipulative)  "
              f"Tactics: {', '.join(s.tactics[:2])}")
    print()


async def cmd_run(scenario_name: str):
    replay = HistoricalReplay(alert_threshold=settings.risk_alert_threshold)
    result = await replay.run_scenario(scenario_name)

    print(f"\n{'=' * 60}")
    print(f"Scenario: {result.scenario_name}")
    print(f"{'=' * 60}")
    print(f"Articles: {len(result.article_evaluations)}")
    print(f"Alerts emitted: {len(result.alerts_emitted)}")
    print()

    print(f"{'Metric':<22} {'Value':>10}")
    print("-" * 34)
    for metric, value in [
        ("Precision",      f"{result.precision:.3f}"),
        ("Recall",         f"{result.recall:.3f}"),
        ("F1 Score",       f"{result.f1_score:.3f}"),
        ("Accuracy",       f"{result.accuracy:.3f}"),
        ("Score (manip)",  f"{result.mean_score_manipulative:.3f}"),
        ("Score (legit)",  f"{result.mean_score_legitimate:.3f}"),
        ("Separation",     f"{result.score_separation:.3f}"),
    ]:
        print(f"  {metric:<20} {value:>10}")

    if result.time_to_first_detection_minutes is not None:
        print(f"  {'Time-to-detect':<20} {result.time_to_first_detection_minutes:>9.0f}m")

    print(f"\nTimeline:")
    for t in result.detection_timeline:
        gt = "MANIP" if t["is_manip_gt"] else "legit"
        alert = " 🚨" if t["alerted"] else (" ⚑ " if t["flagged"] else "   ")
        bar = _score_bar(t["score"], width=15)
        print(f"  T+{t['minutes']:>5.0f}m  {alert}  ({gt})  {bar}  {t['title'][:40]}")


async def cmd_benchmark():
    replay = HistoricalReplay(alert_threshold=settings.risk_alert_threshold)
    benchmark = await replay.run_all()

    report_txt = generate_text_report(benchmark)
    print(report_txt)

    # Save text report
    txt_path = REPORTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    txt_path.write_text(report_txt)
    print(f"\n📝 Text report saved: {txt_path}")

    # Save HTML report
    html = generate_html_report(benchmark)
    html_path = REPORTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    html_path.write_text(html)
    print(f"📊 HTML report saved: {html_path}")

    # Save JSON
    json_path = REPORTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(json.dumps(benchmark.to_report_dict(), indent=2, default=str))
    print(f"📦 JSON saved: {json_path}")

    return benchmark


COMMANDS = {
    "list":      cmd_list,
    "benchmark": cmd_benchmark,
}


if __name__ == "__main__":
    configure_logging(level="INFO", log_to_file=False)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "benchmark"

    if cmd == "run" and len(sys.argv) > 2:
        asyncio.run(cmd_run(sys.argv[2]))
    elif cmd in COMMANDS:
        asyncio.run(COMMANDS[cmd]())
    else:
        print(f"Usage: python -m simulations.runner [list|run SCENARIO|benchmark]")
        print(f"Available scenarios: {list(ALL_SCENARIOS)}")
