"""
MarketShield — Application entry point.

Run modes:
    python main.py ingest       # Start data ingestion loop
    python main.py analyse      # Run NLP analysis on queued articles
    python main.py market       # Start market price monitor
    python main.py pipeline     # Run full pipeline (all stages)
    python main.py status       # Print system status

The dashboard is launched separately:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys

from config import settings
from database import init_db
from logger import configure_logging, get_logger

log = get_logger(__name__)


def _banner() -> None:
    print(
        r"""
 ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗███████╗██╗  ██╗██╗███████╗██╗     ██████╗
 ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗
 ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║   ███████╗███████║██║█████╗  ██║     ██║  ██║
 ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║   ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║
 ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║   ███████║██║  ██║██║███████╗███████╗██████╔╝
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝
        AI Financial News Manipulation Detection System  |  v{version}
""".format(version=settings.version)
    )


def cmd_status() -> None:
    """Print current configuration and system health."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    _banner()

    t = Table(title="MarketShield — System Configuration", show_header=True)
    t.add_column("Setting", style="cyan")
    t.add_column("Value", style="white")

    rows = [
        ("Version", settings.version),
        ("Debug mode", str(settings.debug)),
        ("Log level", settings.log_level),
        ("DB path", str(settings.db_path)),
        ("Stock tickers", ", ".join(settings.stock_tickers)),
        ("Crypto tickers", ", ".join(settings.crypto_tickers)),
        ("AI prob. threshold", f"{settings.ai_probability_threshold:.0%}"),
        ("Risk alert threshold", f"{settings.risk_alert_threshold:.0%}"),
        ("Risk weights sum", f"{settings.risk_weights_sum:.2f}"),
        (
            "Reddit API",
            "✓ configured" if settings.reddit_client_id else "✗ not configured",
        ),
        (
            "Twitter API",
            "✓ configured" if settings.twitter_bearer_token else "✗ not configured",
        ),
    ]
    for k, v in rows:
        t.add_row(k, v)

    console.print(t)


def cmd_ingest() -> None:
    """Start the data ingestion loop (RSS + Reddit)."""
    log.info("Starting data ingestion pipeline…")
    from data_ingestion.runner import run_ingestion_loop
    run_ingestion_loop()


def cmd_analyse() -> None:
    """Run NLP analysis on all unprocessed articles."""
    log.info("Starting NLP analysis pipeline…")
    from ai_detection.runner import run_analysis_pipeline
    run_analysis_pipeline()


def cmd_market() -> None:
    """Start the market price monitor."""
    log.info("Starting market monitor…")
    from market_analysis.monitor import run_market_monitor
    run_market_monitor()


def cmd_pipeline() -> None:
    """Run all pipeline stages sequentially (for development/testing)."""
    import threading

    log.info("Starting full MarketShield pipeline…")

    threads = [
        threading.Thread(target=cmd_ingest, daemon=True),
        threading.Thread(target=cmd_market, daemon=True),
    ]
    for t in threads:
        t.start()

    log.info("All pipeline threads started. Press Ctrl+C to stop.")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Shutting down gracefully…")


COMMANDS = {
    "ingest": cmd_ingest,
    "analyse": cmd_analyse,
    "market": cmd_market,
    "pipeline": cmd_pipeline,
    "status": cmd_status,
}


if __name__ == "__main__":
    configure_logging(level=settings.log_level, log_to_file=True)
    init_db()

    command = sys.argv[1] if len(sys.argv) > 1 else "status"

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[command]()
