#  MarketShield

**AI-powered Financial News Manipulation Detection System**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)

MarketShield monitors financial news and social media in real time, detects AI-generated or coordinated misinformation, and evaluates whether it could manipulate financial markets.

---

##  What It Does

| Stage | Component | What Happens |
|-------|-----------|--------------|
| 1 | **Data Ingestion** | Pulls RSS feeds + Reddit posts continuously |
| 2 | **NLP Analysis** | Sentiment scoring + AI-text detection (FinBERT + perplexity) |
| 3 | **Propagation Analysis** | Deduplication, clustering, coordinated-account detection via graph |
| 4 | **Market Monitor** | Fetches real-time stock & crypto prices, detects volatility spikes |
| 5 | **Risk Engine** | Combines all signals into a 0–1 composite manipulation score |
| 6 | **Dashboard** | Streamlit UI with live alerts, sentiment/price charts, network graphs |

---

## 🗂️ Project Structure

```
MarketShield/
├── config.py                  # Central settings (pydantic-settings)
├── models.py                  # Shared Pydantic data models
├── database.py                # SQLite persistence layer
├── logger.py                  # Loguru-based logging
├── main.py                    # CLI entry point
│
├── data_ingestion/
│   ├── rss_collector.py       # RSS feed scraper (async)
│   ├── reddit_collector.py    # Reddit PRAW collector
│   └── runner.py              # Orchestrates ingestion loop
│
├── ai_detection/
│   ├── sentiment.py           # FinBERT sentiment analysis
│   ├── ai_detector.py         # AI-generated text detection
│   ├── perplexity.py          # GPT-2 perplexity scorer
│   └── runner.py              # Analysis pipeline orchestrator
│
├── propagation_analysis/
│   ├── deduplicator.py        # Semantic similarity deduplication
│   ├── graph_builder.py       # NetworkX propagation graph
│   └── coordination.py        # Coordinated behaviour detection
│
├── market_analysis/
│   ├── price_fetcher.py       # yfinance + CoinGecko data
│   ├── volatility.py          # Z-score volatility detection
│   ├── correlator.py          # Sentiment ↔ price correlation
│   └── monitor.py             # Continuous price monitoring loop
│
├── risk_engine/
│   ├── scorer.py              # Weighted composite risk score
│   └── alert_manager.py       # Alert threshold and dispatch
│
├── dashboard/
│   ├── app.py                 # Streamlit dashboard main
│   ├── components/            # Reusable UI components
│   └── charts.py              # Plotly chart builders
│
├── simulations/
│   ├── historical_replay.py   # Replay historical misinformation events
│   └── scenarios/             # Pre-built manipulation scenarios
│
├── tests/                     # pytest test suite
├── data/                      # Local data storage (gitignored)
├── pyproject.toml             # Dependencies + tooling config
├── Makefile                   # Developer shortcuts
└── .env.example               # Configuration template
```

---

##  Quick Start

### 1. Prerequisites

- Python 3.11
- Git
- ~4 GB disk (Hugging Face models download on first use)

### 2. Clone & Install

```bash
git clone https://github.com/yourusername/MarketShield.git
cd MarketShield

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — Reddit/Twitter credentials are optional
```

### 4. Initialise Database

```bash
python database.py
```

### 5. Run

```bash
# Check system status
python main.py status

# Start full pipeline (ingestion + market monitor)
python main.py pipeline

# Or start components individually:
python main.py ingest    # Data collection only
python main.py market    # Market monitor only
python main.py analyse   # NLP analysis only

# Launch dashboard (in a separate terminal)
streamlit run dashboard/app.py
```

---

##  API Credentials

MarketShield works **without any API keys** — RSS feeds and market data require no auth. Credentials unlock additional sources:

| Source | Required? | Get Credentials |
|--------|-----------|-----------------|
| RSS Feeds | ✗ No | — |
| Yahoo Finance / CoinGecko | ✗ No | — |
| Reddit | Optional | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| Twitter/X | Optional | [developer.twitter.com](https://developer.twitter.com) |

---

##  Testing

```bash
make test
# or
pytest tests/ -v --cov=.
```

---

## 📊 Risk Score Calculation

The composite manipulation risk score is a weighted combination:

```
Risk = 0.35 × AI_Probability
     + 0.30 × Propagation_Anomaly
     + 0.35 × Market_Impact
```

| Score | Risk Level | Meaning |
|-------|------------|---------|
| 0.0 – 0.39 | 🟢 LOW | Normal activity |
| 0.40 – 0.64 | 🟡 MEDIUM | Worth monitoring |
| 0.65 – 0.84 | 🟠 HIGH | Likely manipulation attempt |
| 0.85 – 1.00 | 🔴 CRITICAL | Active manipulation campaign |

---

## 🔬 Technology Stack

| Layer | Technology |
|-------|------------|
| NLP / Sentiment | 🤗 Transformers, FinBERT |
| AI Text Detection | RoBERTa OpenAI Detector, perplexity scoring |
| Graph Analysis | NetworkX, python-louvain |
| Market Data | yfinance, pycoingecko |
| Storage | SQLite (SQLAlchemy) |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| ML | PyTorch, scikit-learn |

---

##  Disclaimer

MarketShield is a **research and educational project**. It is not financial advice. Detection models have false-positive rates — always apply human judgement before acting on alerts.

