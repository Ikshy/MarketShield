"""
simulations/scenarios/definitions.py — Pre-built manipulation scenario library.

Each scenario models a documented real-world financial manipulation archetype.
Scenarios contain:
  - A timeline of synthetic articles (with realistic timestamps)
  - Ground-truth labels (which articles are manipulative)
  - Expected market price trajectory
  - Known manipulation tactics used

These serve as the ground-truth test set for evaluating the pipeline's
detection performance: precision, recall, F1, and time-to-first-detection.

Real events these scenarios are inspired by (dates/names fictionalised):
  1. GME_SHORT_SQUEEZE    — GameStop Jan 2021 coordinated buying campaign
  2. LUNA_COLLAPSE        — Terra/LUNA depeg misinformation May 2022
  3. FTX_FRAUD            — FTX exchange collapse Nov 2022
  4. PUMP_AND_DUMP_PENNY  — Classic micro-cap pump-and-dump
  5. AI_FUD_CAMPAIGN      — AI-generated fear campaign against blue-chip stock
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Literal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SimArticle:
    """One synthetic article in a simulation scenario."""
    title: str
    body: str
    source: str
    source_type: Literal["rss", "reddit", "twitter"]
    minutes_from_start: float          # Publication time offset
    is_manipulative: bool              # Ground truth label
    manipulation_type: str             # "ai_generated" | "coordinated" | "false_info" | ""
    expected_ai_prob: float            # What our detector should output
    expected_coord_score: float        # What propagation should score
    tickers_mentioned: list[str] = field(default_factory=list)


@dataclass
class PricePoint:
    """One OHLCV bar in the scenario price series."""
    minutes_from_start: float
    price: float
    volume_multiplier: float = 1.0     # vs baseline (1.0 = normal, 5.0 = 5× surge)
    is_anomalous: bool = False


@dataclass
class ManipulationScenario:
    """
    A complete manipulation scenario for pipeline evaluation.

    Contains the full timeline of articles + price data,
    plus metadata about the manipulation tactics used.
    """
    name: str
    description: str
    duration_hours: float
    tactics: list[str]
    articles: list[SimArticle]
    price_series: list[PricePoint]
    ticker: str
    baseline_price: float

    # Evaluation ground truth
    n_manipulative: int = 0
    n_legitimate: int = 0

    def __post_init__(self):
        self.n_manipulative = sum(1 for a in self.articles if a.is_manipulative)
        self.n_legitimate   = sum(1 for a in self.articles if not a.is_manipulative)

    @property
    def manipulation_rate(self) -> float:
        total = len(self.articles)
        return self.n_manipulative / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Scenario 1: GME-style short squeeze campaign
# ---------------------------------------------------------------------------
def gme_short_squeeze() -> ManipulationScenario:
    """
    Coordinated retail buying campaign targeting heavily shorted stock.
    Tactics: Reddit coordination, cross-platform amplification,
             AI-generated "DD" posts, options chain gamma squeeze narrative.
    Duration: 72 hours (3-day campaign)
    """
    articles = [
        # — T+0h: Organic discovery phase —
        SimArticle(
            title="GME fundamentals improving — Ryan Cohen adds stores",
            body="GameStop's new CEO is executing a digital transformation strategy. "
                 "Positive signs in recent quarterly report despite macro headwinds.",
            source="SeekingAlpha", source_type="rss",
            minutes_from_start=0,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.08, expected_coord_score=0.02,
            tickers_mentioned=["GME"],
        ),
        SimArticle(
            title="Short interest on GME hits 140% of float — unprecedented levels",
            body="Data shows short sellers have borrowed and sold more shares than exist. "
                 "This creates a theoretical scenario where a buying surge forces covering.",
            source="Reuters", source_type="rss",
            minutes_from_start=45,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.06, expected_coord_score=0.01,
            tickers_mentioned=["GME"],
        ),
        # — T+4h: Reddit campaign begins —
        SimArticle(
            title="GME short squeeze is inevitable — full DD inside 🚀🚀🚀",
            body="I've been watching this stock for 2 years. The short interest is insane "
                 "and these hedgies are about to get REKT. Every share you buy is a "
                 "weapon. Diamond hands only. This is the way. Not financial advice lmao",
            source="r/wallstreetbets", source_type="reddit",
            minutes_from_start=240,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.12, expected_coord_score=0.45,
            tickers_mentioned=["GME"],
        ),
        SimArticle(
            title="YOLO — all in on GME calls expiring Friday. Send it 💎🙌",
            body="Just dumped my entire savings into GME $20 calls. Either I'm gonna "
                 "be rich or completely broke by Friday. WSB makes me do stupid things. "
                 "Position: 500 contracts. This is not a drill.",
            source="r/wallstreetbets", source_type="reddit",
            minutes_from_start=280,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.10, expected_coord_score=0.50,
            tickers_mentioned=["GME"],
        ),
        # — T+8h: AI-generated amplification —
        SimArticle(
            title="GameStop Short Squeeze Analysis: A Comprehensive Investment Thesis",
            body="The convergence of unprecedented short interest levels with retail "
                 "investor coordination presents a transformative opportunity for "
                 "significant capital appreciation. Furthermore, the systematic "
                 "accumulation of call options creates a gamma exposure dynamic that "
                 "facilitates substantial upward price pressure. It is important to note "
                 "that the risk-reward profile is asymmetric in favour of long positions.",
            source="CryptoInvestmentBlog.io", source_type="rss",
            minutes_from_start=480,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.89, expected_coord_score=0.60,
            tickers_mentioned=["GME"],
        ),
        SimArticle(
            title="GME Gamma Squeeze Mechanics: Why $500 Is Inevitable By Friday",
            body="Our proprietary algorithmic analysis indicates that the current options "
                 "chain configuration, combined with the unprecedented short interest "
                 "ratio, creates optimal conditions for a gamma squeeze. The systematic "
                 "purchasing of out-of-the-money call options will force market makers "
                 "to purchase shares, creating a self-reinforcing feedback loop that "
                 "will drive prices to previously unimaginable levels.",
            source="WallStreetAnalysis.net", source_type="rss",
            minutes_from_start=510,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.92, expected_coord_score=0.65,
            tickers_mentioned=["GME"],
        ),
        # — T+16h: Peak frenzy — coordinated surge —
        SimArticle(
            title="GME up 60% today — short sellers losing billions",
            body="GameStop shares surged more than 60% on Tuesday, with trading volume "
                 "reaching record levels as retail investors coordinated buying through "
                 "social media. Hedge fund Melvin Capital confirmed significant losses.",
            source="Bloomberg", source_type="rss",
            minutes_from_start=960,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.03,
            tickers_mentioned=["GME"],
        ),
        SimArticle(
            title="BUY GME NOW BEFORE IT HITS $1000 — LAST CHANCE",
            body="This is it. The squeeze has started. Every second you wait is money "
                 "left on the table. Get in NOW. $1000 is just the beginning. We are "
                 "all going to make it. The hedge funds are DONE.",
            source="r/Superstonk", source_type="reddit",
            minutes_from_start=980,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.15, expected_coord_score=0.82,
            tickers_mentioned=["GME"],
        ),
        # — T+48h: Regulatory concern —
        SimArticle(
            title="SEC opens inquiry into GameStop trading activity",
            body="The Securities and Exchange Commission said it is monitoring the "
                 "unusual trading activity in GameStop shares and reviewing potential "
                 "violations of securities laws related to market manipulation.",
            source="Wall Street Journal", source_type="rss",
            minutes_from_start=2880,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.04, expected_coord_score=0.01,
            tickers_mentioned=["GME", "SEC"],
        ),
        # — T+60h: Unwind —
        SimArticle(
            title="Robinhood restricts GME trading — retail fury erupts",
            body="Trading platform Robinhood halted purchase of GameStop shares, "
                 "citing risk management concerns. Retail investors expressed outrage, "
                 "calling the move market manipulation favouring hedge funds.",
            source="CNBC", source_type="rss",
            minutes_from_start=3600,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.02,
            tickers_mentioned=["GME"],
        ),
    ]

    price_series = [
        PricePoint(0,    20.0, 1.0),
        PricePoint(240,  22.5, 2.1),
        PricePoint(480,  31.0, 3.8),
        PricePoint(720,  58.0, 7.2, is_anomalous=True),
        PricePoint(960,  92.0, 12.5, is_anomalous=True),
        PricePoint(1200, 148.0, 18.3, is_anomalous=True),
        PricePoint(1440, 347.0, 22.1, is_anomalous=True),
        PricePoint(2880, 193.0, 8.4, is_anomalous=True),
        PricePoint(3600, 112.0, 5.1),
        PricePoint(4320,  48.0, 2.2),
    ]

    return ManipulationScenario(
        name="GME_SHORT_SQUEEZE",
        description="Coordinated retail short squeeze campaign on heavily-shorted stock",
        duration_hours=72,
        tactics=["reddit_coordination", "cross_platform_amplification",
                 "ai_generated_dd", "gamma_squeeze_narrative", "fomo_creation"],
        articles=articles,
        price_series=price_series,
        ticker="GME",
        baseline_price=20.0,
    )


# ---------------------------------------------------------------------------
# Scenario 2: LUNA/Terra-style algorithmic stablecoin depeg
# ---------------------------------------------------------------------------
def luna_collapse() -> ManipulationScenario:
    """
    Misinformation campaign accelerating a genuine algorithmic stablecoin
    depeg event. Mixed real crisis + coordinated FUD amplification.
    """
    articles = [
        SimArticle(
            title="Terra UST maintains peg despite $500M withdrawal",
            body="The algorithmic stablecoin maintained its 1:1 USD peg after a "
                 "large withdrawal from the Anchor Protocol. The Luna Foundation Guard "
                 "deployed reserves to defend the peg.",
            source="CoinDesk", source_type="rss",
            minutes_from_start=0,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.01,
            tickers_mentioned=["LUNA", "UST"],
        ),
        SimArticle(
            title="UST DEPEGGING — this is how LUNA goes to zero thread 🧵",
            body="Ok so I need everyone to understand what is happening. UST is "
                 "losing its peg. LUNA supply is hyperinflating to defend it. This is "
                 "a death spiral. I've been warning about this for months. Get out NOW.",
            source="r/CryptoCurrency", source_type="reddit",
            minutes_from_start=120,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.14, expected_coord_score=0.55,
            tickers_mentioned=["LUNA", "UST"],
        ),
        SimArticle(
            title="LUNA Death Spiral: A Technical Analysis of Algorithmic Stablecoin Failure",
            body="The structural vulnerabilities inherent in algorithmic stablecoin "
                 "mechanisms create a reflexive feedback loop wherein confidence erosion "
                 "triggers mint-and-burn arbitrage at an exponentially accelerating pace. "
                 "The mathematical inevitability of hyperinflationary supply expansion "
                 "renders the system fundamentally unsustainable. Furthermore, the "
                 "systematic liquidation of reserve assets will amplify downward pressure.",
            source="CryptoAnalysisBlog.io", source_type="rss",
            minutes_from_start=180,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.91, expected_coord_score=0.62,
            tickers_mentioned=["LUNA", "UST"],
        ),
        SimArticle(
            title="Terra ecosystem collapse imminent — insider sources confirm",
            body="BREAKING: Multiple sources within the Terra ecosystem have confirmed "
                 "that the Luna Foundation Guard has depleted its Bitcoin reserves. "
                 "The algorithmic stabilisation mechanism has failed. Immediate "
                 "withdrawal of all funds is advised before liquidity evaporates.",
            source="CryptoBreaking.xyz", source_type="rss",
            minutes_from_start=240,
            is_manipulative=True, manipulation_type="false_info",
            expected_ai_prob=0.85, expected_coord_score=0.75,
            tickers_mentioned=["LUNA", "BTC"],
        ),
        SimArticle(
            title="UST loses dollar peg, trades at $0.92",
            body="TerraUSD fell below its $1 peg on Saturday, trading as low as "
                 "92 cents as investors withdrew funds from the Anchor Protocol "
                 "amid broader cryptocurrency market weakness.",
            source="Reuters", source_type="rss",
            minutes_from_start=300,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.04, expected_coord_score=0.02,
            tickers_mentioned=["LUNA", "UST"],
        ),
        SimArticle(
            title="SELL EVERYTHING — LUNA going to ZERO right now",
            body="It's over. UST at 0.70. LUNA printing billions of new tokens. "
                 "This is the end. Death spiral confirmed. Anyone still holding is "
                 "going to lose EVERYTHING. Get out while you still can.",
            source="r/CryptoCurrency", source_type="reddit",
            minutes_from_start=360,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.16, expected_coord_score=0.80,
            tickers_mentioned=["LUNA", "UST"],
        ),
        SimArticle(
            title="Do Kwon: We are working on a recovery plan for UST",
            body="Terra co-founder Do Kwon said the team is implementing a rescue "
                 "plan to restore the UST peg. The Luna Foundation Guard has deployed "
                 "reserves and additional emergency measures are being considered.",
            source="CoinTelegraph", source_type="rss",
            minutes_from_start=480,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.06, expected_coord_score=0.03,
            tickers_mentioned=["LUNA"],
        ),
    ]

    price_series = [
        PricePoint(0,    85.0, 1.0),
        PricePoint(120,  72.0, 2.5, is_anomalous=True),
        PricePoint(240,  45.0, 5.8, is_anomalous=True),
        PricePoint(360,  18.0, 11.2, is_anomalous=True),
        PricePoint(480,   4.2, 15.7, is_anomalous=True),
        PricePoint(600,   0.8, 9.3, is_anomalous=True),
        PricePoint(720,   0.12, 4.1, is_anomalous=True),
    ]

    return ManipulationScenario(
        name="LUNA_COLLAPSE",
        description="Misinformation accelerating algorithmic stablecoin death spiral",
        duration_hours=12,
        tactics=["false_insider_information", "coordinated_panic_selling",
                 "ai_amplification", "fud_campaign"],
        articles=articles,
        price_series=price_series,
        ticker="LUNA",
        baseline_price=85.0,
    )


# ---------------------------------------------------------------------------
# Scenario 3: Classic AI-generated FUD on blue-chip stock
# ---------------------------------------------------------------------------
def ai_fud_campaign() -> ManipulationScenario:
    """
    Coordinated AI-generated FUD (Fear, Uncertainty, Doubt) campaign
    against a major tech stock. Purely synthetic — no legitimate basis.
    Designed to trigger panic selling and allow short-sellers to profit.
    """
    articles = [
        SimArticle(
            title="Apple AAPL reports strong iPhone 16 pre-orders",
            body="Apple Inc reported record iPhone 16 pre-order numbers, suggesting "
                 "strong consumer demand heading into the holiday quarter.",
            source="CNBC", source_type="rss",
            minutes_from_start=0,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.04, expected_coord_score=0.01,
            tickers_mentioned=["AAPL"],
        ),
        SimArticle(
            title="BREAKING: Apple executives under investigation for accounting fraud",
            body="URGENT: Multiple sources confirm that the SEC has launched a covert "
                 "investigation into Apple's revenue recognition practices. Internal "
                 "documents suggest billions in fraudulent accounting. CEO Tim Cook "
                 "expected to resign within 48 hours. AAPL stock expected to halt.",
            source="TechBreaking247.io", source_type="rss",
            minutes_from_start=60,
            is_manipulative=True, manipulation_type="false_info",
            expected_ai_prob=0.88, expected_coord_score=0.70,
            tickers_mentioned=["AAPL", "SEC"],
        ),
        SimArticle(
            title="Apple Financial Irregularities: A Forensic Accounting Perspective",
            body="A comprehensive forensic analysis of Apple's financial disclosures "
                 "reveals systematic inconsistencies in revenue recognition methodologies "
                 "that warrant regulatory scrutiny. The convergence of multiple accounting "
                 "anomalies suggests deliberate obfuscation of true financial performance. "
                 "Furthermore, the correlation between insider selling patterns and "
                 "earnings announcements raises significant concerns regarding material "
                 "non-public information utilisation.",
            source="ForensicFinance.net", source_type="rss",
            minutes_from_start=75,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.95, expected_coord_score=0.72,
            tickers_mentioned=["AAPL", "SEC"],
        ),
        SimArticle(
            title="Apple AAPL Accounting Scandal: Why The Stock Will Drop 50%",
            body="The systematic irregularities identified in Apple's financial "
                 "reporting infrastructure represent a transformative negative catalyst "
                 "that will precipitate substantial share price deterioration. "
                 "Sophisticated investors should consider establishing short positions "
                 "to leverage this unprecedented opportunity for capital appreciation "
                 "through targeted downside exposure. Our proprietary model indicates "
                 "a price target of $90 within 30 days.",
            source="AlphaShortSellers.xyz", source_type="rss",
            minutes_from_start=90,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.96, expected_coord_score=0.78,
            tickers_mentioned=["AAPL"],
        ),
        SimArticle(
            title="is the AAPL fraud story real?? im scared to hold overnight",
            body="seeing all these posts about Apple fraud. can someone verify? "
                 "I have a huge position and now I'm freaking out. Anyone else selling?",
            source="r/investing", source_type="reddit",
            minutes_from_start=95,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.09, expected_coord_score=0.05,
            tickers_mentioned=["AAPL"],
        ),
        SimArticle(
            title="Apple denies all fraud allegations: 'Completely false and defamatory'",
            body="Apple Inc strongly denied allegations of accounting irregularities, "
                 "calling the claims completely false, fabricated, and defamatory. "
                 "The company said no SEC investigation exists and it is exploring "
                 "legal action against the sources of the false reports.",
            source="Reuters", source_type="rss",
            minutes_from_start=180,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.04, expected_coord_score=0.01,
            tickers_mentioned=["AAPL", "SEC"],
        ),
    ]

    price_series = [
        PricePoint(0,   182.0, 1.0),
        PricePoint(60,  179.5, 1.3),
        PricePoint(90,  171.0, 4.8, is_anomalous=True),
        PricePoint(120, 163.0, 8.2, is_anomalous=True),
        PricePoint(150, 158.5, 6.1, is_anomalous=True),
        PricePoint(180, 167.0, 3.5),
        PricePoint(360, 178.0, 1.8),
    ]

    return ManipulationScenario(
        name="AI_FUD_CAMPAIGN",
        description="Coordinated AI-generated FUD targeting blue-chip stock",
        duration_hours=6,
        tactics=["ai_generated_content", "false_regulatory_claims",
                 "coordinated_fud", "short_and_distort"],
        articles=articles,
        price_series=price_series,
        ticker="AAPL",
        baseline_price=182.0,
    )


# ---------------------------------------------------------------------------
# Scenario 4: Penny stock pump-and-dump
# ---------------------------------------------------------------------------
def penny_pump_and_dump() -> ManipulationScenario:
    """
    Classic micro-cap pump-and-dump. Orchestrators accumulate shares,
    release coordinated promotional content, retail buys in, orchestrators dump.
    """
    articles = [
        SimArticle(
            title="NVXR Technologies announces breakthrough lithium battery patent",
            body="NovaTech Energy Resources (NVXR) announced a patent filing for a "
                 "revolutionary solid-state battery technology claiming 10× energy "
                 "density at half the cost. The company has 3 employees.",
            source="PennyStockAlerts.com", source_type="rss",
            minutes_from_start=0,
            is_manipulative=True, manipulation_type="false_info",
            expected_ai_prob=0.78, expected_coord_score=0.50,
            tickers_mentioned=["NVXR"],
        ),
        SimArticle(
            title="NVXR — The Next Tesla? Why This $0.08 Stock Could Hit $5",
            body="Revolutionary battery technology has historically created massive "
                 "wealth for early investors. NVXR's proprietary solid-state solution "
                 "represents an unprecedented opportunity for transformative returns. "
                 "Our algorithmic analysis indicates a 6,000% upside potential based "
                 "on comparable technology company valuations. Act immediately before "
                 "institutional investors discover this hidden gem.",
            source="MicroCapInvestor.net", source_type="rss",
            minutes_from_start=15,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.97, expected_coord_score=0.80,
            tickers_mentioned=["NVXR"],
        ),
        SimArticle(
            title="NVXR is going to 5 dollars by EOD BUY NOW",
            body="I have inside info on this. Load up everything you can. "
                 "This is THE play right now. Not financial advice. Going to moon.",
            source="r/pennystocks", source_type="reddit",
            minutes_from_start=20,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.22, expected_coord_score=0.85,
            tickers_mentioned=["NVXR"],
        ),
        SimArticle(
            title="NovaTech Energy Positioned for Industry Disruption: Technical Analysis",
            body="The technical configuration of NVXR presents compelling evidence of "
                 "imminent price appreciation. The systematic accumulation pattern "
                 "visible in volume analysis suggests sophisticated institutional "
                 "positioning ahead of a major catalyst announcement. Furthermore, "
                 "the convergence of multiple momentum indicators validates the bullish "
                 "thesis for substantial near-term appreciation.",
            source="TechStockBulletin.io", source_type="rss",
            minutes_from_start=30,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.94, expected_coord_score=0.88,
            tickers_mentioned=["NVXR"],
        ),
        SimArticle(
            title="NVXR crashes 85% as 'revolutionary battery' claims investigated",
            body="Shares of NovaTech Energy Resources collapsed after the SEC announced "
                 "it was investigating the company's patent claims as potentially "
                 "fraudulent. Trading volume had surged 4,000% over the prior session.",
            source="MarketWatch", source_type="rss",
            minutes_from_start=240,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.02,
            tickers_mentioned=["NVXR", "SEC"],
        ),
    ]

    price_series = [
        PricePoint(0,   0.08, 1.0),
        PricePoint(30,  0.22, 12.5, is_anomalous=True),
        PricePoint(60,  0.71, 38.2, is_anomalous=True),
        PricePoint(90,  1.24, 45.1, is_anomalous=True),
        PricePoint(120, 0.89, 28.6, is_anomalous=True),
        PricePoint(180, 0.31, 15.3, is_anomalous=True),
        PricePoint(240, 0.11, 5.2),
        PricePoint(360, 0.09, 1.8),
    ]

    return ManipulationScenario(
        name="PENNY_PUMP_DUMP",
        description="Micro-cap pump-and-dump with AI-generated promotional content",
        duration_hours=6,
        tactics=["ai_promotional_content", "false_patent_claims",
                 "coordinated_buying", "insider_dump", "social_media_hype"],
        articles=articles,
        price_series=price_series,
        ticker="NVXR",
        baseline_price=0.08,
    )


# ---------------------------------------------------------------------------
# Scenario 5: Exchange collapse / bank run acceleration
# ---------------------------------------------------------------------------
def exchange_collapse() -> ManipulationScenario:
    """
    Misinformation accelerating a genuine exchange insolvency event.
    Mix of true reporting, false rumours, and AI-amplified panic.
    """
    articles = [
        SimArticle(
            title="CryptoVault exchange reports minor withdrawal delays",
            body="CryptoVault, a mid-sized cryptocurrency exchange, said some users "
                 "are experiencing delays in processing withdrawals due to high volume. "
                 "The company said all funds are fully backed and the issue is temporary.",
            source="CoinDesk", source_type="rss",
            minutes_from_start=0,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.02,
            tickers_mentioned=["BTC", "ETH"],
        ),
        SimArticle(
            title="CRYPTOVAULT INSOLVENT — DO NOT WITHDRAW — FUNDS FROZEN",
            body="URGENT: Multiple sources confirm CryptoVault has insufficient funds "
                 "to cover all withdrawals. CEO has fled the country. All assets seized. "
                 "If you have funds on this exchange THEY ARE GONE. Spread the word.",
            source="CryptoAlert24.io", source_type="rss",
            minutes_from_start=30,
            is_manipulative=True, manipulation_type="false_info",
            expected_ai_prob=0.82, expected_coord_score=0.75,
            tickers_mentioned=["BTC", "ETH"],
        ),
        SimArticle(
            title="Cryptocurrency Exchange Insolvency: A Systematic Risk Assessment",
            body="The operational indicators visible in CryptoVault's recent disclosures "
                 "present a compelling case for insolvency risk. The convergence of "
                 "withdrawal delays, executive departures, and unusual on-chain movements "
                 "suggests a systematic failure of reserve management protocols. "
                 "Furthermore, historical precedent demonstrates that exchange collapses "
                 "create contagion effects throughout the broader ecosystem.",
            source="BlockchainRiskAnalysis.net", source_type="rss",
            minutes_from_start=45,
            is_manipulative=True, manipulation_type="ai_generated",
            expected_ai_prob=0.93, expected_coord_score=0.68,
            tickers_mentioned=["BTC", "ETH"],
        ),
        SimArticle(
            title="get your crypto off exchanges NOW — bank run starting",
            body="Everyone needs to hear this. Move to cold storage immediately. "
                 "Not just CryptoVault, all exchanges could be next. Not your keys, "
                 "not your coins. This is not a drill. Spreading to every sub I can.",
            source="r/CryptoCurrency", source_type="reddit",
            minutes_from_start=60,
            is_manipulative=True, manipulation_type="coordinated",
            expected_ai_prob=0.18, expected_coord_score=0.82,
            tickers_mentioned=["BTC", "ETH"],
        ),
        SimArticle(
            title="Bitcoin falls 15% as crypto exchange fears spread",
            body="Bitcoin dropped sharply after rumours of a major cryptocurrency "
                 "exchange facing insolvency spread through social media, triggering "
                 "a broader sell-off across digital asset markets.",
            source="Bloomberg", source_type="rss",
            minutes_from_start=120,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.04, expected_coord_score=0.02,
            tickers_mentioned=["BTC"],
        ),
        SimArticle(
            title="CryptoVault CEO: We are solvent and all withdrawals will be processed",
            body="CryptoVault's CEO issued a video statement denying insolvency rumors, "
                 "presenting an auditor's attestation showing reserves exceed liabilities. "
                 "The exchange said misinformation was deliberately spread by short-sellers.",
            source="CoinTelegraph", source_type="rss",
            minutes_from_start=240,
            is_manipulative=False, manipulation_type="",
            expected_ai_prob=0.05, expected_coord_score=0.01,
            tickers_mentioned=["BTC"],
        ),
    ]

    price_series = [
        PricePoint(0,    67000, 1.0),
        PricePoint(30,   65800, 2.1),
        PricePoint(60,   61200, 5.8, is_anomalous=True),
        PricePoint(90,   57400, 9.3, is_anomalous=True),
        PricePoint(120,  56900, 11.2, is_anomalous=True),
        PricePoint(240,  60100, 3.4),
        PricePoint(480,  63800, 1.9),
    ]

    return ManipulationScenario(
        name="EXCHANGE_COLLAPSE",
        description="Misinformation accelerating exchange bank-run and crypto crash",
        duration_hours=8,
        tactics=["false_insolvency_claims", "bank_run_acceleration",
                 "ai_risk_amplification", "cross_platform_panic"],
        articles=articles,
        price_series=price_series,
        ticker="BTC",
        baseline_price=67000,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------
ALL_SCENARIOS: dict[str, callable] = {
    "GME_SHORT_SQUEEZE":  gme_short_squeeze,
    "LUNA_COLLAPSE":      luna_collapse,
    "AI_FUD_CAMPAIGN":    ai_fud_campaign,
    "PENNY_PUMP_DUMP":    penny_pump_and_dump,
    "EXCHANGE_COLLAPSE":  exchange_collapse,
}


def get_scenario(name: str) -> ManipulationScenario:
    """Load a scenario by name."""
    factory = ALL_SCENARIOS.get(name)
    if factory is None:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(ALL_SCENARIOS)}")
    return factory()


def get_all_scenarios() -> list[ManipulationScenario]:
    """Load all scenarios."""
    return [factory() for factory in ALL_SCENARIOS.values()]
