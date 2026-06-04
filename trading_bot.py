#!/usr/bin/env python3
"""
AI Trading Bot — Alpaca + Claude
Multi-source news (Alpaca + Financial Times), full day-trading / swing / position analysis.
"""

import os
import sys
import json
import time
import argparse
import logging
import schedule
import pytz
import feedparser
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import NewsRequest, StockLatestQuoteRequest

# ── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PAPER_TRADING     = os.getenv("PAPER_TRADING", "true").lower() == "true"
CHECK_INTERVAL    = int(os.getenv("CHECK_INTERVAL_MINUTES", "30"))
MAX_POSITIONS     = int(os.getenv("MAX_TOTAL_POSITIONS", "20"))

_watchlist_raw = os.getenv("WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,SPY,QQQ,AMD,NFLX,JPM,GS,BAC")
WATCHLIST = [s.strip().upper() for s in _watchlist_raw.split(",") if s.strip()]

ET = pytz.timezone("America/New_York")

FT_RSS_FEEDS = [
    "https://www.ft.com/rss/home",
    "https://www.ft.com/markets?format=rss",
]

# ── Clients ───────────────────────────────────────────────────────────────────

def build_clients():
    missing = [k for k, v in {
        "ALPACA_API_KEY": ALPACA_API_KEY,
        "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    }.items() if not v or v.startswith("your_")]

    if missing:
        log.error(f"Missing or unconfigured keys in .env: {', '.join(missing)}")
        sys.exit(1)

    trader     = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
    news       = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    stock_data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    ai         = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        acct = trader.get_account()
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        log.info(f"Alpaca [{mode}] connected — portfolio value: ${float(acct.portfolio_value):,.2f}")
    except Exception as e:
        log.error(f"Alpaca connection failed: {e}")
        sys.exit(1)

    return trader, news, stock_data, ai


# ── Market-hours guard ────────────────────────────────────────────────────────

def in_trading_window() -> bool:
    """Mon–Fri, 8:00 AM – 5:30 PM ET."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=8,  minute=0,  second=0, microsecond=0)
    end   = now.replace(hour=17, minute=30, second=0, microsecond=0)
    return start <= now <= end


def session_phase() -> str:
    """Return the current market session phase label."""
    now  = datetime.now(ET)
    hour = now.hour + now.minute / 60.0
    if hour < 9.5:
        return "pre_market"
    elif hour < 10.5:
        return "opening"          # high vol, gap-fills, momentum
    elif hour < 11.5:
        return "mid_morning"      # trend establishing
    elif hour < 14.0:
        return "lunch_lull"       # low volume
    elif hour < 15.5:
        return "afternoon"        # institutional activity
    elif hour < 16.0:
        return "power_hour"       # high vol, trend amplification
    else:
        return "after_hours"


# ── News: Alpaca ──────────────────────────────────────────────────────────────

def fetch_alpaca_news(news_client: NewsClient, lookback_hours: int = 3) -> list[dict]:
    since = datetime.now(pytz.utc) - timedelta(hours=lookback_hours)
    try:
        response = news_client.get_news(
            NewsRequest(start=since, limit=50, include_content=False)
        )
        raw = response.news if hasattr(response, "news") else list(response)
        unwrapped = []
        for item in raw:
            if isinstance(item, tuple):
                val = item[1] if len(item) > 1 else item[0]
                if isinstance(val, list):
                    unwrapped.extend(val)
                else:
                    unwrapped.append(val)
            else:
                unwrapped.append(item)

        def _field(a, key, default=""):
            return (a.get(key, default) if isinstance(a, dict) else getattr(a, key, default)) or default

        articles = []
        for a in unwrapped:
            articles.append({
                "source":    "Alpaca/" + (_field(a, "source") or "news"),
                "headline":  _field(a, "headline"),
                "summary":   _field(a, "summary"),
                "symbols":   _field(a, "symbols") or [],
                "published": str(_field(a, "created_at")),
            })
        log.info(f"Alpaca news: {len(articles)} articles (last {lookback_hours}h)")
        return articles
    except Exception as e:
        log.warning(f"Alpaca news fetch failed: {e}")
        return []


# ── News: Financial Times ─────────────────────────────────────────────────────

def fetch_ft_news(lookback_hours: int = 4) -> list[dict]:
    cutoff = datetime.now(pytz.utc) - timedelta(hours=lookback_hours)
    articles: list[dict] = []

    for feed_url in FT_RSS_FEEDS:
        try:
            feed = feedparser.parse(
                feed_url,
                agent="Mozilla/5.0 (compatible; TradingBot/1.0; +https://github.com/trading-bot)",
                request_headers={"Accept": "application/rss+xml, application/xml, text/xml"},
            )
            if feed.bozo and not feed.entries:
                log.warning(f"FT RSS parse issue ({feed_url}): {feed.bozo_exception}")
                continue

            count = 0
            for entry in feed.entries:
                # Parse publish time
                published_parsed = getattr(entry, "published_parsed", None)
                if published_parsed:
                    pub_dt = datetime(*published_parsed[:6], tzinfo=pytz.utc)
                    if pub_dt < cutoff:
                        continue
                    pub_str = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
                else:
                    pub_str = getattr(entry, "published", "")

                title   = getattr(entry, "title", "").strip()
                summary = getattr(entry, "summary", "").strip()
                # Strip HTML tags from summary (FT sometimes includes them)
                if "<" in summary:
                    import re
                    summary = re.sub(r"<[^>]+>", " ", summary).strip()

                if not title:
                    continue

                articles.append({
                    "source":    "Financial Times",
                    "headline":  title,
                    "summary":   summary[:400] if summary else "",
                    "symbols":   [],  # Claude extracts tickers from text
                    "published": pub_str,
                })
                count += 1

            if count:
                log.info(f"FT RSS ({feed_url.split('/')[-1] or 'home'}): {count} articles")

        except Exception as e:
            log.warning(f"FT RSS fetch failed ({feed_url}): {e}")

    return articles


# ── News: combined ────────────────────────────────────────────────────────────

def fetch_all_news(news_client: NewsClient, lookback_hours: int = 3) -> list[dict]:
    alpaca  = fetch_alpaca_news(news_client, lookback_hours)
    ft      = fetch_ft_news(lookback_hours + 1)

    # Deduplicate by headline similarity (exact matches only — rough dedup)
    seen: set[str] = set()
    combined: list[dict] = []
    for art in alpaca + ft:
        key = art["headline"].lower()[:80]
        if key not in seen:
            seen.add(key)
            combined.append(art)

    log.info(f"Total news articles combined: {len(combined)} (Alpaca: {len(alpaca)}, FT: {len(ft)})")
    return combined


# ── Prices ────────────────────────────────────────────────────────────────────

def fetch_prices(stock_data_client: StockHistoricalDataClient, symbols: list[str]) -> dict:
    stock_syms = [s for s in symbols if "/" not in s and s]
    if not stock_syms:
        return {}
    try:
        quotes = stock_data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=stock_syms)
        )
        prices = {}
        for sym, q in quotes.items():
            price = getattr(q, "ask_price", None) or getattr(q, "bid_price", None)
            if price and float(price) > 0:
                prices[sym] = round(float(price), 4)
        log.info(f"Fetched prices for {len(prices)}/{len(stock_syms)} symbols")
        return prices
    except Exception as e:
        log.warning(f"Price fetch failed: {e} — continuing without prices")
        return {}


# ── Portfolio ─────────────────────────────────────────────────────────────────

def get_portfolio(trading_client: TradingClient) -> dict:
    acct      = trading_client.get_account()
    positions = trading_client.get_all_positions()
    return {
        "buying_power":    float(acct.buying_power),
        "portfolio_value": float(acct.portfolio_value),
        "cash":            float(acct.cash),
        "positions": [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2),
                "current_price":   float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            }
            for p in positions
        ],
    }


# ── Claude analysis ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite quantitative trader and portfolio manager at a top-tier hedge fund.
Your mandate: maximize risk-adjusted returns by analyzing multi-source financial news,
real-time prices, portfolio state, and market session context across all time horizons.

═══════════════════════════════════════════════════════
TRADING HORIZONS
═══════════════════════════════════════════════════════
• day_trade  — Open AND close within the same session. Best for: earnings surprises,
               FDA approvals, M&A announcements, gap-ups on high volume, short squeezes.
               CRITICAL: All day_trade positions MUST be closed before 15:45 ET.
• swing      — Hold 1–5 days. Best for: trend continuations, sector rotations,
               macro developments, post-earnings drift, chart breakouts.
• position   — Hold weeks to months. Best for: structural shifts, product cycles,
               regulatory tailwinds, undervaluation confirmed by news catalyst.

═══════════════════════════════════════════════════════
ANALYSIS FRAMEWORK (apply ALL every cycle)
═══════════════════════════════════════════════════════

1. NEWS CATALYST QUALITY
   Score each article:
   - Source tier: Financial Times / Reuters / Bloomberg (A) > wire services (B) > blogs (C)
   - Recency: <30 min = hot signal, <2h = warm, >3h = stale
   - Type: earnings beat/miss, M&A, FDA, macro data, product launch, legal/regulatory
   - Directness: company named directly (strong) vs sector peer mentioned (weak)
   - Magnitude: beat by 5% vs beat by 50% — size matters
   FT articles are high-credibility; always analyze them carefully.

2. MARKET SESSION CONTEXT
   - pre_market  (before 9:30): gaps forming, low liquidity — set up for open
   - opening     (9:30–10:30): highest volatility — momentum trades, gap-fills
   - mid_morning (10:30–11:30): trend establishing — confirm breakouts
   - lunch_lull  (11:30–14:00): low volume — avoid new positions unless exceptional catalyst
   - afternoon   (14:00–15:30): institutional activity resumes — trend following
   - power_hour  (15:30–16:00): high volume surge — trend amplification or reversal
   - after_hours (after 16:00): news digestion, pre-position for tomorrow

3. PORTFOLIO RISK ASSESSMENT
   - Total exposure vs available cash
   - Sector concentration (avoid >40% in one sector)
   - Losing positions: consider closing anything below -5% unrealized P&L
   - Winning positions: trim at +15%, full close at +25% unless strong ongoing catalyst
   - Correlation: don't open multiple highly-correlated positions (e.g. NVDA + AMD + SMCI)

4. POSITION SIZING — use available buying_power freely but intelligently
   - conviction 5 (extremely high): 25–35% of available buying_power
   - conviction 4 (high): 15–25%
   - conviction 3 (medium): 8–15%
   - conviction 2 (low): 3–8% or skip
   - conviction 1 (speculative): skip unless asymmetric upside
   - Never deploy >80% of buying_power in a single cycle (keep dry powder)
   - Multiple buys per cycle are allowed when each has a distinct independent catalyst

5. EXIT DISCIPLINE
   - Always include stop_loss_pct and take_profit_pct with every buy
   - Stop-loss: 2–5% for day trades, 5–10% for swing, 8–15% for positions
   - Take-profit: 5–10% for day trades, 10–25% for swing, 20–50% for positions
   - Use partial_close to lock in gains while keeping exposure on strong trends
   - Close losers decisively — do not hope; do not average down

6. DAY-TRADING SPECIFICS
   - Best catalysts: pre-market earnings, FDA decisions, M&A, macro beats
   - Enter early in the session on confirmed momentum
   - Use power_hour (15:30–16:00) to close remaining day trades
   - Do NOT hold day_trade positions overnight

═══════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════
- Only trade assets on Alpaca: US stocks, ETFs, crypto.
- Crypto symbols: BTC/USD, ETH/USD, SOL/USD format.
- No actions = valid and often correct. Do not force trades in quiet markets.
- No averaging down into losing positions.
- If a day_trade position exists and it is power_hour or later → close it.
- Prioritize closing bad positions before opening new ones.

═══════════════════════════════════════════════════════
OUTPUT — ONLY valid JSON, no markdown fences, no extra text
═══════════════════════════════════════════════════════
{
  "market_sentiment": "bullish | bearish | neutral | mixed",
  "session_phase": "pre_market | opening | mid_morning | lunch_lull | afternoon | power_hour | after_hours",
  "key_insights": [
    "<specific insight with source and ticker>",
    "..."
  ],
  "risk_assessment": "<one sentence: current portfolio risk level and main concern>",
  "actions": [
    {
      "symbol": "TICKER",
      "action": "buy | close | partial_close | hold",
      "trade_type": "day_trade | swing | position",
      "notional_usd": 2000,
      "conviction": 4,
      "stop_loss_pct": 3.5,
      "take_profit_pct": 9.0,
      "reasoning": "<specific: catalyst source, price context, why this size>"
    }
  ]
}

Field notes:
- "notional_usd": dollar amount to invest (buy only). Size based on conviction × buying_power.
- "partial_close": closes ~50% of position. No notional needed.
- "close": fully exits position. No notional needed.
- "hold": explicitly keep — logged but no order placed.
- "stop_loss_pct" / "take_profit_pct": targets logged; include even if approximate.
- "conviction": 1 (low) to 5 (very high). Only act on conviction 3+.
- Empty "actions" array is correct when there is nothing worth trading.
"""


def analyze(
    ai_client: anthropic.Anthropic,
    news: list[dict],
    portfolio: dict,
    prices: dict,
    phase: str,
) -> dict | None:
    news_blob      = json.dumps(news[:40], indent=2)
    portfolio_blob = json.dumps(portfolio, indent=2)
    prices_blob    = json.dumps(prices, indent=2) if prices else "{}"

    user_msg = f"""\
CURRENT DATE/TIME (ET): {datetime.now(ET).strftime('%Y-%m-%d %H:%M')}
MARKET SESSION PHASE: {phase}
AVAILABLE BUYING POWER: ${portfolio['buying_power']:,.2f}
PORTFOLIO VALUE: ${portfolio['portfolio_value']:,.2f}
OPEN POSITIONS: {len(portfolio['positions'])}

=== RECENT FINANCIAL NEWS (Alpaca + Financial Times) ===
{news_blob}

=== CURRENT MARKET PRICES (watchlist + news symbols + holdings) ===
{prices_blob}

=== CURRENT PORTFOLIO ===
{portfolio_blob}

Analyze all news sources carefully. Identify the strongest catalysts.
Apply the full analysis framework. Provide specific, actionable trading recommendations.
"""
    try:
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        result = json.loads(raw)
        log.info(f"Sentiment: {result.get('market_sentiment', '?').upper()} | Phase: {result.get('session_phase', phase)}")
        log.info(f"Risk: {result.get('risk_assessment', '')}")
        for insight in result.get("key_insights", []):
            log.info(f"  » {insight}")
        return result

    except json.JSONDecodeError as e:
        log.error(f"Claude returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.error(f"Claude API call failed: {e}")
        return None


# ── Trade execution ───────────────────────────────────────────────────────────

def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def execute(trading_client: TradingClient, actions: list[dict], portfolio: dict, dry_run: bool) -> None:
    if not actions:
        log.info("No trading actions this cycle.")
        return

    open_positions = {p["symbol"]: p for p in portfolio.get("positions", [])}
    buying_power   = portfolio.get("buying_power", 0.0)
    total_bp       = buying_power  # track spend across this cycle

    for action in actions:
        symbol   = action.get("symbol", "").upper().strip()
        act      = action.get("action", "hold").lower()
        notional = float(action.get("notional_usd") or 0)
        reason   = action.get("reasoning", "")
        ttype    = action.get("trade_type", "swing")
        conv     = action.get("conviction", 3)
        sl_pct   = action.get("stop_loss_pct")
        tp_pct   = action.get("take_profit_pct")

        if not symbol:
            continue

        # ── HOLD ──────────────────────────────────────────────────────────────
        if act == "hold":
            log.info(f"  HOLD {symbol} [{ttype}]: {reason}")
            continue

        # ── BUY ───────────────────────────────────────────────────────────────
        if act == "buy":
            if notional <= 0:
                log.warning(f"  SKIP buy {symbol}: notional_usd is 0")
                continue
            if notional > buying_power:
                log.warning(f"  SKIP buy {symbol}: need ${notional:.0f}, only ${buying_power:.0f} available")
                continue
            # Safety cap: no single trade exceeds 40% of the cycle's starting buying_power
            hard_cap = total_bp * 0.40
            if notional > hard_cap:
                log.info(f"  Capping {symbol} buy at 40% of buying_power: ${notional:.0f} → ${hard_cap:.0f}")
                notional = hard_cap

            tif = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY
            order_req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
            targets = ""
            if sl_pct:
                targets += f" | SL: -{sl_pct}%"
            if tp_pct:
                targets += f" | TP: +{tp_pct}%"

            if dry_run:
                log.info(f"  [DRY-RUN] BUY ${notional:.0f} of {symbol} [{ttype} | conv={conv}]{targets} | {reason}")
            else:
                try:
                    order = trading_client.submit_order(order_req)
                    log.info(f"  BUY ${notional:.0f} of {symbol} [{ttype} | conv={conv}]{targets} | order={order.id} | {reason}")
                    buying_power -= notional
                    open_positions[symbol] = {"symbol": symbol, "qty": 0}  # placeholder
                except Exception as e:
                    log.error(f"  BUY {symbol} FAILED: {e}")

        # ── CLOSE (full exit) ─────────────────────────────────────────────────
        elif act in ("close", "sell"):
            if symbol not in open_positions:
                log.warning(f"  SKIP close {symbol}: no open position")
                continue
            if dry_run:
                log.info(f"  [DRY-RUN] CLOSE {symbol} | {reason}")
            else:
                try:
                    trading_client.close_position(symbol)
                    log.info(f"  CLOSE {symbol} | {reason}")
                    open_positions.pop(symbol, None)
                except Exception as e:
                    log.error(f"  CLOSE {symbol} FAILED: {e}")

        # ── PARTIAL CLOSE (~50%) ──────────────────────────────────────────────
        elif act == "partial_close":
            if symbol not in open_positions:
                log.warning(f"  SKIP partial_close {symbol}: no open position")
                continue
            if dry_run:
                log.info(f"  [DRY-RUN] PARTIAL CLOSE {symbol} (~50%) | {reason}")
            else:
                try:
                    trading_client.close_position(symbol, ClosePositionRequest(percentage="0.5"))
                    log.info(f"  PARTIAL CLOSE {symbol} (~50%) | {reason}")
                except Exception as e:
                    log.error(f"  PARTIAL CLOSE {symbol} FAILED: {e}")

        else:
            log.warning(f"  Unknown action '{act}' for {symbol} — skipping")


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run: bool) -> None:
    if not in_trading_window():
        log.info("Outside trading window — waiting.")
        return

    phase = session_phase()
    log.info("-" * 60)
    log.info(f"Cycle start: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} | Session: {phase.upper()}")
    log.info("-" * 60)

    lookback = max(3, CHECK_INTERVAL // 20 + 1)
    news      = fetch_all_news(news_client, lookback_hours=lookback)
    portfolio = get_portfolio(trading_client)

    news_syms   = [s for a in news for s in a.get("symbols", [])]
    held_syms   = [p["symbol"] for p in portfolio["positions"]]
    all_symbols = list(set(news_syms + WATCHLIST + held_syms))
    prices      = fetch_prices(stock_data_client, all_symbols)

    log.info(
        f"Portfolio: ${portfolio['portfolio_value']:,.2f} | "
        f"Buying power: ${portfolio['buying_power']:,.2f} | "
        f"Open positions: {len(portfolio['positions'])}"
    )

    analysis = analyze(ai_client, news, portfolio, prices, phase)
    if analysis is None:
        log.error("Analysis failed — skipping execution this cycle.")
        return

    execute(trading_client, analysis.get("actions", []), portfolio, dry_run)
    log.info("Cycle complete.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot (Alpaca + Claude)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and log decisions but do NOT place any real orders",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle immediately and exit (useful for testing)",
    )
    args = parser.parse_args()

    mode_tag = "DRY-RUN " if args.dry_run else ""
    mode_tag += "PAPER" if PAPER_TRADING else "LIVE"
    log.info(f"Starting AI Trading Bot [{mode_tag}] — interval: {CHECK_INTERVAL} min")

    if not PAPER_TRADING and not args.dry_run:
        log.warning("⚠  LIVE TRADING IS ENABLED — real money will be used!")

    trading_client, news_client, stock_data_client, ai_client = build_clients()

    def cycle():
        run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run=args.dry_run)

    if args.once:
        log.info("Running single cycle (--once flag ignores market-hours guard).")
        run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run=True)
        return

    cycle()
    schedule.every(CHECK_INTERVAL).minutes.do(cycle)

    log.info(f"Scheduler running — next cycle in {CHECK_INTERVAL} min. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")


if __name__ == "__main__":
    main()
