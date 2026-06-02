#!/usr/bin/env python3
"""
AI Trading Bot — Alpaca + Claude
Analyzes financial news every N minutes during US market hours
(8:30 AM – 5:00 PM ET, Mon–Fri) and executes trades via Alpaca.
"""

import os
import sys
import json
import time
import argparse
import logging
import schedule
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
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
MAX_POSITION_USD  = float(os.getenv("MAX_POSITION_SIZE_USD", "500"))
MAX_POSITIONS     = int(os.getenv("MAX_TOTAL_POSITIONS", "10"))

_watchlist_raw = os.getenv("WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,SPY,QQQ")
WATCHLIST = [s.strip().upper() for s in _watchlist_raw.split(",") if s.strip()]

ET = pytz.timezone("America/New_York")

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
    """Mon–Fri, 9:00 AM – 4:30 PM ET (30 min before open, 30 min after close)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    start = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    end   = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return start <= now <= end


# ── News ──────────────────────────────────────────────────────────────────────

def fetch_news(news_client: NewsClient, lookback_hours: int = 3) -> list[dict]:
    since = datetime.now(pytz.utc) - timedelta(hours=lookback_hours)
    try:
        response = news_client.get_news(
            NewsRequest(start=since, limit=50, include_content=False)
        )
        raw = response.news if hasattr(response, "news") else list(response)
        # Newer alpaca-py returns (key, news_obj) tuples — unwrap them
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
                "headline":  _field(a, "headline"),
                "summary":   _field(a, "summary"),
                "source":    _field(a, "source"),
                "symbols":   _field(a, "symbols") or [],
                "published": str(_field(a, "created_at")),
            })
        log.info(f"Fetched {len(articles)} news articles (last {lookback_hours}h)")
        return articles
    except Exception as e:
        log.warning(f"News fetch failed: {e} — continuing without news")
        return []


# ── Prices ────────────────────────────────────────────────────────────────────

def fetch_prices(stock_data_client: StockHistoricalDataClient, symbols: list[str]) -> dict:
    """Return latest ask/bid price for each stock symbol (skips crypto)."""
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
You are an experienced quantitative analyst and portfolio manager.
Your job: study recent financial news, current market prices, and the current
portfolio, then decide whether to open new positions, close existing ones, or do nothing.

STRICT RULES
- Only trade assets available on Alpaca (US stocks, ETFs, crypto).
- Crypto symbols must use Alpaca format: BTC/USD, ETH/USD, SOL/USD, etc.
- Keep each new buy within the configured MAX_POSITION_USD limit.
- Never exceed MAX_TOTAL_POSITIONS open positions.
- Avoid over-concentration: do not pile into the same sector.
- Only act on high-conviction signals — when in doubt, do nothing.
- Do not average down into losing positions.
- Maximum 3 actions per cycle.

OUTPUT — reply with ONLY valid JSON, no markdown fences, no extra text:
{
  "market_sentiment": "bullish | bearish | neutral",
  "key_insights": ["<insight>", "..."],
  "actions": [
    {
      "symbol": "TICKER",
      "action": "buy | close | hold",
      "notional_usd": 500,
      "reasoning": "<one sentence>"
    }
  ]
}

Notes:
- "notional_usd" is the dollar amount to spend (only required for "buy").
- "close" fully exits an existing position (no notional needed).
- "hold" means explicitly keeping a position — no order is placed.
- If nothing should be done, return an empty "actions" array.
"""

def analyze(ai_client: anthropic.Anthropic, news: list[dict], portfolio: dict, prices: dict) -> dict | None:
    news_blob      = json.dumps(news[:30], indent=2)
    portfolio_blob = json.dumps(portfolio, indent=2)
    prices_blob    = json.dumps(prices, indent=2) if prices else "{}"

    user_msg = f"""\
CURRENT DATE/TIME (ET): {datetime.now(ET).strftime('%Y-%m-%d %H:%M')}
MAX_POSITION_USD:  ${MAX_POSITION_USD:,.0f}
MAX_TOTAL_POSITIONS: {MAX_POSITIONS}

=== RECENT NEWS ===
{news_blob}

=== CURRENT MARKET PRICES (watchlist + news symbols) ===
{prices_blob}

=== CURRENT PORTFOLIO ===
{portfolio_blob}

Analyze and provide trading recommendations.
"""
    try:
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown code fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        result = json.loads(raw)
        log.info(f"Sentiment: {result.get('market_sentiment', '?').upper()}")
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

    open_symbols  = {p["symbol"] for p in portfolio.get("positions", [])}
    buying_power  = portfolio.get("buying_power", 0.0)
    position_count = len(open_symbols)

    for action in actions:
        symbol   = action.get("symbol", "").upper().strip()
        act      = action.get("action", "hold").lower()
        notional = float(action.get("notional_usd") or 0)
        reason   = action.get("reasoning", "")

        if not symbol:
            continue

        if act == "hold":
            log.info(f"  HOLD {symbol}: {reason}")
            continue

        if act == "buy":
            # Guard rails
            if notional <= 0:
                log.warning(f"  SKIP buy {symbol}: notional_usd is 0")
                continue
            if notional > MAX_POSITION_USD:
                log.warning(f"  Capping {symbol} buy: ${notional:.0f} → ${MAX_POSITION_USD:.0f}")
                notional = MAX_POSITION_USD
            if notional > buying_power:
                log.warning(f"  SKIP buy {symbol}: need ${notional:.0f}, only ${buying_power:.0f} available")
                continue
            if position_count >= MAX_POSITIONS and symbol not in open_symbols:
                log.warning(f"  SKIP buy {symbol}: at max positions ({MAX_POSITIONS})")
                continue

            tif = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY
            order_req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
            if dry_run:
                log.info(f"  [DRY-RUN] BUY ${notional:.0f} of {symbol} | {reason}")
            else:
                try:
                    order = trading_client.submit_order(order_req)
                    log.info(f"  BUY ${notional:.0f} of {symbol} | order={order.id} | {reason}")
                    buying_power   -= notional
                    position_count += 1
                    open_symbols.add(symbol)
                except Exception as e:
                    log.error(f"  BUY {symbol} FAILED: {e}")

        elif act in ("close", "sell"):
            if symbol not in open_symbols:
                log.warning(f"  SKIP close {symbol}: no open position")
                continue
            if dry_run:
                log.info(f"  [DRY-RUN] CLOSE {symbol} | {reason}")
            else:
                try:
                    trading_client.close_position(symbol)
                    log.info(f"  CLOSE {symbol} | {reason}")
                    open_symbols.discard(symbol)
                    position_count -= 1
                except Exception as e:
                    log.error(f"  CLOSE {symbol} FAILED: {e}")

        else:
            log.warning(f"  Unknown action '{act}' for {symbol} — skipping")


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run: bool) -> None:
    if not in_trading_window():
        log.info("Outside trading window — waiting.")
        return

    log.info("-" * 60)
    log.info(f"Cycle start: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    log.info("-" * 60)

    news      = fetch_news(news_client, lookback_hours=max(3, CHECK_INTERVAL // 20 + 1))
    portfolio = get_portfolio(trading_client)

    # Gather all symbols: news mentions + watchlist + current holdings
    news_syms     = [s for a in news for s in a.get("symbols", [])]
    held_syms     = [p["symbol"] for p in portfolio["positions"]]
    all_symbols   = list(set(news_syms + WATCHLIST + held_syms))
    prices        = fetch_prices(stock_data_client, all_symbols)

    log.info(
        f"Portfolio: ${portfolio['portfolio_value']:,.2f} | "
        f"Buying power: ${portfolio['buying_power']:,.2f} | "
        f"Open positions: {len(portfolio['positions'])}"
    )

    analysis = analyze(ai_client, news, portfolio, prices)
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

    # Run immediately on start, then repeat on schedule
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
