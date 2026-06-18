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
import pytz
import feedparser
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, ClosePositionRequest, TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import NewsRequest, StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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
MAX_POSITIONS     = int(os.getenv("MAX_TOTAL_POSITIONS", "20"))

# Fixed daily run times in ET. Last run is 15:50 (before the 16:00 close) so the
# end-of-day close/hold decision can actually fill while the market is still open.
RUN_TIMES_ET      = [(8, 30), (10, 30), (12, 30), (14, 30), (15, 50)]
NEWS_LOOKBACK_HOURS = 3  # look back slightly more than the 2-hour cycle

_watchlist_raw = os.getenv(
    "WATCHLIST",
    "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,SPY,QQQ,AMD,NFLX,JPM,GS,BAC,"
    "TQQQ,UPRO,SOXL,TNA,UDOW,FAS,QLD,SSO,IWM,PLTR,MSTR,COIN,MARA,SMCI,RIVN,SOFI"
)
WATCHLIST = [s.strip().upper() for s in _watchlist_raw.split(",") if s.strip()]

LEVERAGE_ENABLED = os.getenv("LEVERAGE_ENABLED", "true").lower() == "true"
MAX_LEVERAGE     = float(os.getenv("MAX_LEVERAGE", "5.0"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
HINTS_FILE         = "hints.txt"
STATE_FILE         = "daily_state.json"   # remembers the day's actions across runs

ANALYSIS_MODEL     = "claude-opus-4-8"    # strongest reasoning model for the trade decision

# Base ETF → 2x → 3x leveraged equivalents
LEVERAGE_MAP: dict[str, dict] = {
    "SPY":  {"2x": "SSO",  "3x": "UPRO"},
    "QQQ":  {"2x": "QLD",  "3x": "TQQQ"},
    "IWM":  {"2x": "UWM",  "3x": "TNA"},
    "DIA":  {"2x": "DDM",  "3x": "UDOW"},
    "SOXX": {"2x": "USD",  "3x": "SOXL"},
    "XLF":  {"2x": "UYG",  "3x": "FAS"},
    "XLE":  {"2x": "DIG",  "3x": "ERX"},
    "XBI":  {"2x": None,   "3x": "LABU"},
    "GLD":  {"2x": "UGL",  "3x": None},
    "TLT":  {"2x": "UBT",  "3x": None},
}

ET = pytz.timezone("America/New_York")


def next_run_time() -> datetime:
    """Return the next fixed run slot (ET), skipping weekends."""
    now = datetime.now(ET)
    today = now.date()
    if today.weekday() < 5:  # today is a weekday — check remaining slots
        for hour, minute in RUN_TIMES_ET:
            candidate = ET.localize(datetime(today.year, today.month, today.day, hour, minute))
            if candidate > now:
                return candidate
    # No slots left today (or today is weekend) — advance to next weekday
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    hour, minute = RUN_TIMES_ET[0]
    return ET.localize(datetime(next_day.year, next_day.month, next_day.day, hour, minute))


def run_slot_position() -> tuple[bool, bool]:
    """Return (is_first_slot, is_last_slot) for the run we are executing, by ET time."""
    now = datetime.now(ET)
    minutes_now = now.hour * 60 + now.minute
    slot_minutes = [h * 60 + m for h, m in RUN_TIMES_ET]
    # The slot being executed is the latest one at/before now (15-min tolerance for jitter).
    past = [i for i, sm in enumerate(slot_minutes) if minutes_now >= sm - 15]
    idx = past[-1] if past else 0
    return idx == 0, idx == len(slot_minutes) - 1


# ── Daily state (memory across runs) ────────────────────────────────────────────

def load_state() -> dict:
    """Load today's running state, or a fresh state if it's a new day / missing."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("date") == today:
            return state
    except Exception:
        pass
    return {"date": today, "cycles": [], "start_portfolio_value": None}


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning(f"Could not save {STATE_FILE}: {e}")


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


# ── Asset validation ──────────────────────────────────────────────────────────

_asset_cache: dict[str, bool] = {}

def is_asset_tradeable(trading_client: TradingClient, symbol: str) -> bool:
    if "/" in symbol:
        return True  # crypto — validated separately
    if symbol in _asset_cache:
        return _asset_cache[symbol]
    try:
        asset = trading_client.get_asset(symbol)
        ok = bool(getattr(asset, "tradable", False)) and "active" in str(getattr(asset, "status", "")).lower()
        _asset_cache[symbol] = ok
        return ok
    except Exception:
        _asset_cache[symbol] = False
        return False


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
            summary = _field(a, "summary")
            articles.append({
                "source":    "Alpaca/" + (_field(a, "source") or "news"),
                "headline":  _field(a, "headline"),
                "summary":   summary[:240] if summary else "",
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
                    "summary":   summary[:240] if summary else "",
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

def fetch_market_data(stock_data_client: StockHistoricalDataClient, symbols: list[str]) -> dict:
    """Per symbol: live price + context (prev close, day OHLC, % moves, relative volume).

    Gives Claude the price action its strategy actually needs (gaps, intraday
    direction, volume), instead of a single bare quote.
    """
    stock_syms = [s for s in symbols if "/" not in s and s]
    if not stock_syms:
        return {}

    data: dict[str, dict] = {}

    # 1) Daily bars → previous close, today's range, average + relative volume.
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=45)  # ~30 trading days
        bars = stock_data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=stock_syms, timeframe=TimeFrame.Day, start=start)
        )
        bar_map  = bars.data if hasattr(bars, "data") else {}
        today_et = datetime.now(ET).date()
        for sym, sym_bars in bar_map.items():
            if not sym_bars:
                continue
            last = sym_bars[-1]
            try:
                last_date = last.timestamp.astimezone(ET).date()
            except Exception:
                last_date = None
            vols = [b.volume for b in sym_bars]
            if last_date == today_et and len(sym_bars) >= 2:
                today_bar  = last
                prev_close = sym_bars[-2].close
                hist_vols  = vols[:-1]
            else:
                today_bar  = None
                prev_close = last.close
                hist_vols  = vols
            recent_vols = hist_vols[-20:]
            avg_vol = (sum(recent_vols) / len(recent_vols)) if recent_vols else 0
            entry = {"prev_close": round(prev_close, 2)}
            if today_bar is not None:
                entry["open"]     = round(today_bar.open, 2)
                entry["day_high"] = round(today_bar.high, 2)
                entry["day_low"]  = round(today_bar.low, 2)
                if avg_vol:
                    entry["rel_volume"] = round(today_bar.volume / avg_vol, 2)
            data[sym] = entry
    except Exception as e:
        log.warning(f"Daily bars fetch failed: {e} — continuing with quotes only")

    # 2) Latest quote → live price (works pre/post market).
    try:
        quotes = stock_data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=stock_syms)
        )
        for sym, q in quotes.items():
            price = getattr(q, "ask_price", None) or getattr(q, "bid_price", None)
            if price and float(price) > 0:
                data.setdefault(sym, {})["price"] = round(float(price), 2)
    except Exception as e:
        log.warning(f"Quote fetch failed: {e}")

    # 3) Derive % moves; fill missing price from bar data so every symbol has one.
    out: dict[str, dict] = {}
    for sym, d in data.items():
        price = d.get("price") or d.get("open") or d.get("prev_close")
        if not price:
            continue
        d["price"] = price
        if d.get("prev_close"):
            d["pct_from_prev_close"] = round((price - d["prev_close"]) / d["prev_close"] * 100, 2)
        if d.get("open"):
            d["pct_from_open"] = round((price - d["open"]) / d["open"] * 100, 2)
        out[sym] = d

    log.info(f"Market data for {len(out)}/{len(stock_syms)} symbols")
    return out


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


# ── User hints ────────────────────────────────────────────────────────────────

_telegram_offset = 0  # tracks last processed Telegram update


def _hints_file_init() -> None:
    """Create hints.txt with comment header if it doesn't exist."""
    if not os.path.exists(HINTS_FILE):
        with open(HINTS_FILE, "w", encoding="utf-8") as f:
            f.write(
                "# Trading hints — edited by hand or via Telegram.\n"
                "# Lines starting with # are ignored.\n"
            )


def load_user_hints() -> str:
    """Return all active (non-comment) lines from hints.txt."""
    _hints_file_init()
    try:
        with open(HINTS_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if lines:
            log.info(f"User hints loaded: {len(lines)} line(s)")
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"Could not read {HINTS_FILE}: {e}")
        return ""


def _append_hints(hints: list[str]) -> None:
    _hints_file_init()
    with open(HINTS_FILE, "a", encoding="utf-8") as f:
        for h in hints:
            f.write(h + "\n")
    log.info(f"Saved {len(hints)} hint(s) from Telegram.")


def _clear_hints() -> None:
    if not os.path.exists(HINTS_FILE):
        return
    with open(HINTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(HINTS_FILE, "w", encoding="utf-8") as f:
        f.writelines(l for l in lines if not l.strip() or l.strip().startswith("#"))
    log.info("Hints cleared via Telegram /clear command.")


def poll_telegram_hints() -> None:
    """Fetch new Telegram messages and handle them as hints or commands."""
    global _telegram_offset
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.request, ssl
        _ssl_ctx = ssl.create_default_context()
        _ssl_ctx.check_hostname = False
        _ssl_ctx.verify_mode = ssl.CERT_NONE
        url = (
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
            f"/getUpdates?offset={_telegram_offset}&timeout=0"
        )
        with urllib.request.urlopen(url, timeout=10, context=_ssl_ctx) as resp:
            data = json.loads(resp.read())

        if not data.get("ok"):
            return

        new_hints: list[str] = []
        for update in data.get("result", []):
            _telegram_offset = update["update_id"] + 1
            msg     = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if chat_id != str(TELEGRAM_CHAT_ID):
                continue
            text = msg.get("text", "").strip()
            if not text:
                continue

            if text.lower() in ("/clear", "clear"):
                _clear_hints()
                send_telegram("All hints cleared.")

            elif text.lower() in ("/hints", "hints"):
                active = load_user_hints()
                if active:
                    send_telegram(f"*Active hints:*\n" + "\n".join(f"• {l}" for l in active.splitlines()))
                else:
                    send_telegram("No active hints.")

            elif text.startswith("/"):
                send_telegram("Commands: /hints — list active hints | /clear — remove all hints\nOr just send any message to add a hint.")

            else:
                new_hints.append(text)

        if new_hints:
            _append_hints(new_hints)
            bullet_list = "\n".join(f"• {h}" for h in new_hints)
            send_telegram(f"Got it! Hint(s) saved — I'll use them in the next cycle:\n{bullet_list}")

    except Exception as e:
        log.warning(f"Telegram poll failed: {e}")


# ── Telegram notifications ────────────────────────────────────────────────────

def send_telegram(message: str) -> None:
    """Send a message to the configured Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.request, urllib.parse, ssl
        _ssl_ctx = ssl.create_default_context()
        _ssl_ctx.check_hostname = False
        _ssl_ctx.verify_mode = ssl.CERT_NONE
        payload = json.dumps({
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10, context=_ssl_ctx)
        log.info("Telegram notification sent.")
    except Exception as e:
        log.warning(f"Telegram notification failed: {e}")


def build_day_summary(ai_client: anthropic.Anthropic, state: dict, portfolio: dict) -> str:
    """End-of-day Telegram message: the day's actions + a short reasoning summary."""
    cycles      = state.get("cycles", [])
    all_actions = [a for c in cycles for a in c.get("actions", [])]
    start_val   = state.get("start_portfolio_value")
    end_val     = portfolio.get("portfolio_value")

    day_change = ""
    if start_val and end_val:
        diff = end_val - start_val
        pct  = (diff / start_val * 100) if start_val else 0
        day_change = f"${diff:+,.2f} ({pct:+.2f}%)"

    open_pos  = portfolio.get("positions", [])
    total_upl = sum(p.get("unrealized_pl", 0) for p in open_pos)

    # Prefer a concise Claude-written narrative; fall back to a plain format on any error.
    try:
        payload = {
            "date":            state.get("date"),
            "actions_today":   all_actions,
            "portfolio_value": end_val,
            "day_change":      day_change or "n/a",
            "open_positions":  [
                {"symbol": p["symbol"], "qty": p["qty"], "unrealized_plpc": p.get("unrealized_plpc")}
                for p in open_pos
            ],
        }
        resp = ai_client.messages.create(
            model=ANALYSIS_MODEL,
            max_tokens=600,
            system=(
                "You write a concise end-of-day summary of an automated trading bot's day "
                "for a Telegram message. Plain text, no markdown headers, under 120 words. "
                "Cover briefly: what was traded and why, how the portfolio did, and what is "
                "held overnight into tomorrow. Output only the summary."
            ),
            messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
        )
        text = "\n".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        if text:
            header = f"*Daily Summary — {state.get('date')}*"
            if day_change:
                header += f"\nP&L today: {day_change}"
            return f"{header}\n\n{text}"
    except Exception as e:
        log.warning(f"Day summary generation failed: {e} — using plain format")

    return _format_day_summary_fallback(state, all_actions, day_change, total_upl)


def _format_day_summary_fallback(state: dict, all_actions: list, day_change: str, total_upl: float) -> str:
    lines = [f"*Daily Summary — {state.get('date')}*"]
    if day_change:
        lines.append(f"P&L today: {day_change}")
    lines.append("")
    if all_actions:
        lines.append(f"*Actions taken ({len(all_actions)}):*")
        for a in all_actions:
            act = (a.get("action") or "").upper()
            sym = a.get("symbol", "?")
            r   = (a.get("reasoning") or "")[:90]
            lines.append(f"• {act} {sym} — {r}")
    else:
        lines.append("_No trades taken today._")
    lines.append(f"\nOpen unrealized P&L: ${total_upl:,.2f}")
    return "\n".join(lines)


# ── Claude analysis ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite intraday news-driven momentum trader. Your mandate: capture explosive
intraday moves triggered by fundamental catalysts, using leveraged instruments when
conviction is highest and precise risk management at all times.

═══════════════════════════════════════════════════════
PRIMARY STRATEGY: GAP-AND-GO (News Momentum)
═══════════════════════════════════════════════════════
Most reliable setup for news-driven day trading:

1. IDENTIFY CATALYST: Pre-market or intraday news with clear directional impact.
2. CONFIRM MOMENTUM: Price gapping >2% on the catalyst direction.
3. ENTRY TIMING:
   - pre_market: pre-position on A-tier catalyst with very high confidence.
   - opening (9:30–10:30): enter on first 5-min candle breakout above pre-market high.
   - intraday: enter only when price breaks a clear level with volume surge.
4. SCALE OUT: Partial close at first target, trail remainder.
5. AUTO-EXIT: stop_loss_pct and take_profit_pct are placed as live bracket orders
   at entry, so they trigger automatically between runs — set them deliberately.

Secondary setups (use when Gap-and-Go not available):
• MOMENTUM CONTINUATION — ride established intraday trend, enter on pullbacks to VWAP.
• SECTOR SYMPATHY — when sector leader has A-tier catalyst, buy high-beta peers.
• REVERSAL FADE — short-term mean-reversion after extreme over-extension (>15% spike on weak catalyst).

═══════════════════════════════════════════════════════
CATALYST TIERS (critical for sizing and leverage decisions)
═══════════════════════════════════════════════════════
A-TIER — explosive, binary, surprise events:
  Earnings beat/miss >10% | FDA approval/rejection | Major M&A (>20% premium)
  Macro shock: CPI/NFP/Fed surprise | Short squeeze trigger | Chapter 11 filing
  → Max size, leverage OK, enter aggressively

B-TIER — strong but expected or partial:
  Earnings beat 5–10% | Product launch | Analyst upgrade >20% price target
  Sector rotation data | Regulatory win | CEO change at major company
  → Normal sizing, 2x leverage OK on high conviction

C-TIER — weak or indirect:
  Earnings beat <5% | Minor product news | Sector peer catalyst | Macro drift
  → Reduced size, NO leverage

═══════════════════════════════════════════════════════
LEVERAGE FRAMEWORK — use only when conditions align
═══════════════════════════════════════════════════════
You have access to regular instruments AND leveraged ETFs. Prefer leveraged ETFs
over margin on regular stocks — they give clean, defined leverage.

Leveraged ETF pairs — LONG (pick the right instrument directly in "symbol"):
  SPY  → SSO (2x)  → UPRO (3x)      QQQ  → QLD (2x)  → TQQQ (3x)
  IWM  → UWM (2x)  → TNA (3x)       DIA  → DDM (2x)  → UDOW (3x)
  SOXX → USD (2x)  → SOXL (3x)      XLF  → UYG (2x)  → FAS (3x)
  XLE  → DIG (2x)  → ERX (3x)       XBI  → — (2x)    → LABU (3x)
  GLD  → UGL (2x)  → — (3x)         TLT  → UBT (2x)  → — (3x)

Inverse leveraged ETF pairs — SHORT exposure (use "buy" action on these tickers):
  SPY  → SDS (2x)  → SPXU (3x)      QQQ  → QID (2x)  → SQQQ (3x)
  IWM  → TWM (2x)  → SRTY (3x)      DIA  → DXD (2x)  → SDOW (3x)
  SOXX → SSG (2x)  → SOXS (3x)      XLF  → SKF (2x)  → FAZ (3x)
  XLE  → DUG (2x)  → ERY (3x)       XBI  → BIS (2x)  → LABD (3x)
  GLD  → GLL (2x)  → — (3x)         TLT  → TBT (2x)  → TMV (3x)

For individual stocks (no leveraged ETF equivalent): set "leverage": 2–5
  → Bot multiplies notional by that factor using available margin.

WHEN TO USE LEVERAGE:
  ✓ Catalyst is A-tier AND session is opening or power_hour
  ✓ Conviction = 5, direction confirmed by multiple signals
  ✓ Clear stop level within 2–3% of entry (tight risk)
  ✓ Portfolio has no major losing positions currently

WHEN NOT TO USE LEVERAGE:
  ✗ lunch_lull, pre_market (first entry), after_hours
  ✗ C-tier catalyst or conviction ≤ 3
  ✗ Existing positions down >5% unrealized
  ✗ High macro uncertainty (FOMC day, CPI day before release)

Maximum effective leverage: 5x. Do not exceed.

═══════════════════════════════════════════════════════
SHORT SELLING — bet on price declines
═══════════════════════════════════════════════════════
Two ways to go short:

1. INVERSE ETFs (preferred — no margin, defined risk):
   Use "action": "buy" on an inverse ETF ticker (SQQQ, SPXU, SOXS, FAZ…).
   Treat exactly like a regular buy. Leverage is embedded in the ticker.

2. DIRECT SHORT (individual stocks only):
   Use "action": "short" to borrow and sell shares.
   Use "action": "cover" to close the full short position.
   Use "action": "cover_partial" to close ~50% of the short position.
   Leverage field applies the same way as with longs.

WHEN TO GO SHORT (direct or inverse ETF):

  Hard catalysts (A-tier — act fast):
  ✓ Earnings miss >10%, revenue miss, guidance cut
  ✓ FDA rejection, clinical trial failure
  ✓ Chapter 11 / bankruptcy / delisting notice
  ✓ Major fraud, accounting scandal, DOJ/SEC investigation announced
  ✓ Index breakdown: market-wide selloff from macro shock (CPI beat, hawkish Fed surprise)
  ✓ Reversal fade: stock spiked >15% on a weak or fraudulent catalyst — fade the hype
  ✓ Sector contagion: sector leader collapses, short high-beta peers

  Valuation / overpricing signals (B-tier — use swing trade_type):
  ✓ High-profile investor publicly says stock is overvalued or calls it a bubble
    (e.g. Buffett, Burry, Ackman, Einhorn, prominent short-sellers like Hindenburg/Citron)
  ✓ Analyst downgrades to Sell/Underperform WITH a price target well below current price (>15% downside)
  ✓ Multiple analysts simultaneously cut price targets after a weak earnings or outlook
  ✓ Credible research report flags overvaluation, inflated revenue, or accounting irregularities
  ✓ Stock trading at extreme multiple (P/E >100x) AND news sentiment turning negative
  ✓ ETF or index sector widely described as "overbought", "bubble", or "frothy" by credible sources
  ✓ Fed/treasury officials warn about specific sector valuations (e.g. AI bubble, housing bubble)

WHEN NOT TO SHORT:
  ✗ Strong uptrend without a clear catalyst reversal or credible overvaluation claim
  ✗ Heavily shorted stock (short interest >20%) — risk of short squeeze
  ✗ Earnings approaching for target (binary event risk)
  ✗ Conviction ≤ 3 or catalyst is C-tier
  ✗ Valuation concern alone with no supporting news or credible source — not enough

SHORT SIZING: same rules as longs. Stop-loss is price RISING above entry.
  Day short:  SL 1.5–3% above entry | TP 6–12% below entry
  Swing short: SL 4–8% above entry  | TP 12–25% below entry (for valuation-based shorts)

═══════════════════════════════════════════════════════
DYNAMIC STOCK UNIVERSE — ANY stock or ETF on Alpaca
═══════════════════════════════════════════════════════
You are NOT limited to a fixed watchlist. Trade ANY US stock or ETF mentioned
in the news if it has a strong catalyst and meets entry criteria.
Prioritize: high relative volume, liquid bid/ask, direct fundamental catalyst.
Use full leveraged ETF tickers directly (TQQQ, SOXL, UPRO, TNA, FAS, etc.)
when you want leveraged index/sector exposure.

═══════════════════════════════════════════════════════
TRADING HORIZONS
═══════════════════════════════════════════════════════
• day_trade  — Typically open AND close same session. PRIMARY focus. Best ROI/risk ratio.
               Requires: A or B-tier catalyst, opening/power_hour phase, or confirmed momentum.
               At the FINAL run of the day you decide per position: close it to avoid
               overnight risk, or hold overnight when the thesis and risk clearly justify it.
• swing      — Hold 1–5 days. For B-tier multi-day thesis or post-earnings drift.
• position   — Weeks to months. Only for structural A-tier thesis with strong conviction.

═══════════════════════════════════════════════════════
SESSION STRATEGY
═══════════════════════════════════════════════════════
pre_market  (before 9:30): Scan catalysts. Pre-position ONLY on extreme A-tier news.
                            Size small (50% of intended), full size at open confirmation.
opening     (9:30–10:30):  PRIME WINDOW. Gap-and-go execution. Enter fast on confirmed
                            breakouts. Highest volatility = highest opportunity.
mid_morning (10:30–11:30): Trend confirmation. Enter only on strong, established momentum.
                            Reduce leverage. Take partial profits on winners.
lunch_lull  (11:30–14:00): LOW ACTIVITY. No new leveraged positions. Take profits.
                            Tighten stops on existing. Close marginal trades.
afternoon   (14:00–15:30): Institutional re-entry. Trend following OK. Watch for reversals.
power_hour  (15:30–16:00): Decide exits. Use the volume surge to exit anything you don't
                            want to carry overnight at the best price.
after_hours (after 16:00): FINAL run. Review EVERY open position and decide explicitly —
                            close it, or hold it overnight if the thesis and risk justify
                            carrying it. No new day-trade entries this late.

═══════════════════════════════════════════════════════
POSITION SIZING
═══════════════════════════════════════════════════════
Base notional (BEFORE leverage) as % of buying_power:
  conviction 5 + A-tier: 20–30%   conviction 5 + B-tier: 12–20%
  conviction 4 + A-tier: 12–18%   conviction 4 + B-tier:  8–12%
  conviction 3 (any tier): 4–8%   conviction ≤ 2: skip
  Keep ≥20% buying_power as dry powder always.
  Multiple buys per cycle OK when each has a distinct independent catalyst.

═══════════════════════════════════════════════════════
EXIT DISCIPLINE
═══════════════════════════════════════════════════════
Always set stop_loss_pct and take_profit_pct. They are placed as LIVE bracket orders
the moment a position opens, so they auto-exit between runs without waiting for the next cycle.
  Day trade: SL 1.5–3% | TP 6–12% first target, trail remainder
  Swing:     SL 4–8%   | TP 12–25%
  Position:  SL 8–15%  | TP 25–60%

Rules:
  - Move stop to breakeven when position reaches +5% profit.
  - Use partial_close at first TP to lock gains, keep exposure on strong trend.
  - Close losers below -5% unrealized — no averaging down, no hoping.
  - Prioritize closing bad positions BEFORE opening new ones.

═══════════════════════════════════════════════════════
PORTFOLIO RISK
═══════════════════════════════════════════════════════
  - Max 35% in any single sector (leveraged ETFs count toward their sector).
  - Avoid high-correlated pairs simultaneously (e.g. TQQQ + QQQ, SOXL + NVDA + AMD).
  - If total unrealized P&L is below -8%, go to capital preservation mode: close losers, no new leveraged buys.

═══════════════════════════════════════════════════════
OUTPUT — ONLY valid JSON, no markdown fences, no extra text
═══════════════════════════════════════════════════════
{
  "market_sentiment": "bullish | bearish | neutral | mixed",
  "session_phase": "pre_market | opening | mid_morning | lunch_lull | afternoon | power_hour | after_hours",
  "catalyst_summary": "<top 2-3 actionable catalysts with tickers, tier, and expected direction>",
  "key_insights": [
    "<specific insight: source, ticker, catalyst tier, expected move direction>",
    "..."
  ],
  "risk_assessment": "<portfolio risk level, concentration, leverage in use, main concern>",
  "actions": [
    {
      "symbol": "TICKER",
      "action": "buy | short | close | cover | partial_close | cover_partial | hold",
      "trade_type": "day_trade | swing | position",
      "notional_usd": 2000,
      "leverage": 1,
      "catalyst_tier": "A | B | C",
      "conviction": 4,
      "stop_loss_pct": 2.5,
      "take_profit_pct": 9.0,
      "reasoning": "<catalyst source, tier, entry rationale, size justification, leverage rationale if >1>"
    }
  ]
}

Field notes:
- "symbol": ANY valid US stock or ETF ticker. Use leveraged ETF tickers directly
  (TQQQ, SOXL, UPRO, TNA, FAS…) for ETF leverage. No separate "leverage" field needed
  in that case — set leverage=1 and just pick the leveraged ticker.
- "leverage": 1–5 integer. Applied as margin multiplier for non-leveraged stocks.
  Only set >1 for individual stocks that have no ETF leveraged equivalent.
  Default: 1.
- "notional_usd": base dollar amount BEFORE leverage multiplier.
- "catalyst_tier": A / B / C — drives sizing and leverage eligibility validation.
- "partial_close": closes ~50% of a long position. No notional needed.
- "close": fully exits a long position. No notional needed.
- "cover": fully closes a short position. No notional needed.
- "cover_partial": closes ~50% of a short position. No notional needed.
- "hold": logged, no order.
- Empty "actions" array is correct and preferred when no setup meets criteria.
- Crypto symbols: BTC/USD, ETH/USD, SOL/USD format. No leverage on crypto.
- Do NOT short crypto.
"""


def analyze(
    ai_client: anthropic.Anthropic,
    news: list[dict],
    portfolio: dict,
    market: dict,
    phase: str,
    user_hints: str = "",
    day_history: list[dict] | None = None,
    is_last_run: bool = False,
) -> dict | None:
    # Compact JSON — no indentation — to cut input tokens with zero loss of information.
    news_blob      = json.dumps(news[:40], separators=(",", ":"))
    portfolio_blob = json.dumps(portfolio, separators=(",", ":"))
    market_blob    = json.dumps(market, separators=(",", ":")) if market else "{}"

    hints_section = ""
    if user_hints:
        hints_section = f"\n=== OWNER GUIDANCE (apply this when analysing) ===\n{user_hints}\n"

    # Memory: what was already done earlier today, so Claude manages rather than re-decides.
    memory_section = ""
    if day_history:
        memory_section = (
            "\n=== ACTIONS YOU ALREADY TOOK EARLIER TODAY (manage these, don't re-open blindly) ===\n"
            + json.dumps(day_history, separators=(",", ":")) + "\n"
        )

    last_run_section = ""
    if is_last_run:
        last_run_section = (
            "\n=== THIS IS THE FINAL RUN OF THE TRADING DAY ===\n"
            "For EACH open position decide explicitly: close it now to avoid overnight risk, "
            "or hold it overnight only if the thesis and risk clearly justify carrying it. "
            "Close day-trade positions unless there is a strong reason to hold.\n"
        )

    user_msg = f"""\
CURRENT DATE/TIME (ET): {datetime.now(ET).strftime('%Y-%m-%d %H:%M')}
MARKET SESSION PHASE: {phase}
AVAILABLE BUYING POWER: ${portfolio['buying_power']:,.2f}
PORTFOLIO VALUE: ${portfolio['portfolio_value']:,.2f}
OPEN POSITIONS: {len(portfolio['positions'])}
{hints_section}{memory_section}{last_run_section}
=== RECENT FINANCIAL NEWS (Alpaca + Financial Times) ===
{news_blob}

=== MARKET DATA (per symbol: price, prev_close, open, day_high, day_low, pct_from_prev_close, pct_from_open, rel_volume) ===
{market_blob}
Use pct_from_prev_close for the gap / day move, pct_from_open for intraday direction,
and rel_volume (>1 = above-average volume) to confirm momentum before acting.

=== CURRENT PORTFOLIO ===
{portfolio_blob}

Analyze all news sources carefully. Identify the strongest catalysts.
Apply the full analysis framework. Provide specific, actionable trading recommendations.
"""
    try:
        response = ai_client.messages.create(
            model=ANALYSIS_MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        if response.stop_reason == "max_tokens":
            log.warning("Claude hit max_tokens — JSON may be truncated this cycle.")

        # Pull the text block (adaptive thinking emits a separate thinking block first).
        raw = "\n".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
        if not raw:
            log.error("Claude returned no text content this cycle.")
            return None

        # Strip markdown fences if present
        if "```" in raw:
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        # Extract the outermost JSON object, ignoring any surrounding text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]

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


def _round_tick(price: float) -> float:
    """Round to a valid order price (penny for >= $1, else 4 decimals)."""
    return round(price, 2) if price >= 1 else round(price, 4)


def fetch_latest_price(stock_data_client, symbol: str) -> float | None:
    """Single-symbol fresh quote, used right before a retry so the stop is
    computed against a current reference price (pre-market spreads drift fast)."""
    try:
        quotes = stock_data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        q = quotes.get(symbol)
        if q:
            price = getattr(q, "ask_price", None) or getattr(q, "bid_price", None)
            if price and float(price) > 0:
                return round(float(price), 2)
    except Exception as e:
        log.warning(f"  Fresh quote fetch failed for {symbol}: {e}")
    return None


def _retry_buy_bracket_then_market(trading_client, stock_data_client, symbol, qty,
                                   effective, sl_pct, tp_pct, first_error):
    """A bracket BUY was rejected (usually a stale-price stop above Alpaca's
    base_price). Refetch a fresh quote, retry the bracket with the stop clamped
    strictly below it, and only if that also fails fall back to a plain market
    order so the trade isn't lost. Returns the order, or None if all attempts
    fail. The dropped stop on the fallback path is re-attached on a later cycle."""
    log.warning(f"  BUY {symbol} bracket rejected ({first_error}); retrying with fresh price.")

    fresh = fetch_latest_price(stock_data_client, symbol) if stock_data_client else None
    if fresh:
        # Stop must sit strictly below Alpaca's reference; clamp to <= fresh - 1 tick
        # so a tight sl_pct can never land at/above base_price.
        sl_price = _round_tick(min(fresh * (1 - sl_pct / 100), fresh - 0.01))
        tp_price = _round_tick(fresh * (1 + tp_pct / 100))
        if 0 < sl_price < fresh < tp_price:
            try:
                order = trading_client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=sl_price),
                ))
                log.info(f"  BUY {symbol} bracket retry OK | {qty}sh | "
                         f"SL ${sl_price} / TP ${tp_price} | order={order.id}")
                return order
            except Exception as e2:
                log.warning(f"  BUY {symbol} bracket retry failed ({e2}); "
                            f"falling back to plain market order.")

    # Fallback: plain market order, no bracket. Stop re-attached next cycle.
    try:
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, notional=effective, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
        log.info(f"  BUY {symbol} placed WITHOUT bracket (${effective:.0f}) | "
                 f"order={order.id} | stop re-attached next cycle")
        return order
    except Exception as e3:
        log.error(f"  BUY {symbol} FAILED after retries: {e3}")
        return None


def execute(trading_client: TradingClient, actions: list[dict], portfolio: dict, dry_run: bool, prices: dict | None = None, stock_data_client=None) -> None:
    if not actions:
        log.info("No trading actions this cycle.")
        return

    open_positions = {p["symbol"]: p for p in portfolio.get("positions", [])}
    buying_power   = portfolio.get("buying_power", 0.0)
    total_bp       = buying_power

    for action in actions:
        symbol   = action.get("symbol", "").upper().strip()
        act      = action.get("action", "hold").lower()
        notional = float(action.get("notional_usd") or 0)
        reason   = action.get("reasoning", "")
        ttype    = action.get("trade_type", "swing")
        conv     = action.get("conviction", 3)
        sl_pct   = action.get("stop_loss_pct")
        tp_pct   = action.get("take_profit_pct")
        tier     = action.get("catalyst_tier", "B")
        leverage = 1
        if LEVERAGE_ENABLED:
            leverage = max(1, min(int(MAX_LEVERAGE), int(action.get("leverage", 1))))

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

            effective = notional * leverage

            # Hard cap: no single trade > 50% of cycle starting buying_power
            hard_cap = total_bp * 0.50
            if effective > hard_cap:
                log.info(f"  Capping {symbol} at 50% BP: ${effective:.0f} → ${hard_cap:.0f}")
                effective = hard_cap

            # Never exhaust buying power; keep 5% buffer
            bp_cap = buying_power * 0.95
            if effective > bp_cap:
                effective = bp_cap
                log.info(f"  Capping {symbol} to available buying_power: ${effective:.0f}")

            if effective <= 0:
                log.warning(f"  SKIP buy {symbol}: insufficient buying power")
                continue

            # Validate the asset is tradeable on Alpaca before placing any order
            if not is_asset_tradeable(trading_client, symbol):
                log.warning(f"  SKIP buy {symbol}: not tradeable on Alpaca")
                continue

            price  = (prices or {}).get(symbol)
            crypto = is_crypto(symbol)
            qty    = int(effective / price) if (price and price > 0) else 0
            use_bracket = (not crypto) and qty >= 1 and sl_pct and tp_pct

            if use_bracket:
                tp_price = _round_tick(price * (1 + tp_pct / 100))
                sl_price = _round_tick(price * (1 - sl_pct / 100))
                order_req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=sl_price),
                )
                exec_note = f"{qty}sh | auto SL ${sl_price} / TP ${tp_price}"
            else:
                tif = TimeInForce.GTC if crypto else TimeInForce.DAY
                order_req = MarketOrderRequest(
                    symbol=symbol, notional=effective, side=OrderSide.BUY, time_in_force=tif,
                )
                tgt = (f" SL-{sl_pct}%" if sl_pct else "") + (f" TP+{tp_pct}%" if tp_pct else "")
                exec_note = f"${effective:.0f} | no bracket (targets:{tgt or ' none'})"

            lev_tag = f" | LEV {leverage}x" if leverage > 1 else ""
            if dry_run:
                log.info(f"  [DRY-RUN] BUY {symbol} [{ttype} | tier={tier} | conv={conv}{lev_tag}] {exec_note} | {reason}")
            else:
                try:
                    order = trading_client.submit_order(order_req)
                    log.info(f"  BUY {symbol} [{ttype} | tier={tier} | conv={conv}{lev_tag}] {exec_note} | order={order.id} | {reason}")
                    buying_power -= effective
                    open_positions[symbol] = {"symbol": symbol, "qty": 0}
                except Exception as e:
                    order = None
                    if use_bracket:
                        order = _retry_buy_bracket_then_market(
                            trading_client, stock_data_client, symbol, qty,
                            effective, sl_pct, tp_pct, first_error=e,
                        )
                    if order:
                        buying_power -= effective
                        open_positions[symbol] = {"symbol": symbol, "qty": 0}
                    else:
                        log.error(f"  BUY {symbol} FAILED: {e}")

        # ── SHORT (open a new short position) ────────────────────────────────
        elif act == "short":
            if is_crypto(symbol):
                log.warning(f"  SKIP short {symbol}: crypto shorting not supported")
                continue
            if notional <= 0:
                log.warning(f"  SKIP short {symbol}: notional_usd is 0")
                continue

            effective = notional * leverage

            hard_cap = total_bp * 0.50
            if effective > hard_cap:
                log.info(f"  Capping short {symbol} at 50% BP: ${effective:.0f} → ${hard_cap:.0f}")
                effective = hard_cap

            bp_cap = buying_power * 0.95
            if effective > bp_cap:
                effective = bp_cap
                log.info(f"  Capping short {symbol} to available buying_power: ${effective:.0f}")

            if effective <= 0:
                log.warning(f"  SKIP short {symbol}: insufficient buying power")
                continue

            if not is_asset_tradeable(trading_client, symbol):
                log.warning(f"  SKIP short {symbol}: not tradeable on Alpaca")
                continue

            # Short orders require qty (integer shares), not notional
            price = (prices or {}).get(symbol)
            if not price or price <= 0:
                log.warning(f"  SKIP short {symbol}: no price available to calculate qty")
                continue
            qty = int(effective / price)
            if qty <= 0:
                log.warning(f"  SKIP short {symbol}: calculated qty=0 (effective=${effective:.0f}, price=${price})")
                continue

            lev_tag = f" | LEV {leverage}x" if leverage > 1 else ""

            if sl_pct and tp_pct:
                tp_price = _round_tick(price * (1 - tp_pct / 100))   # profit when price falls
                sl_price = _round_tick(price * (1 + sl_pct / 100))   # loss when price rises
                order_req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=sl_price),
                )
                exec_note = f"{qty}sh (~${effective:.0f}) | auto SL ${sl_price} / TP ${tp_price}"
            else:
                order_req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                )
                exec_note = f"{qty}sh (~${effective:.0f}) | no bracket"

            if dry_run:
                log.info(f"  [DRY-RUN] SHORT {symbol} [{ttype} | tier={tier} | conv={conv}{lev_tag}] {exec_note} | {reason}")
            else:
                try:
                    order = trading_client.submit_order(order_req)
                    log.info(f"  SHORT {symbol} [{ttype} | tier={tier} | conv={conv}{lev_tag}] {exec_note} | order={order.id} | {reason}")
                    buying_power -= effective
                    open_positions[symbol] = {"symbol": symbol, "qty": 0}
                except Exception as e:
                    log.error(f"  SHORT {symbol} FAILED: {e}")

        # ── CLOSE (full exit of long) ──────────────────────────────────────────
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

        # ── COVER (full exit of short) ─────────────────────────────────────────
        elif act == "cover":
            if symbol not in open_positions:
                log.warning(f"  SKIP cover {symbol}: no open position")
                continue
            if dry_run:
                log.info(f"  [DRY-RUN] COVER {symbol} | {reason}")
            else:
                try:
                    trading_client.close_position(symbol)
                    log.info(f"  COVER {symbol} | {reason}")
                    open_positions.pop(symbol, None)
                except Exception as e:
                    log.error(f"  COVER {symbol} FAILED: {e}")

        # ── PARTIAL CLOSE (~50% of long) ──────────────────────────────────────
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

        # ── COVER PARTIAL (~50% of short) ─────────────────────────────────────
        elif act == "cover_partial":
            if symbol not in open_positions:
                log.warning(f"  SKIP cover_partial {symbol}: no open position")
                continue
            if dry_run:
                log.info(f"  [DRY-RUN] COVER PARTIAL {symbol} (~50%) | {reason}")
            else:
                try:
                    trading_client.close_position(symbol, ClosePositionRequest(percentage="0.5"))
                    log.info(f"  COVER PARTIAL {symbol} (~50%) | {reason}")
                except Exception as e:
                    log.error(f"  COVER PARTIAL {symbol} FAILED: {e}")

        else:
            log.warning(f"  Unknown action '{act}' for {symbol} — skipping")


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run: bool, force: bool = False) -> None:
    if not force and not in_trading_window():
        log.info("Outside trading window — waiting.")
        return

    phase = session_phase()
    state = load_state()
    is_first_run   = len(state.get("cycles", [])) == 0
    _, is_last_run = run_slot_position()

    log.info("-" * 60)
    log.info(
        f"Cycle start: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} | Session: {phase.upper()} "
        f"| first_run={is_first_run} last_run={is_last_run}"
    )
    log.info("-" * 60)

    # Telegram: only check for owner suggestions on the FIRST run of the day.
    if is_first_run:
        poll_telegram_hints()

    news      = fetch_all_news(news_client, lookback_hours=NEWS_LOOKBACK_HOURS)
    portfolio = get_portfolio(trading_client)
    if state.get("start_portfolio_value") is None:
        state["start_portfolio_value"] = portfolio["portfolio_value"]

    news_syms   = [s for a in news for s in a.get("symbols", [])]
    held_syms   = [p["symbol"] for p in portfolio["positions"]]
    all_symbols = list(set(news_syms + WATCHLIST + held_syms))
    market      = fetch_market_data(stock_data_client, all_symbols)
    price_map   = {s: d["price"] for s, d in market.items() if "price" in d}

    log.info(
        f"Portfolio: ${portfolio['portfolio_value']:,.2f} | "
        f"Buying power: ${portfolio['buying_power']:,.2f} | "
        f"Open positions: {len(portfolio['positions'])}"
    )

    hints    = load_user_hints()
    analysis = analyze(
        ai_client, news, portfolio, market, phase,
        user_hints=hints, day_history=state.get("cycles", []), is_last_run=is_last_run,
    )
    if analysis is None:
        log.error("Analysis failed — skipping execution this cycle.")
        save_state(state)
        return

    actions = analysis.get("actions", [])
    execute(trading_client, actions, portfolio, dry_run, prices=price_map, stock_data_client=stock_data_client)

    # Memory: remember what we did this cycle (for later runs + the end-of-day summary).
    executed = [
        {
            "symbol":     a.get("symbol"),
            "action":     a.get("action"),
            "trade_type": a.get("trade_type"),
            "reasoning":  (a.get("reasoning") or "")[:200],
        }
        for a in actions if a.get("action", "hold") != "hold"
    ]
    state.setdefault("cycles", []).append({
        "time":      datetime.now(ET).strftime("%H:%M"),
        "phase":     phase,
        "sentiment": analysis.get("market_sentiment", ""),
        "catalyst":  (analysis.get("catalyst_summary", "") or "")[:300],
        "actions":   executed,
    })
    save_state(state)

    # Telegram: only send a day summary at the LAST run of the day.
    if is_last_run:
        summary = build_day_summary(ai_client, state, portfolio)
        send_telegram(summary)

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
    parser.add_argument(
        "--notify-test",
        action="store_true",
        help="Send a test Telegram message to verify the bot is configured correctly",
    )
    args = parser.parse_args()

    mode_tag = "DRY-RUN " if args.dry_run else ""
    mode_tag += "PAPER" if PAPER_TRADING else "LIVE"
    slots = ", ".join(f"{h:02d}:{m:02d}" for h, m in RUN_TIMES_ET)
    log.info(f"Starting AI Trading Bot [{mode_tag}] — fixed runs at {slots} ET")

    if not PAPER_TRADING and not args.dry_run:
        log.warning("⚠  LIVE TRADING IS ENABLED — real money will be used!")

    if args.notify_test:
        msg = (
            "*Trading Bot* — test notification\n"
            "If you see this, Telegram is configured correctly."
        )
        send_telegram(msg)
        log.info("Test notification sent (or failed — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return

    trading_client, news_client, stock_data_client, ai_client = build_clients()

    # Owner suggestions are checked on the first run of each day (see run_cycle).

    def cycle():
        run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run=args.dry_run)

    if args.once:
        log.info("Running single cycle (--once flag ignores market-hours guard).")
        run_cycle(trading_client, news_client, stock_data_client, ai_client, dry_run=True, force=True)
        return

    try:
        while True:
            nxt = next_run_time()
            wait_secs = (nxt - datetime.now(ET)).total_seconds()
            log.info(f"Next run at {nxt.strftime('%H:%M ET')} — sleeping {wait_secs / 60:.0f} min.")
            time.sleep(max(0, wait_secs))
            cycle()
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")


if __name__ == "__main__":
    main()
