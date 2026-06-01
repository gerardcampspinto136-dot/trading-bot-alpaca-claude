#!/usr/bin/env python3
"""
Run this once before starting the bot to verify all connections work.
  py check_setup.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

PASS = "  [OK]"
FAIL = "  [FAIL]"

SEP = "-" * 40

print(f"\n{SEP}")
print("  Trading Bot -- Setup Check")
print(f"{SEP}\n")

errors = 0

# 1. Check .env keys exist
print("1. Checking .env keys...")
for name, val in [
    ("ALPACA_API_KEY",    ALPACA_API_KEY),
    ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY),
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
]:
    if val and not val.startswith("your_"):
        print(f"{PASS} {name} is set")
    else:
        print(f"{FAIL} {name} is missing -- edit your .env file")
        errors += 1

# 2. Alpaca paper trading connection
print("\n2. Testing Alpaca paper trading connection...")
try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    acct = client.get_account()
    buying_power    = float(acct.buying_power)
    portfolio_value = float(acct.portfolio_value)
    cash            = float(acct.cash)
    print(f"{PASS} Connected to Alpaca Paper Trading")
    print(f"       Portfolio value : ${portfolio_value:>12,.2f}")
    print(f"       Cash            : ${cash:>12,.2f}")
    print(f"       Buying power    : ${buying_power:>12,.2f}")
    positions = client.get_all_positions()
    print(f"       Open positions  : {len(positions)}")
except Exception as e:
    print(f"{FAIL} Alpaca connection failed: {e}")
    errors += 1

# 3. Alpaca news API
print("\n3. Testing Alpaca news API...")
try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    from datetime import datetime, timedelta
    import pytz

    nc = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    since = datetime.now(pytz.utc) - timedelta(hours=6)
    resp = nc.get_news(NewsRequest(start=since, limit=5, include_content=False))
    raw  = resp.news if hasattr(resp, "news") else list(resp)
    # Newer alpaca-py returns (key, news_obj) tuples — unwrap them
    unwrapped = []
    for item in raw:
        if isinstance(item, tuple):
            val = item[1] if len(item) > 1 else item[0]
            unwrapped.extend(val) if isinstance(val, list) else unwrapped.append(val)
        else:
            unwrapped.append(item)
    count = len(unwrapped)
    print(f"{PASS} News API reachable -- {count} recent articles found")
    if count > 0:
        a = unwrapped[0]
        headline = a.get("headline", "") if isinstance(a, dict) else getattr(a, "headline", "")
        print(f"       Latest: {headline[:70]}")
except Exception as e:
    print(f"{FAIL} News API failed: {e}")
    print("       (You may need a paid Alpaca data subscription for news)")
    errors += 1

# 4. Stock price feed
print("\n4. Testing stock price feed...")
try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    sd = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    quotes = sd.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=["AAPL", "SPY"]))
    for sym, q in quotes.items():
        price = getattr(q, "ask_price", None) or getattr(q, "bid_price", None)
        print(f"{PASS} {sym}: ${float(price):,.2f}")
except Exception as e:
    print(f"{FAIL} Stock price feed failed: {e}")
    errors += 1

# 5. Anthropic / Claude connection
print("\n5. Testing Claude (Anthropic) connection...")
try:
    import anthropic
    ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with: OK"}],
    )
    reply = msg.content[0].text.strip()
    print(f"{PASS} Claude API connected -- response: '{reply}'")
except Exception as e:
    print(f"{FAIL} Claude API failed: {e}")
    errors += 1

# Summary
print(f"\n{SEP}")
if errors == 0:
    print("  All 5 checks passed -- you're ready to run the bot!")
    print("  Command: py trading_bot.py --dry-run")
else:
    print(f"  {errors} check(s) failed -- fix the issues above before starting.")
print(f"{SEP}\n")
sys.exit(errors)
