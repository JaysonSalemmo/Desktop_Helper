"""
Stock quotes via yfinance.

If the user's message names tickers ("How's NVDA doing?") those are used;
otherwise the config watchlist is quoted. Result string matches the training
format: "AAPL: $189.44 (+1.2%)" joined by ", ".
"""
import re

import yfinance as yf

# uppercase 1-5 letter tokens that look like tickers; common uppercase words
# people actually type in questions are excluded
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_NOT_TICKERS = {"A", "I", "OK", "USD", "ETF", "CEO", "IPO", "AI", "US", "UP"}


def extract_symbols(message: str) -> list[str]:
    seen = []
    for tok in _TICKER_RE.findall(message):
        if tok not in _NOT_TICKERS and tok not in seen:
            seen.append(tok)
    return seen


def _quote(symbol: str) -> str:
    info = yf.Ticker(symbol).fast_info
    price = info["last_price"]
    prev = info["previous_close"]
    change = (price - prev) / prev * 100
    sign = "+" if change >= 0 else ""
    return f"{symbol}: ${price:.2f} ({sign}{change:.1f}%)"


def quotes(message: str, watchlist: list[str]) -> str:
    symbols = extract_symbols(message) or watchlist
    parts = []
    for sym in symbols[:5]:  # keep the injected result short for the model
        try:
            parts.append(_quote(sym))
        except Exception:
            parts.append(f"{sym}: no data")
    return ", ".join(parts)
