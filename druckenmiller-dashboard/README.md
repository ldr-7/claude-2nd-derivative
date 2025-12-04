# Second-Derivative Rate-of-Change Trading Dashboard

A Python trading dashboard that implements second-derivative analysis for identifying trend inflection points, inspired by Stanley Druckenmiller's approach to catching rotations early.

## Core Concept

The dashboard analyzes price movements through the lens of calculus:

- **Price** = f(x) = zeroth derivative (position)
- **ROC/Velocity** = f'(x) = first derivative (momentum)
- **Acceleration** = f''(x) = second derivative (change in momentum)

By tracking when acceleration crosses zero, we can identify potential trend reversals before they become obvious in price action.

## Features

### Data Layer
- Uses `yfinance` for real-time price data
- Supports multiple tickers and timeframes (daily, weekly, monthly)
- Local caching to avoid repeated API calls

### Calculation Engine
- Configurable EMA/SMA smoothing (default 21-period)
- First derivative: percentage ROC over N periods (default 14)
- Second derivative: ROC of the ROC (acceleration)
- Optional Savitzky-Golay filter for cleaner derivatives
- All parameters are user-adjustable

### Signal Detection
- Flags when acceleration crosses zero from below while ROC < 0 (potential bottom)
- Flags when acceleration crosses zero from above while ROC > 0 (potential top)
- Detects divergences: price making new highs but acceleration negative (and vice versa)
- Stores signals with timestamps for backtesting

### Visualization
- **Main chart panel**: Candlestick or line chart of price with smoothed overlay
- **Second panel**: ROC with zero line highlighted
- **Third panel**: Acceleration with zero line highlighted, shaded green when positive, red when negative
- Vertical markers on all panels when acceleration crosses zero
- Color-coded signals: green triangles for potential bottoms, red triangles for potential tops

### Screening Module
- Input a watchlist of tickers
- Scan all tickers and rank by:
  - Acceleration just crossed zero
  - Magnitude of acceleration change
  - Confirmation with volume
- Output a table sorted by signal strength

### Dashboard Features
- Ticker search/input
- Timeframe selector (1D, 1W, 1M bars)
- Parameter sliders: smoothing period, ROC lookback, filter strength
- Toggle between raw and smoothed derivatives
- Export signals to CSV

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app/dashboard.py
```

The dashboard will open in your default web browser.

## Usage

### Single Ticker Analysis

1. Enter a ticker symbol in the sidebar
2. Select your preferred timeframe and data period
3. Adjust calculation parameters as needed:
   - Smoothing type (EMA/SMA) and period
   - ROC lookback period
   - Enable/disable Savitzky-Golay filter
4. View the combined chart showing price, ROC, and acceleration
5. Review recent signals in the table below
6. Export signals to CSV if needed

### Watchlist Screener

1. Navigate to the "Watchlist Screener" tab
2. Enter tickers (one per line) or use the default watchlist
3. Set the signal lookback period (default: 3 days)
4. Click "Scan Watchlist"
5. Review ranked results sorted by signal strength
6. Top signals with zero crossings are highlighted
7. Export results to CSV

## Default Watchlist

The dashboard comes with a default watchlist including:
- Major indices: SPY, QQQ, IWM
- Sectors: XLF, XLE, XLK, XLI, XLV
- Tech stocks: AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSM
- Other: GLD, TLT, UUP

You can modify `config/watchlist.txt` to customize your watchlist.

## Configuration

Default parameters can be adjusted in `config/settings.py`:

- `DEFAULT_SMOOTHING_PERIOD`: 21
- `DEFAULT_SMOOTHING_TYPE`: "EMA"
- `DEFAULT_ROC_PERIOD`: 14
- `USE_SAVITZKY_GOLAY`: True
- `SIGNAL_LOOKBACK_DAYS`: 3
- `DEFAULT_TIMEFRAME`: "1d"
- `CACHE_EXPIRY_HOURS`: 1

## File Structure

```
druckenmiller-dashboard/
├── src/
│   ├── data_fetcher.py      # yfinance wrapper with caching
│   ├── derivatives.py        # calculation engine for ROC and acceleration
│   ├── signals.py            # signal detection logic
│   └── screener.py           # watchlist scanning
├── app/
│   ├── dashboard.py          # main Streamlit app
│   └── charts.py             # Plotly chart components
├── config/
│   ├── settings.py           # default parameters
│   └── watchlist.txt         # default tickers to scan
├── requirements.txt
└── README.md
```

## Key Use Case

The primary use case is quickly scanning for "what's about to turn" - the screener output and signal highlighting are designed to show at a glance which names have acceleration crossing zero TODAY or in the last 3 sessions.

## Technical Notes

- Data is cached locally in `.cache/` directory to minimize API calls
- Cache expires after 1 hour by default (configurable)
- The Savitzky-Golay filter helps smooth noisy derivatives but can be disabled
- Signal detection focuses on zero crossings of acceleration, which indicate momentum shifts

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## License

This project is provided as-is for educational purposes.
