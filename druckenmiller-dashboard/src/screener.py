"""
Screening module for scanning watchlists and ranking by signal strength
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

from src.data_fetcher import DataFetcher
from src.derivatives import DerivativesCalculator
from src.signals import SignalDetector
from config.settings import DEFAULT_TIMEFRAME, SIGNAL_LOOKBACK_DAYS

# ETF Sector Mapping
ETF_SECTOR_MAP = {
    'SPY': 'Broad Market',
    'QQQ': 'Tech/Growth',
    'IWM': 'Small Cap',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLI': 'Industrials',
    'XLV': 'Healthcare',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communications',
    'GLD': 'Commodities',
    'TLT': 'Bonds',
    'UUP': 'Currency',
}


class Screener:
    """Scans watchlist and ranks tickers by signal strength"""
    
    def __init__(
        self,
        timeframe: str = DEFAULT_TIMEFRAME,
        lookback_days: int = SIGNAL_LOOKBACK_DAYS
    ):
        self.data_fetcher = DataFetcher()
        self.derivatives_calc = DerivativesCalculator()
        self.signal_detector = SignalDetector(lookback_days=lookback_days)
        self.timeframe = timeframe
        self._sector_cache = {}  # Cache for sector lookups
    
    def scan_ticker(
        self,
        ticker: str,
        period: str = "6mo"
    ) -> Optional[Dict]:
        """
        Scan a single ticker and return signal information
        
        Returns:
            Dictionary with ticker info and signals, or None if error
        """
        # Fetch data
        df = self.data_fetcher.fetch_data(ticker, self.timeframe, period)
        if df is None or df.empty:
            return None
        
        # Calculate derivatives
        df = self.derivatives_calc.calculate_all(df)
        
        # Detect signals
        df = self.signal_detector.analyze(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Get current signal_type (from quadrant logic)
        current_signal_type = latest.get('signal_type', 'NEUTRAL')
        
        # Check for recent zero crossings (for historical context)
        recent_signals = self.signal_detector.get_recent_signals(df)
        
        # Calculate signal strength
        acceleration = latest['acceleration']
        prev_acceleration = prev['acceleration']
        roc = latest['roc']
        
        # Check if acceleration just crossed zero
        just_crossed = False
        cross_direction = None
        if (prev_acceleration < 0 and acceleration >= 0):
            just_crossed = True
            cross_direction = "up"
        elif (prev_acceleration > 0 and acceleration <= 0):
            just_crossed = True
            cross_direction = "down"
        
        # Calculate days since last zero cross
        days_since_cross = None
        if 'zero_crossing' in df.columns:
            # Find the most recent zero crossing
            crossings = df[df['zero_crossing'] == True]
            if not crossings.empty:
                # Get the index of the most recent crossing
                last_cross_idx = crossings.index[-1]
                # Calculate trading days since cross
                # Count rows between last cross and latest (inclusive)
                # If cross is at index i and latest is at index -1, days = (len - 1) - i
                last_cross_position = df.index.get_loc(last_cross_idx)
                latest_position = len(df) - 1
                days_since_cross = latest_position - last_cross_position
        
        # Magnitude of acceleration change
        accel_change = abs(acceleration - prev_acceleration) if not pd.isna(prev_acceleration) else 0
        
        # Volume confirmation (if available)
        volume_confirmation = 0
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            if avg_volume > 0:
                volume_confirmation = recent_volume / avg_volume
        
        # Calculate signal strength score
        # Higher score = stronger signal
        signal_strength = 0
        
        # Base score based on current signal type
        if current_signal_type == 'POTENTIAL_BOTTOM':
            signal_strength += 40
        elif current_signal_type == 'POTENTIAL_TOP':
            signal_strength += 40
        elif current_signal_type == 'ACCELERATING_UP':
            signal_strength += 20
        elif current_signal_type == 'ACCELERATING_DOWN':
            signal_strength += 20
        
        if just_crossed:
            signal_strength += 30  # Bonus for zero crossing
        
        # Add points for magnitude of change
        signal_strength += min(accel_change * 10, 20)
        
        # Add points for volume confirmation
        if volume_confirmation > 1.2:
            signal_strength += 10
        
        # Get sector information
        sector = self.get_sector(ticker)
        
        return {
            'ticker': ticker,
            'sector': sector,
            'price': latest['close'],
            'roc': roc,
            'acceleration': acceleration,
            'acceleration_change': accel_change,
            'signal_type': current_signal_type,  # Current state, not historical
            'just_crossed_zero': just_crossed,
            'cross_direction': cross_direction,
            'days_since_cross': days_since_cross,
            'volume_confirmation': volume_confirmation,
            'signal_strength': signal_strength,
            'recent_signals': recent_signals,
            'signal_count': len(recent_signals),
            'latest_signal_date': recent_signals[0]['timestamp'] if recent_signals else None
        }
    
    def scan_watchlist(
        self,
        tickers: List[str],
        period: str = "6mo"
    ) -> pd.DataFrame:
        """
        Scan entire watchlist and return ranked results
        
        Args:
            tickers: List of ticker symbols
            period: Data period to fetch
        
        Returns:
            DataFrame sorted by signal strength
        """
        results = []
        
        for ticker in tickers:
            try:
                result = self.scan_ticker(ticker, period)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by signal strength (descending)
        df = df.sort_values('signal_strength', ascending=False)
        
        # Format columns for display
        if 'latest_signal_date' in df.columns:
            df['latest_signal_date'] = pd.to_datetime(df['latest_signal_date'])
        
        return df
    
    def get_top_signals(
        self,
        tickers: List[str],
        top_n: int = 10,
        period: str = "6mo"
    ) -> pd.DataFrame:
        """
        Get top N signals from watchlist
        
        Args:
            tickers: List of ticker symbols
            top_n: Number of top signals to return
            period: Data period to fetch
        
        Returns:
            DataFrame with top N signals
        """
        df = self.scan_watchlist(tickers, period)
        return df.head(top_n)
    
    def get_sector(self, ticker: str) -> str:
        """
        Get sector for a ticker, using cache and ETF mapping
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Sector name or 'ETF/Other' if not found
        """
        ticker_upper = ticker.upper()
        
        # Check cache first
        if ticker_upper in self._sector_cache:
            return self._sector_cache[ticker_upper]
        
        # Check ETF mapping
        if ticker_upper in ETF_SECTOR_MAP:
            sector = ETF_SECTOR_MAP[ticker_upper]
            self._sector_cache[ticker_upper] = sector
            return sector
        
        # Fetch from yfinance
        try:
            ticker_info = yf.Ticker(ticker_upper).info
            sector = ticker_info.get('sector', 'ETF/Other')
            if not sector or sector == 'None':
                sector = 'ETF/Other'
            self._sector_cache[ticker_upper] = sector
            return sector
        except Exception as e:
            print(f"Error fetching sector for {ticker_upper}: {e}")
            sector = 'ETF/Other'
            self._sector_cache[ticker_upper] = sector
            return sector
