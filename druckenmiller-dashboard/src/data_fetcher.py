"""
Data fetching module using yfinance with local caching
"""

import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from config.settings import CACHE_DIR, CACHE_EXPIRY_HOURS


class DataFetcher:
    """Fetches and caches stock price data using yfinance"""
    
    def __init__(self, cache_dir: str = CACHE_DIR, cache_expiry_hours: float = CACHE_EXPIRY_HOURS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
    
    def _get_cache_path(self, ticker: str, timeframe: str) -> Path:
        """Generate cache file path for a ticker and timeframe"""
        safe_ticker = ticker.replace(".", "_").upper()
        safe_timeframe = timeframe.replace("/", "_")
        return self.cache_dir / f"{safe_ticker}_{safe_timeframe}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is still valid"""
        if not cache_path.exists():
            return False
        
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        return age < timedelta(hours=self.cache_expiry_hours)
    
    def fetch_data(
        self, 
        ticker: str, 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Bar timeframe ("1d", "1wk", "1mo")
            period: Data period ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        
        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        cache_path = self._get_cache_path(ticker, timeframe)
        
        # Try to load from cache
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data
            except Exception as e:
                print(f"Error loading cache for {ticker}: {e}")
        
        # Fetch from yfinance
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1d": "1d",
                "1wk": "1wk",
                "1mo": "1mo"
            }
            interval = interval_map.get(timeframe, "1d")
            
            # Fetch data
            df = ticker_obj.history(period=period, interval=interval)
            
            if df.empty:
                print(f"No data returned for {ticker}")
                return None
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Cache the data
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
            except Exception as e:
                print(f"Error caching data for {ticker}: {e}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def fetch_multiple(
        self, 
        tickers: list[str], 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            data = self.fetch_data(ticker, timeframe, period)
            if data is not None:
                results[ticker] = data
        return results
    
    def clear_cache(self, ticker: Optional[str] = None):
        """Clear cache for a specific ticker or all tickers"""
        if ticker:
            # Clear specific ticker for all timeframes
            pattern = ticker.replace(".", "_").upper()
            for cache_file in self.cache_dir.glob(f"{pattern}_*.pkl"):
                cache_file.unlink()
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
