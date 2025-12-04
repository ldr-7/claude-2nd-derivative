"""
Signal detection logic for identifying trend inflection points
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from config.settings import SIGNAL_LOOKBACK_DAYS


class SignalDetector:
    """Detects trading signals based on acceleration zero-crossings and divergences"""
    
    def __init__(self, lookback_days: int = SIGNAL_LOOKBACK_DAYS):
        self.lookback_days = lookback_days
    
    def calculate_current_signal_type(
        self,
        df: pd.DataFrame,
        acceleration_col: str = "acceleration",
        roc_col: str = "roc"
    ) -> pd.DataFrame:
        """
        Calculate current signal_type based on acceleration/ROC quadrant matrix.
        This reflects the CURRENT state, not historical events.
        
        Quadrant logic:
        - acceleration > 0 and roc < 0: POTENTIAL_BOTTOM (deceleration of downtrend)
        - acceleration < 0 and roc > 0: POTENTIAL_TOP (deceleration of uptrend)
        - acceleration > 0 and roc > 0: ACCELERATING_UP (bullish momentum increasing)
        - acceleration < 0 and roc < 0: ACCELERATING_DOWN (bearish momentum increasing)
        - else: NEUTRAL
        
        Args:
            df: DataFrame with acceleration and ROC columns
            acceleration_col: Column name for acceleration
            roc_col: Column name for ROC
        
        Returns:
            DataFrame with added signal_type column
        """
        result = df.copy()
        
        acceleration = result[acceleration_col]
        roc = result[roc_col]
        
        # Initialize signal_type column
        result['signal_type'] = 'NEUTRAL'
        
        # Handle NaN values - set to NEUTRAL if either is NaN
        valid_mask = acceleration.notna() & roc.notna()
        
        # Quadrant 1: acceleration > 0 and roc < 0 -> POTENTIAL_BOTTOM
        mask1 = valid_mask & (acceleration > 0) & (roc < 0)
        result.loc[mask1, 'signal_type'] = 'POTENTIAL_BOTTOM'
        
        # Quadrant 2: acceleration < 0 and roc > 0 -> POTENTIAL_TOP
        mask2 = valid_mask & (acceleration < 0) & (roc > 0)
        result.loc[mask2, 'signal_type'] = 'POTENTIAL_TOP'
        
        # Quadrant 3: acceleration > 0 and roc > 0 -> ACCELERATING_UP
        mask3 = valid_mask & (acceleration > 0) & (roc > 0)
        result.loc[mask3, 'signal_type'] = 'ACCELERATING_UP'
        
        # Quadrant 4: acceleration < 0 and roc < 0 -> ACCELERATING_DOWN
        mask4 = valid_mask & (acceleration < 0) & (roc < 0)
        result.loc[mask4, 'signal_type'] = 'ACCELERATING_DOWN'
        
        return result
    
    def detect_zero_crossings(
        self, 
        df: pd.DataFrame,
        acceleration_col: str = "acceleration",
        roc_col: str = "roc"
    ) -> pd.DataFrame:
        """
        Detect when acceleration crosses zero (for historical analysis)
        
        Args:
            df: DataFrame with acceleration and ROC columns
            acceleration_col: Column name for acceleration
            roc_col: Column name for ROC
        
        Returns:
            DataFrame with added column: zero_crossing
        """
        result = df.copy()
        
        # Detect zero crossings
        acceleration = result[acceleration_col]
        prev_acceleration = acceleration.shift(1)
        
        # Cross from below (negative to positive)
        cross_up = (prev_acceleration < 0) & (acceleration >= 0)
        
        # Cross from above (positive to negative)
        cross_down = (prev_acceleration > 0) & (acceleration <= 0)
        
        # Initialize zero_crossing column
        result['zero_crossing'] = cross_up | cross_down
        
        return result
    
    def detect_divergences(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        acceleration_col: str = "acceleration",
        roc_col: str = "roc",
        lookback_periods: int = 20
    ) -> pd.DataFrame:
        """
        Detect divergences between price and acceleration/ROC
        
        Args:
            df: DataFrame with price, acceleration, and ROC
            price_col: Column name for price
            acceleration_col: Column name for acceleration
            roc_col: Column name for ROC
            lookback_periods: Periods to look back for divergence detection
        
        Returns:
            DataFrame with added columns: bearish_divergence, bullish_divergence
        """
        result = df.copy()
        
        price = result[price_col]
        acceleration = result[acceleration_col]
        roc = result[roc_col]
        
        # Initialize divergence columns
        result['bearish_divergence'] = False
        result['bullish_divergence'] = False
        
        # Rolling max/min for price
        price_max = price.rolling(window=lookback_periods).max()
        price_min = price.rolling(window=lookback_periods).min()
        
        # Rolling max/min for acceleration
        accel_max = acceleration.rolling(window=lookback_periods).max()
        accel_min = acceleration.rolling(window=lookback_periods).min()
        
        # Bearish divergence: price making new highs but acceleration negative/declining
        new_price_high = price >= price_max.shift(1)
        negative_acceleration = acceleration < 0
        declining_acceleration = acceleration < accel_max.shift(1)
        
        result['bearish_divergence'] = new_price_high & (negative_acceleration | declining_acceleration)
        
        # Bullish divergence: price making new lows but acceleration positive/rising
        new_price_low = price <= price_min.shift(1)
        positive_acceleration = acceleration > 0
        rising_acceleration = acceleration > accel_min.shift(1)
        
        result['bullish_divergence'] = new_price_low & (positive_acceleration | rising_acceleration)
        
        return result
    
    def get_recent_signals(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal_type"
    ) -> List[Dict]:
        """
        Extract recent signals within lookback period
        
        Args:
            df: DataFrame with signals
            signal_col: Column name containing signal types
        
        Returns:
            List of dictionaries with signal information
        """
        signals = []
        
        # Filter for non-null signals
        signal_mask = df[signal_col].notna()
        signal_df = df[signal_mask].copy()
        
        if signal_df.empty:
            return signals
        
        # Get recent signals (last N days)
        recent_df = signal_df.tail(self.lookback_days * 2)  # Extra buffer for safety
        
        for idx, row in recent_df.iterrows():
            signal_info = {
                'timestamp': idx if isinstance(idx, (pd.Timestamp, datetime)) else pd.Timestamp(idx),
                'signal_type': row[signal_col],
                'price': row.get('close', row.get('smoothed_price', None)),
                'roc': row.get('roc', None),
                'acceleration': row.get('acceleration', None),
            }
            signals.append(signal_info)
        
        return signals
    
    def analyze(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        acceleration_col: str = "acceleration",
        roc_col: str = "roc"
    ) -> pd.DataFrame:
        """
        Complete signal analysis: current signal types, zero crossings, and divergences
        
        Args:
            df: DataFrame with price, ROC, and acceleration
            price_col: Column name for price
            acceleration_col: Column name for acceleration
            roc_col: Column name for ROC
        
        Returns:
            DataFrame with all signal columns added
        """
        # Calculate current signal_type based on quadrant logic (CURRENT state)
        df = self.calculate_current_signal_type(df, acceleration_col, roc_col)
        
        # Detect zero crossings (for historical analysis)
        df = self.detect_zero_crossings(df, acceleration_col, roc_col)
        
        # Detect divergences
        df = self.detect_divergences(df, price_col, acceleration_col, roc_col)
        
        return df
