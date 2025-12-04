"""
Derivative calculation engine for ROC and acceleration analysis
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional

from config.settings import (
    DEFAULT_SMOOTHING_PERIOD,
    DEFAULT_SMOOTHING_TYPE,
    DEFAULT_ROC_PERIOD,
    USE_SAVITZKY_GOLAY,
    SG_WINDOW_LENGTH,
    SG_POLYORDER
)


class DerivativesCalculator:
    """Calculates first and second derivatives (ROC and acceleration)"""
    
    def __init__(
        self,
        smoothing_period: int = DEFAULT_SMOOTHING_PERIOD,
        smoothing_type: str = DEFAULT_SMOOTHING_TYPE,
        roc_period: int = DEFAULT_ROC_PERIOD,
        use_savgol: bool = USE_SAVITZKY_GOLAY,
        sg_window: int = SG_WINDOW_LENGTH,
        sg_polyorder: int = SG_POLYORDER
    ):
        self.smoothing_period = smoothing_period
        self.smoothing_type = smoothing_type.upper()
        self.roc_period = roc_period
        self.use_savgol = use_savgol
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
    
    def calculate_smoothed_price(self, df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        """
        Calculate smoothed price using EMA or SMA
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price (default: "close")
        
        Returns:
            Series of smoothed prices
        """
        if self.smoothing_type == "EMA":
            return df[price_col].ewm(span=self.smoothing_period, adjust=False).mean()
        elif self.smoothing_type == "SMA":
            return df[price_col].rolling(window=self.smoothing_period).mean()
        else:
            raise ValueError(f"Unknown smoothing type: {self.smoothing_type}")
    
    def calculate_roc(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Rate of Change (first derivative)
        
        Args:
            prices: Series of prices (smoothed or raw)
            period: Lookback period (defaults to self.roc_period)
        
        Returns:
            Series of ROC values as percentages
        """
        if period is None:
            period = self.roc_period
        
        roc = prices.pct_change(periods=period) * 100
        
        # Apply Savitzky-Golay filter if enabled
        if self.use_savgol and len(roc) > self.sg_window:
            roc_values = roc.values
            # Handle NaN values
            valid_mask = ~np.isnan(roc_values)
            valid_count = valid_mask.sum()
            if valid_count > self.sg_window:
                filtered_values = np.full_like(roc_values, np.nan)
                # Ensure window length is odd and doesn't exceed valid data
                window_len = min(self.sg_window, valid_count)
                if window_len % 2 == 0:
                    window_len -= 1
                if window_len >= 3:  # Minimum window length for Savitzky-Golay
                    filtered_values[valid_mask] = savgol_filter(
                        roc_values[valid_mask],
                        window_length=window_len,
                        polyorder=min(self.sg_polyorder, window_len - 1)
                    )
                roc = pd.Series(filtered_values, index=roc.index)
        
        return roc
    
    def calculate_acceleration(self, roc: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate acceleration (second derivative = ROC of ROC)
        
        Args:
            roc: Series of ROC values
            period: Lookback period for acceleration (defaults to self.roc_period)
        
        Returns:
            Series of acceleration values
        """
        if period is None:
            period = self.roc_period
        
        # Acceleration is the rate of change of ROC
        acceleration = roc.diff(periods=period)
        
        # Apply Savitzky-Golay filter if enabled
        if self.use_savgol and len(acceleration) > self.sg_window:
            accel_values = acceleration.values
            valid_mask = ~np.isnan(accel_values)
            valid_count = valid_mask.sum()
            if valid_count > self.sg_window:
                filtered_values = np.full_like(accel_values, np.nan)
                # Ensure window length is odd and doesn't exceed valid data
                window_len = min(self.sg_window, valid_count)
                if window_len % 2 == 0:
                    window_len -= 1
                if window_len >= 3:  # Minimum window length for Savitzky-Golay
                    filtered_values[valid_mask] = savgol_filter(
                        accel_values[valid_mask],
                        window_length=window_len,
                        polyorder=min(self.sg_polyorder, window_len - 1)
                    )
                acceleration = pd.Series(filtered_values, index=acceleration.index)
        
        return acceleration
    
    def calculate_all(
        self, 
        df: pd.DataFrame, 
        price_col: str = "close",
        return_raw: bool = False
    ) -> pd.DataFrame:
        """
        Calculate all derivatives and return as DataFrame
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            return_raw: If True, also return raw (unfiltered) derivatives
        
        Returns:
            DataFrame with columns: smoothed_price, roc, acceleration
            If return_raw=True, also includes: roc_raw, acceleration_raw
        """
        result = df.copy()
        
        # Calculate smoothed price
        result['smoothed_price'] = self.calculate_smoothed_price(df, price_col)
        
        # Calculate ROC (first derivative)
        if return_raw:
            # Temporarily disable filter for raw calculation
            original_use_savgol = self.use_savgol
            self.use_savgol = False
            result['roc_raw'] = self.calculate_roc(df[price_col])
            self.use_savgol = original_use_savgol
        
        result['roc'] = self.calculate_roc(result['smoothed_price'])
        
        # Calculate acceleration (second derivative)
        if return_raw:
            original_use_savgol = self.use_savgol
            self.use_savgol = False
            result['acceleration_raw'] = self.calculate_acceleration(result['roc_raw'])
            self.use_savgol = original_use_savgol
        
        result['acceleration'] = self.calculate_acceleration(result['roc'])
        
        return result
