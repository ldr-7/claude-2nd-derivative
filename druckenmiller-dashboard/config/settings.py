"""
Default configuration parameters for the Druckenmiller Trading Dashboard
"""

# Price smoothing
DEFAULT_SMOOTHING_PERIOD = 21
DEFAULT_SMOOTHING_TYPE = "EMA"  # "EMA" or "SMA"

# First derivative (ROC)
DEFAULT_ROC_PERIOD = 14

# Second derivative (Acceleration)
# Calculated as ROC of ROC

# Savitzky-Golay filter
USE_SAVITZKY_GOLAY = True
SG_WINDOW_LENGTH = 11  # Must be odd
SG_POLYORDER = 3

# Signal detection
SIGNAL_LOOKBACK_DAYS = 3  # Look for signals in last N sessions

# Default timeframe
DEFAULT_TIMEFRAME = "1d"  # Options: "1d", "1wk", "1mo"

# Cache settings
CACHE_DIR = ".cache"
CACHE_EXPIRY_HOURS = 1  # Refresh cache after 1 hour
