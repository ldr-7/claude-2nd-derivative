"""
Main Streamlit dashboard application
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional

from src.data_fetcher import DataFetcher
from src.derivatives import DerivativesCalculator
from src.signals import SignalDetector
from src.screener import Screener
from app.charts import create_combined_chart, create_main_chart, create_roc_chart, create_acceleration_chart
from config.settings import (
    DEFAULT_SMOOTHING_PERIOD,
    DEFAULT_SMOOTHING_TYPE,
    DEFAULT_ROC_PERIOD,
    USE_SAVITZKY_GOLAY,
    DEFAULT_TIMEFRAME
)


# Page configuration
st.set_page_config(
    page_title="Druckenmiller Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'derivatives_calc' not in st.session_state:
    st.session_state.derivatives_calc = DerivativesCalculator()
if 'signal_detector' not in st.session_state:
    st.session_state.signal_detector = SignalDetector()
if 'screener' not in st.session_state:
    st.session_state.screener = Screener()


def load_watchlist(file_path: Optional[str] = None) -> list:
    """Load watchlist from file"""
    if file_path is None:
        # Use absolute path relative to project root
        file_path = Path(__file__).parent.parent / "config" / "watchlist.txt"
    else:
        file_path = Path(file_path)
    
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        return []


def main():
    st.title("📈 Second-Derivative Rate-of-Change Trading Dashboard")
    st.markdown("**Inspired by Stanley Druckenmiller's approach to catching rotations early**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker input
    ticker_input = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
    
    # Timeframe selector
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["1d", "1wk", "1mo"],
        index=0 if DEFAULT_TIMEFRAME == "1d" else (1 if DEFAULT_TIMEFRAME == "1wk" else 2)
    )
    
    # Data period
    period = st.sidebar.selectbox(
        "Data Period",
        options=["3mo", "6mo", "1y", "2y"],
        index=1
    )
    
    st.sidebar.divider()
    
    # Parameters
    st.sidebar.subheader("Calculation Parameters")
    
    smoothing_type = st.sidebar.selectbox(
        "Smoothing Type",
        options=["EMA", "SMA"],
        index=0 if DEFAULT_SMOOTHING_TYPE == "EMA" else 1
    )
    
    smoothing_period = st.sidebar.slider(
        "Smoothing Period",
        min_value=5,
        max_value=50,
        value=DEFAULT_SMOOTHING_PERIOD,
        step=1
    )
    
    roc_period = st.sidebar.slider(
        "ROC Period",
        min_value=5,
        max_value=30,
        value=DEFAULT_ROC_PERIOD,
        step=1
    )
    
    use_savgol = st.sidebar.checkbox(
        "Use Savitzky-Golay Filter",
        value=USE_SAVITZKY_GOLAY
    )
    
    # Update calculators with new parameters
    st.session_state.derivatives_calc = DerivativesCalculator(
        smoothing_period=smoothing_period,
        smoothing_type=smoothing_type,
        roc_period=roc_period,
        use_savgol=use_savgol
    )
    
    st.session_state.screener = Screener(timeframe=timeframe)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Single Ticker Analysis", "Watchlist Screener"])
    
    with tab1:
        st.header(f"Analysis: {ticker_input}")
        
        # Fetch and process data
        with st.spinner(f"Fetching data for {ticker_input}..."):
            df = st.session_state.data_fetcher.fetch_data(ticker_input, timeframe, period)
        
        if df is None or df.empty:
            st.error(f"Could not fetch data for {ticker_input}")
        else:
            # Calculate derivatives
            df = st.session_state.derivatives_calc.calculate_all(df)
            
            # Detect signals
            df = st.session_state.signal_detector.analyze(df)
            
            # Display latest values
            col1, col2, col3, col4 = st.columns(4)
            latest = df.iloc[-1]
            
            with col1:
                st.metric("Price", f"${latest['close']:.2f}")
            with col2:
                st.metric("ROC", f"{latest['roc']:.2f}%")
            with col3:
                st.metric("Acceleration", f"{latest['acceleration']:.4f}")
            with col4:
                signal_type = latest.get('signal_type', 'None')
                if pd.notna(signal_type):
                    st.metric("Signal", signal_type.replace('_', ' ').title())
                else:
                    st.metric("Signal", "None")
            
            st.divider()
            
            # Chart options
            chart_col1, chart_col2 = st.columns([3, 1])
            
            with chart_col2:
                show_candlestick = st.checkbox("Candlestick Chart", value=True)
                show_signals = st.checkbox("Show Signals", value=True)
                show_raw = st.checkbox("Show Raw Derivatives", value=False)
            
            # Create and display combined chart
            fig = create_combined_chart(df, show_candlestick=show_candlestick, show_signals=show_signals)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent signals
            st.subheader("Recent Signals")
            recent_signals = st.session_state.signal_detector.get_recent_signals(df)
            
            if recent_signals:
                signals_df = pd.DataFrame(recent_signals)
                st.dataframe(signals_df, use_container_width=True)
                
                # Export button
                csv = signals_df.to_csv(index=False)
                st.download_button(
                    label="Download Signals as CSV",
                    data=csv,
                    file_name=f"{ticker_input}_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No recent signals detected")
            
            # Data table
            with st.expander("View Full Data"):
                display_cols = ['close', 'smoothed_price', 'roc', 'acceleration', 'signal_type']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols].tail(50), use_container_width=True)
    
    with tab2:
        st.header("Watchlist Screener")
        st.markdown("**Scan for tickers with acceleration crossing zero in recent sessions**")
        
        # Load watchlist
        default_watchlist = load_watchlist()
        
        # Watchlist input
        watchlist_text = st.text_area(
            "Watchlist (one ticker per line)",
            value="\n".join(default_watchlist),
            height=150
        )
        
        watchlist = [ticker.strip().upper() for ticker in watchlist_text.split("\n") if ticker.strip()]
        
        lookback_days = st.slider(
            "Signal Lookback Days",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        st.session_state.screener.signal_detector.lookback_days = lookback_days
        
        if st.button("Scan Watchlist", type="primary"):
            if not watchlist:
                st.warning("Please enter at least one ticker in the watchlist")
            else:
                with st.spinner(f"Scanning {len(watchlist)} tickers..."):
                    results_df = st.session_state.screener.scan_watchlist(watchlist, period=period)
                
                if results_df.empty:
                    st.warning("No results found")
                else:
                    # Display results
                    st.subheader("Ranked Results (by Signal Strength)")
                    
                    # Format display columns
                    display_df = results_df.copy()
                    
                    # Helper functions for signal processing
                    def get_latest_signal_type(signals):
                        """Extract the most recent signal type or return NONE"""
                        if not signals or len(signals) == 0:
                            return "NONE"
                        # Find the signal with the latest timestamp
                        if isinstance(signals, list) and len(signals) > 0:
                            latest_signal = None
                            latest_timestamp = None
                            
                            for signal in signals:
                                if isinstance(signal, dict):
                                    timestamp = signal.get('timestamp', None)
                                    if timestamp:
                                        # Convert to comparable timestamp
                                        if isinstance(timestamp, pd.Timestamp):
                                            ts = timestamp
                                        else:
                                            try:
                                                ts = pd.Timestamp(timestamp)
                                            except:
                                                continue
                                        
                                        if latest_timestamp is None or ts > latest_timestamp:
                                            latest_timestamp = ts
                                            latest_signal = signal
                            
                            # If we found a signal with timestamp, use it
                            if latest_signal:
                                signal_type = latest_signal.get('signal_type', None)
                                if signal_type:
                                    return str(signal_type).upper()
                            
                            # Fallback: if no timestamps, use the last signal in the list
                            signal = signals[-1]
                            if isinstance(signal, dict):
                                signal_type = signal.get('signal_type', None)
                                if signal_type:
                                    return str(signal_type).upper()
                        return "NONE"
                    
                    def format_recent_signals(signals):
                        """Convert signal objects to formatted strings"""
                        if not signals or len(signals) == 0:
                            return ""
                        formatted = []
                        for signal in signals:
                            if isinstance(signal, dict):
                                signal_type = signal.get('signal_type', 'UNKNOWN')
                                timestamp = signal.get('timestamp', None)
                                if timestamp:
                                    if isinstance(timestamp, pd.Timestamp):
                                        date_str = timestamp.strftime('%Y-%m-%d')
                                    else:
                                        date_str = str(timestamp)[:10]  # Extract date part
                                    formatted.append(f"{signal_type.upper()} ({date_str})")
                                else:
                                    formatted.append(signal_type.upper())
                        return " | ".join(formatted)
                    
                    # Add signal_type column BEFORE formatting recent_signals
                    if 'recent_signals' in display_df.columns:
                        display_df['signal_type'] = display_df['recent_signals'].apply(get_latest_signal_type)
                        # Format recent_signals column after extracting signal_type
                        display_df['recent_signals'] = display_df['recent_signals'].apply(format_recent_signals)
                    else:
                        # If recent_signals column doesn't exist, set signal_type to NONE
                        display_df['signal_type'] = "NONE"
                    
                    # Format numeric columns
                    if 'price' in display_df.columns:
                        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                    if 'roc' in display_df.columns:
                        display_df['roc'] = display_df['roc'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    if 'acceleration' in display_df.columns:
                        display_df['acceleration'] = display_df['acceleration'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    if 'acceleration_change' in display_df.columns:
                        display_df['acceleration_change'] = display_df['acceleration_change'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    if 'volume_confirmation' in display_df.columns:
                        display_df['volume_confirmation'] = display_df['volume_confirmation'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
                    if 'latest_signal_date' in display_df.columns:
                        display_df['latest_signal_date'] = display_df['latest_signal_date'].apply(
                            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ""
                        )
                    
                    # Reorder columns: ticker | price | signal_type | roc | acceleration | acceleration_change | volume_confirmation | latest_signal_date
                    column_order = ['ticker', 'price', 'signal_type', 'roc', 'acceleration', 'acceleration_change', 'volume_confirmation', 'latest_signal_date']
                    # Add any remaining columns that aren't in the order list
                    remaining_cols = [col for col in display_df.columns if col not in column_order]
                    column_order.extend(remaining_cols)
                    # Filter to only columns that exist
                    column_order = [col for col in column_order if col in display_df.columns]
                    display_df = display_df[column_order]
                    
                    # Color code signal_type column
                    def style_signal_type(val):
                        """Apply color coding to signal_type column"""
                        if val == "POTENTIAL_BOTTOM":
                            return 'background-color: #90EE90; color: #006400; font-weight: bold'
                        elif val == "POTENTIAL_TOP":
                            return 'background-color: #FFB6C1; color: #8B0000; font-weight: bold'
                        elif val == "NONE":
                            return 'background-color: #D3D3D3; color: #696969'
                        return ''
                    
                    # Apply styling
                    styled_df = display_df.style.applymap(
                        style_signal_type,
                        subset=['signal_type']
                    )
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Screener Results as CSV",
                        data=csv,
                        file_name=f"screener_results_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Show top signals summary
                    top_signals = results_df[results_df['just_crossed_zero'] == True].head(5)
                    if not top_signals.empty:
                        st.subheader("🔥 Top Signals (Zero Crossings Today)")
                        for _, row in top_signals.iterrows():
                            signal_type = "Potential Bottom" if row['cross_direction'] == "up" else "Potential Top"
                            st.success(
                                f"**{row['ticker']}**: {signal_type} - "
                                f"ROC: {row['roc']:.2f}%, "
                                f"Acceleration: {row['acceleration']:.4f}"
                            )


if __name__ == "__main__":
    main()
