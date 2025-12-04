"""
Main Streamlit dashboard application
"""

import sys
from pathlib import Path

# Add the project root to the path so imports work on Streamlit Cloud
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

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
                signal_type = latest.get('signal_type', 'NEUTRAL')
                if pd.notna(signal_type) and signal_type != 'NEUTRAL':
                    st.metric("Signal", signal_type.replace('_', ' ').title())
                else:
                    st.metric("Signal", "NEUTRAL")
            
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
        
        # Store results in session state
        if 'screener_results' not in st.session_state:
            st.session_state.screener_results = None
        
        if st.button("Scan Watchlist", type="primary"):
            if not watchlist:
                st.warning("Please enter at least one ticker in the watchlist")
            else:
                with st.spinner(f"Scanning {len(watchlist)} tickers..."):
                    results_df = st.session_state.screener.scan_watchlist(watchlist, period=period)
                    st.session_state.screener_results = results_df
        
        # Display results if available
        if st.session_state.screener_results is not None and not st.session_state.screener_results.empty:
            results_df = st.session_state.screener_results.copy()
            
            # Filter section: "Just Crossed" Filter
            st.subheader("Filters")
            filter_col1, filter_col2 = st.columns([1, 1])
            
            with filter_col1:
                show_only_crosses = st.checkbox("Show Only Recent Zero Crosses", value=False)
                cross_lookback_days = st.slider(
                    "Lookback Days",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    key="cross_lookback"
                )
            
            with filter_col2:
                # Sector filter
                if 'sector' in results_df.columns:
                    available_sectors = sorted(results_df['sector'].dropna().unique())
                    selected_sectors = st.multiselect(
                        "Filter by Sector",
                        options=available_sectors,
                        default=[],
                        key="sector_filter"
                    )
                else:
                    selected_sectors = []
            
            # Apply "Just Crossed" filter
            if show_only_crosses:
                # Filter to show only tickers where zero cross happened within lookback days
                filtered_df = results_df[
                    (results_df['days_since_cross'].notna()) & 
                    (results_df['days_since_cross'] <= cross_lookback_days)
                ].copy()
            else:
                filtered_df = results_df.copy()
            
            # Apply sector filter
            if selected_sectors:
                filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)].copy()
            
            # Sorting section
            st.subheader("Sorting")
            sort_options = {
                'Signal Strength': 'signal_strength',
                'Acceleration Change (magnitude)': 'acceleration_change',
                'Most Recent Cross': 'days_since_cross',
                'ROC (absolute value)': 'roc'
            }
            
            sort_by = st.selectbox(
                "Sort By",
                options=list(sort_options.keys()),
                index=0,  # Default to Signal Strength
                key="sort_by"
            )
            
            sort_column = sort_options[sort_by]
            
            # Apply sorting
            if sort_column == 'days_since_cross':
                # For days_since_cross, sort ascending (most recent first), but handle nulls
                filtered_df = filtered_df.sort_values(
                    sort_column,
                    ascending=True,
                    na_position='last'
                )
            elif sort_column == 'acceleration_change':
                # For acceleration_change (magnitude), sort by absolute value descending
                filtered_df['accel_change_abs'] = filtered_df['acceleration_change'].abs()
                filtered_df = filtered_df.sort_values('accel_change_abs', ascending=False, na_position='last')
                filtered_df = filtered_df.drop('accel_change_abs', axis=1)
            elif sort_column == 'roc':
                # For ROC, sort by absolute value descending
                filtered_df['roc_abs'] = filtered_df['roc'].abs()
                filtered_df = filtered_df.sort_values('roc_abs', ascending=False, na_position='last')
                filtered_df = filtered_df.drop('roc_abs', axis=1)
            else:
                # Default: descending (highest values first)
                filtered_df = filtered_df.sort_values(sort_column, ascending=False, na_position='last')
            
            # Display summary
            total_count = len(results_df)
            filtered_count = len(filtered_df)
            
            if show_only_crosses:
                st.info(f"Showing {filtered_count} tickers with zero crosses in last {cross_lookback_days} days (out of {total_count} total)")
            else:
                st.info(f"Showing {filtered_count} tickers (out of {total_count} total)")
            
            # Format display columns
            display_df = filtered_df.copy()
            
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
            
            # signal_type is already in the dataframe from screener (current state)
            # Format it if needed
            if 'signal_type' not in display_df.columns:
                display_df['signal_type'] = "NEUTRAL"
            else:
                # Ensure signal_type is uppercase
                display_df['signal_type'] = display_df['signal_type'].apply(
                    lambda x: str(x).upper() if pd.notna(x) else "NEUTRAL"
                )
            
            # Format recent_signals column if it exists
            if 'recent_signals' in display_df.columns:
                display_df['recent_signals'] = display_df['recent_signals'].apply(format_recent_signals)
            
            # Format numeric columns (keep raw values for sorting, but create formatted versions for display)
            display_df_formatted = display_df.copy()
            
            if 'price' in display_df_formatted.columns:
                display_df_formatted['price'] = display_df_formatted['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            if 'roc' in display_df_formatted.columns:
                display_df_formatted['roc'] = display_df_formatted['roc'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            if 'acceleration' in display_df_formatted.columns:
                display_df_formatted['acceleration'] = display_df_formatted['acceleration'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            if 'acceleration_change' in display_df_formatted.columns:
                display_df_formatted['acceleration_change'] = display_df_formatted['acceleration_change'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            if 'volume_confirmation' in display_df_formatted.columns:
                display_df_formatted['volume_confirmation'] = display_df_formatted['volume_confirmation'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
            if 'days_since_cross' in display_df_formatted.columns:
                display_df_formatted['days_since_cross'] = display_df_formatted['days_since_cross'].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else ""
                )
            if 'latest_signal_date' in display_df_formatted.columns:
                display_df_formatted['latest_signal_date'] = display_df_formatted['latest_signal_date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ""
                )
            
            # Reorder columns: ticker | sector | price | signal_type | roc | acceleration | acceleration_change | days_since_cross | volume_confirmation | signal_strength
            column_order = ['ticker', 'sector', 'price', 'signal_type', 'roc', 'acceleration', 'acceleration_change', 'days_since_cross', 'volume_confirmation', 'signal_strength']
            # Add any remaining columns that aren't in the order list
            remaining_cols = [col for col in display_df_formatted.columns if col not in column_order]
            column_order.extend(remaining_cols)
            # Filter to only columns that exist
            column_order = [col for col in column_order if col in display_df_formatted.columns]
            display_df_formatted = display_df_formatted[column_order]
            
            # Color code signal_type column based on quadrant logic
            def style_signal_type(val):
                """Apply color coding to signal_type column"""
                val_str = str(val).upper()
                if val_str == "POTENTIAL_BOTTOM":
                    # Green (buy signal)
                    return 'background-color: #90EE90; color: #006400; font-weight: bold'
                elif val_str == "ACCELERATING_UP":
                    # Light green (bullish continuation)
                    return 'background-color: #C8E6C9; color: #2E7D32; font-weight: bold'
                elif val_str == "POTENTIAL_TOP":
                    # Red (sell signal)
                    return 'background-color: #FFB6C1; color: #8B0000; font-weight: bold'
                elif val_str == "ACCELERATING_DOWN":
                    # Light red/orange (bearish continuation)
                    return 'background-color: #FFCCBC; color: #BF360C; font-weight: bold'
                elif val_str == "NEUTRAL":
                    # Gray
                    return 'background-color: #D3D3D3; color: #696969'
                return ''
            
            # Apply styling
            styled_df = display_df_formatted.style.applymap(
                style_signal_type,
                subset=['signal_type']
            )
            
            st.subheader("Results")
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export button (export filtered results)
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Screener Results as CSV",
                data=csv,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Show top signals summary
            top_signals = filtered_df[filtered_df['just_crossed_zero'] == True].head(5)
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
