"""
Plotly chart components for the dashboard
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


def create_main_chart(
    df: pd.DataFrame,
    show_candlestick: bool = True,
    show_smoothed: bool = True,
    show_signals: bool = True
) -> go.Figure:
    """
    Create main price chart with candlesticks and signals
    
    Args:
        df: DataFrame with OHLCV data and signals
        show_candlestick: Whether to show candlestick chart
        show_smoothed: Whether to overlay smoothed price line
        show_signals: Whether to show signal markers
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Add candlestick chart
    if show_candlestick and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
    else:
        # Fallback to line chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
    
    # Add smoothed price line
    if show_smoothed and 'smoothed_price' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['smoothed_price'],
            mode='lines',
            name='Smoothed Price',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    # Add signal markers
    if show_signals and 'signal_type' in df.columns:
        signal_df = df[df['signal_type'].notna()].copy()
        
        # POTENTIAL_BOTTOM (green triangle up)
        bottoms = signal_df[signal_df['signal_type'].str.upper() == 'POTENTIAL_BOTTOM']
        if not bottoms.empty:
            fig.add_trace(go.Scatter(
                x=bottoms.index,
                y=bottoms['close'],
                mode='markers',
                name='Potential Bottom',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                )
            ))
        
        # POTENTIAL_TOP (red triangle down)
        tops = signal_df[signal_df['signal_type'].str.upper() == 'POTENTIAL_TOP']
        if not tops.empty:
            fig.add_trace(go.Scatter(
                x=tops.index,
                y=tops['close'],
                mode='markers',
                name='Potential Top',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                )
            ))
        
        # ACCELERATING_UP (light green circle)
        accel_up = signal_df[signal_df['signal_type'].str.upper() == 'ACCELERATING_UP']
        if not accel_up.empty:
            fig.add_trace(go.Scatter(
                x=accel_up.index,
                y=accel_up['close'],
                mode='markers',
                name='Accelerating Up',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='lightgreen',
                    line=dict(width=1, color='green')
                )
            ))
        
        # ACCELERATING_DOWN (light red/orange circle)
        accel_down = signal_df[signal_df['signal_type'].str.upper() == 'ACCELERATING_DOWN']
        if not accel_down.empty:
            fig.add_trace(go.Scatter(
                x=accel_down.index,
                y=accel_down['close'],
                mode='markers',
                name='Accelerating Down',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='#FF8C69',
                    line=dict(width=1, color='red')
                )
            ))
    
    # Add vertical lines for zero crossings
    if 'zero_crossing' in df.columns:
        crossings = df[df['zero_crossing'] == True]
        for idx in crossings.index:
            fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text="Zero Cross"
            )
    
    fig.update_layout(
        title='Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_roc_chart(df: pd.DataFrame, show_zero_line: bool = True) -> go.Figure:
    """
    Create ROC (first derivative) chart
    
    Args:
        df: DataFrame with ROC column
        show_zero_line: Whether to highlight zero line
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if 'roc' not in df.columns:
        return fig
    
    # Add ROC line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['roc'],
        mode='lines',
        name='ROC',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add zero line
    if show_zero_line:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            opacity=0.5,
            annotation_text="Zero"
        )
    
    # Add vertical lines for zero crossings
    if 'zero_crossing' in df.columns:
        crossings = df[df['zero_crossing'] == True]
        for idx in crossings.index:
            fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5
            )
    
    fig.update_layout(
        title='Rate of Change (First Derivative)',
        xaxis_title='Date',
        yaxis_title='ROC (%)',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_acceleration_chart(
    df: pd.DataFrame,
    show_zero_line: bool = True,
    show_shading: bool = True
) -> go.Figure:
    """
    Create acceleration (second derivative) chart
    
    Args:
        df: DataFrame with acceleration column
        show_zero_line: Whether to highlight zero line
        show_shading: Whether to shade positive/negative regions
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if 'acceleration' not in df.columns:
        return fig
    
    acceleration = df['acceleration']
    
    # Add acceleration line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=acceleration,
        mode='lines',
        name='Acceleration',
        line=dict(color='purple', width=2),
        fill='tozeroy' if show_shading else None,
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))
    
    # Add shading for positive/negative regions
    if show_shading:
        # Positive region (green)
        positive_mask = acceleration >= 0
        if positive_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[positive_mask],
                y=acceleration[positive_mask],
                mode='lines',
                name='Positive',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=False
            ))
        
        # Negative region (red)
        negative_mask = acceleration < 0
        if negative_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[negative_mask],
                y=acceleration[negative_mask],
                mode='lines',
                name='Negative',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            ))
    
    # Add zero line
    if show_zero_line:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            opacity=0.5,
            annotation_text="Zero"
        )
    
    # Add vertical lines for zero crossings
    if 'zero_crossing' in df.columns:
        crossings = df[df['zero_crossing'] == True]
        for idx in crossings.index:
            fig.add_vline(
                x=idx,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text="Zero Cross"
            )
    
    fig.update_layout(
        title='Acceleration (Second Derivative)',
        xaxis_title='Date',
        yaxis_title='Acceleration',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_combined_chart(
    df: pd.DataFrame,
    show_candlestick: bool = True,
    show_signals: bool = True
) -> go.Figure:
    """
    Create combined chart with price, ROC, and acceleration panels
    
    Args:
        df: DataFrame with all required columns
        show_candlestick: Whether to show candlestick chart
        show_signals: Whether to show signal markers
    
    Returns:
        Plotly Figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price', 'Rate of Change (ROC)', 'Acceleration')
    )
    
    # Price panel
    if show_candlestick and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1)
    
    if 'smoothed_price' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['smoothed_price'],
            mode='lines',
            name='Smoothed',
            line=dict(color='orange', width=2, dash='dash')
        ), row=1, col=1)
    
    # Signal markers on price chart
    if show_signals and 'signal_type' in df.columns:
        signal_df = df[df['signal_type'].notna()].copy()
        
        # POTENTIAL_BOTTOM (green triangle up)
        bottoms = signal_df[signal_df['signal_type'].str.upper() == 'POTENTIAL_BOTTOM']
        if not bottoms.empty:
            fig.add_trace(go.Scatter(
                x=bottoms.index,
                y=bottoms['close'],
                mode='markers',
                name='Bottom',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                showlegend=False
            ), row=1, col=1)
        
        # POTENTIAL_TOP (red triangle down)
        tops = signal_df[signal_df['signal_type'].str.upper() == 'POTENTIAL_TOP']
        if not tops.empty:
            fig.add_trace(go.Scatter(
                x=tops.index,
                y=tops['close'],
                mode='markers',
                name='Top',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                showlegend=False
            ), row=1, col=1)
        
        # ACCELERATING_UP (light green circle)
        accel_up = signal_df[signal_df['signal_type'].str.upper() == 'ACCELERATING_UP']
        if not accel_up.empty:
            fig.add_trace(go.Scatter(
                x=accel_up.index,
                y=accel_up['close'],
                mode='markers',
                name='Accel Up',
                marker=dict(symbol='circle', size=8, color='lightgreen'),
                showlegend=False
            ), row=1, col=1)
        
        # ACCELERATING_DOWN (light red/orange circle)
        accel_down = signal_df[signal_df['signal_type'].str.upper() == 'ACCELERATING_DOWN']
        if not accel_down.empty:
            fig.add_trace(go.Scatter(
                x=accel_down.index,
                y=accel_down['close'],
                mode='markers',
                name='Accel Down',
                marker=dict(symbol='circle', size=8, color='#FF8C69'),
                showlegend=False
            ), row=1, col=1)
    
    # ROC panel
    if 'roc' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['roc'],
            mode='lines',
            name='ROC',
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # Acceleration panel
    if 'acceleration' in df.columns:
        acceleration = df['acceleration']
        fig.add_trace(go.Scatter(
            x=df.index,
            y=acceleration,
            mode='lines',
            name='Acceleration',
            line=dict(color='purple', width=2),
            showlegend=False
        ), row=3, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=3, col=1)
        
        # Shading
        positive_mask = acceleration >= 0
        negative_mask = acceleration < 0
        
        if positive_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[positive_mask],
                y=acceleration[positive_mask],
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=False
            ), row=3, col=1)
        
        if negative_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[negative_mask],
                y=acceleration[negative_mask],
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            ), row=3, col=1)
    
    # Add vertical lines for zero crossings across all panels
    if 'zero_crossing' in df.columns:
        crossings = df[df['zero_crossing'] == True]
        for idx in crossings.index:
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=idx,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.3,
                    row=row,
                    col=1
                )
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ROC (%)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration", row=3, col=1)
    
    return fig
