"""
Enhanced Strategy Visualization Framework for Hummingbot Quants Lab
================================================================

Professional-grade visualization framework optimized for Hummingbot strategy development.
Integrates seamlessly with CLOBDataSource, backtesting results, and research workflows.

Features:
- Standardized signal format: {1: long, 0: neutral/exit, -1: short}
- Optional signal_type column for entry/exit distinction
- Auto-detection of Hummingbot data structures
- Research and backtesting workflow integration

Author: Quants Lab Development Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StrategyVisualizer:
    """
    Enhanced strategy visualization framework for Hummingbot Quants Lab.
    
    Provides comprehensive charting capabilities optimized for:
    - Hummingbot V2 strategy development
    - Research phase analysis with live data
    - Backtesting result visualization
    - Interactive strategy analysis
    
    Signal Format:
    - 1: Long position
    - 0: Neutral/Exit
    - -1: Short position
    
    Optional signal_type column values:
    - 'entry': New position entry
    - 'exit': Position exit
    - 'hold': Maintain current position
    """
    
    # Standard Hummingbot column names
    STANDARD_COLUMNS = {
        'timestamp': 'timestamp',
        'open': 'open', 
        'high': 'high',
        'low': 'low', 
        'close': 'close',
        'volume': 'volume'
    }
    
    def __init__(self, df: pd.DataFrame, auto_detect: bool = True):
        """
        Initialize the Enhanced StrategyVisualizer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV dataframe with price data
        auto_detect : bool
            Whether to auto-detect column names and datetime index
        """
        self.df = df.copy()
        self.original_df = df.copy()
        
        if auto_detect:
            self._auto_detect_structure()
        else:
            self._validate_standard_structure()
            
        # Chart configuration
        self.subcharts = []
        self.main_indicators = []
        self.signal_config = {}
        self.chart_height = 800
        self.subplot_heights = []
        self.show_volume = True
        
        # Auto-add volume subchart if available
        if self._has_volume():
            self._add_volume_subchart()
        
        # Statistics storage
        self.signal_stats = {}
        
        # Enhanced color scheme for Hummingbot
        self.default_colors = {
            'candle_up': '#00d4aa',      # Hummingbot teal
            'candle_down': '#ff6b9d',    # Hummingbot pink  
            'volume': '#1f77b4',
            'long': '#00ff88',           # Bright green for long
            'short': '#ff4444',          # Bright red for short
            'neutral': '#888888',        # Gray for neutral/exit
            'entry_marker': '#ffffff',   # White border for entries
            'exit_marker': '#000000'     # Black border for exits
        }
        
    def _auto_detect_structure(self) -> None:
        """Auto-detect Hummingbot data structure and datetime handling."""
        # Check for timestamp column vs datetime index
        if 'timestamp' in self.df.columns:
            # Convert timestamp to datetime index if needed
            if not isinstance(self.df.index, pd.DatetimeIndex):
                logger.info("Converting timestamp column to datetime index")
                self.df.index = pd.to_datetime(self.df['timestamp'], unit='s')
                self.df.index.name = 'datetime'
        elif isinstance(self.df.index, pd.DatetimeIndex):
            logger.info("Using existing datetime index")
        else:
            warnings.warn("No timestamp column or datetime index found. Using integer index.")
            
        self._validate_price_columns()
        
    def _validate_standard_structure(self) -> None:
        """Validate that DataFrame has standard Hummingbot structure."""
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required OHLC columns: {missing_cols}")
            
        self._validate_price_columns()
        
    def _validate_price_columns(self) -> None:
        """Validate price column data quality."""
        price_cols = ['open', 'high', 'low', 'close']
        
        # Check for NaN values
        for col in price_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                warnings.warn(f"Found NaN values in {col} column")
                
        # Validate OHLC relationships
        if all(col in self.df.columns for col in price_cols):
            invalid_ohlc = (
                (self.df['high'] < self.df['low']) |
                (self.df['high'] < self.df['open']) |
                (self.df['high'] < self.df['close']) |
                (self.df['low'] > self.df['open']) |
                (self.df['low'] > self.df['close'])
            ).any()
            
            if invalid_ohlc:
                warnings.warn("Found invalid OHLC relationships in data")
                
    def _has_volume(self) -> bool:
        """Check if volume data is available."""
        return 'volume' in self.df.columns and not self.df['volume'].isnull().all()
        
    def _add_volume_subchart(self) -> None:
        """Add volume as a subchart."""
        volume_config = {
            'type': 'volume',
            'title': 'Volume', 
            'column': 'volume'
        }
        self.subcharts.append(volume_config)
        
    def add_volume(self, show: bool = True) -> 'StrategyVisualizer':
        """Control volume display."""
        self.show_volume = show
        
        # Remove existing volume subchart
        self.subcharts = [sub for sub in self.subcharts if sub.get('type') != 'volume']
        
        # Add volume subchart if requested and available
        if show and self._has_volume():
            self._add_volume_subchart()
            
        return self
        
    def add_indicator(self, 
                     column: str,
                     chart_type: str = 'main',
                     name: Optional[str] = None,
                     color: Optional[str] = None,
                     line_style: str = 'solid',
                     width: int = 2,
                     opacity: float = 0.8,
                     fill: bool = False,
                     **kwargs) -> 'StrategyVisualizer':
        """
        Add technical indicator to the chart.
        
        Parameters:
        -----------
        column : str
            Column name containing indicator values
        chart_type : str  
            'main' for price overlay, 'sub' for separate subchart
        name : str, optional
            Display name. If None, uses formatted column name
        color : str, optional
            Line color. Auto-assigned if None
        line_style : str
            'solid', 'dash', 'dot', 'dashdot'
        width : int
            Line width
        opacity : float
            Line opacity (0-1)
        fill : bool
            Fill area under line
        **kwargs : dict
            Additional plotly trace parameters
        """
        if column not in self.df.columns:
            raise ValueError(f"Indicator column '{column}' not found")
            
        indicator_config = {
            'column': column,
            'name': name or self._format_indicator_name(column),
            'color': color,
            'line_style': line_style,
            'width': width,
            'opacity': opacity,
            'fill': fill,
            'kwargs': kwargs
        }
        
        if chart_type == 'main':
            self.main_indicators.append(indicator_config)
        elif chart_type == 'sub':
            self.subcharts.append({
                'type': 'indicator',
                'config': indicator_config,
                'title': indicator_config['name']
            })
        else:
            raise ValueError("chart_type must be 'main' or 'sub'")
            
        return self
        
    def _format_indicator_name(self, column: str) -> str:
        """Format column name for display."""
        # Convert snake_case to Title Case
        return column.replace('_', ' ').title()
        
    def add_signals(self,
                   signal_column: str,
                   signal_type_column: Optional[str] = None,
                   name: str = 'Signals',
                   marker_size: int = 12,
                   custom_colors: Optional[Dict] = None,
                   show_neutral: bool = False) -> 'StrategyVisualizer':
        """
        Add trading signals with enhanced entry/exit distinction.
        
        Parameters:
        -----------
        signal_column : str
            Column with signal values: 1=long, 0=neutral/exit, -1=short
        signal_type_column : str, optional
            Column with signal types: 'entry', 'exit', 'hold'
        name : str
            Display name for signals
        marker_size : int
            Size of signal markers
        custom_colors : dict, optional
            Custom color mapping
        show_neutral : bool
            Whether to show neutral signals
        """
        if signal_column not in self.df.columns:
            raise ValueError(f"Signal column '{signal_column}' not found")
            
        # Validate signal values
        signals = self.df[signal_column].dropna()
        valid_signals = {-1, 0, 1}
        unique_signals = set(signals.unique())
        invalid_signals = unique_signals - valid_signals
        
        if invalid_signals:
            warnings.warn(f"Found unexpected signal values: {invalid_signals}. Expected: {valid_signals}")
        
        # Validate signal_type column if provided
        if signal_type_column:
            if signal_type_column not in self.df.columns:
                raise ValueError(f"Signal type column '{signal_type_column}' not found")
                
            valid_types = {'entry', 'exit', 'hold'}
            unique_types = set(self.df[signal_type_column].dropna().unique())
            invalid_types = unique_types - valid_types
            
            if invalid_types:
                warnings.warn(f"Found unexpected signal types: {invalid_types}. Expected: {valid_types}")
        
        # Set up color mapping
        colors = custom_colors or {
            1: self.default_colors['long'],      # Long position
            -1: self.default_colors['short'],    # Short position  
            0: self.default_colors['neutral']    # Neutral/Exit
        }
        
        self.signal_config = {
            'column': signal_column,
            'signal_type_column': signal_type_column,
            'name': name,
            'marker_size': marker_size,
            'colors': colors,
            'show_neutral': show_neutral
        }
        
        # Log signal statistics
        signal_counts = signals.value_counts().sort_index()
        logger.info(f"Added signals from '{signal_column}': {signal_counts.to_dict()}")
        
        if signal_type_column:
            type_counts = self.df[signal_type_column].value_counts()
            logger.info(f"Signal types: {type_counts.to_dict()}")
        
        return self
        
    def calculate_signal_stats(self, signal_column: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive signal statistics."""
        if signal_column is None:
            signal_column = self.signal_config.get('column')
            
        if not signal_column:
            raise ValueError("No signal column configured")
            
        signals = self.df[signal_column]
        signal_type_column = self.signal_config.get('signal_type_column')
        
        # Basic signal counts
        stats = {
            'total_periods': len(signals),
            'total_signals': len(signals[signals != 0]),
            'long_signals': len(signals[signals == 1]),
            'short_signals': len(signals[signals == -1]),
            'neutral_signals': len(signals[signals == 0]),
            'signal_frequency': len(signals[signals != 0]) / len(signals) * 100,
            'long_ratio': len(signals[signals == 1]) / len(signals[signals != 0]) * 100 if len(signals[signals != 0]) > 0 else 0,
            'short_ratio': len(signals[signals == -1]) / len(signals[signals != 0]) * 100 if len(signals[signals != 0]) > 0 else 0
        }
        
        # Enhanced stats with signal_type column
        if signal_type_column and signal_type_column in self.df.columns:
            signal_types = self.df[signal_type_column]
            
            # Entry/Exit analysis
            entries = len(signal_types[signal_types == 'entry'])
            exits = len(signal_types[signal_types == 'exit'])
            holds = len(signal_types[signal_types == 'hold'])
            
            stats.update({
                'total_entries': entries,
                'total_exits': exits,
                'total_holds': holds,
                'entry_ratio': entries / len(signal_types.dropna()) * 100 if len(signal_types.dropna()) > 0 else 0,
                'exit_ratio': exits / len(signal_types.dropna()) * 100 if len(signal_types.dropna()) > 0 else 0
            })
            
            # Position-specific entry/exit breakdown
            long_entries = len(self.df[(signals == 1) & (signal_types == 'entry')])
            short_entries = len(self.df[(signals == -1) & (signal_types == 'entry')])
            long_exits = len(self.df[(signals == 0) & (signal_types == 'exit') & 
                                   (signals.shift(1) == 1)])  # Previous was long
            short_exits = len(self.df[(signals == 0) & (signal_types == 'exit') & 
                                    (signals.shift(1) == -1)])  # Previous was short
            
            stats.update({
                'long_entries': long_entries,
                'short_entries': short_entries, 
                'long_exits': long_exits,
                'short_exits': short_exits
            })
        
        self.signal_stats = stats
        return stats
        
    def _create_base_chart(self) -> go.Figure:
        """Create base chart with proper subplot configuration."""
        n_subplots = 1 + len(self.subcharts)
        
        # Intelligent height allocation
        if not self.subplot_heights:
            main_height = 0.65 if self.subcharts else 1.0
            remaining_height = 1.0 - main_height
            heights = [main_height]
            
            if self.subcharts:
                volume_charts = sum(1 for sub in self.subcharts if sub['type'] == 'volume')
                indicator_charts = len(self.subcharts) - volume_charts
                
                # Volume gets 20% of remaining, indicators split the rest
                volume_height = 0.2 * remaining_height / volume_charts if volume_charts > 0 else 0
                indicator_height = (remaining_height - volume_charts * volume_height) / indicator_charts if indicator_charts > 0 else 0
                
                for subchart in self.subcharts:
                    if subchart['type'] == 'volume':
                        heights.append(volume_height)
                    else:
                        heights.append(indicator_height)
        else:
            heights = self.subplot_heights
            
        # Create subplot titles
        subplot_titles = ['Price Action & Signals']
        for sub in self.subcharts:
            subtitle = sub.get('title', 'Subchart')
            subplot_titles.append(subtitle)
        
        # Create subplots with enhanced configuration
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            row_heights=heights,
            vertical_spacing=0.03,
            specs=[[{"secondary_y": False}] for _ in range(n_subplots)]
        )
        
        return fig
        
    def _add_candlestick(self, fig: go.Figure) -> None:
        """Add enhanced candlestick chart."""
        candlestick = go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'], 
            low=self.df['low'],
            close=self.df['close'],
            name='OHLC',
            increasing=dict(
                line=dict(color=self.default_colors['candle_up'], width=1),
                fillcolor=self.default_colors['candle_up']
            ),
            decreasing=dict(
                line=dict(color=self.default_colors['candle_down'], width=1),
                fillcolor=self.default_colors['candle_down']
            ),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Open: %{open:.4f}<br>'
                'High: %{high:.4f}<br>'
                'Low: %{low:.4f}<br>'
                'Close: %{close:.4f}<br>'
                '<extra></extra>'
            )
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
    def _add_main_indicators(self, fig: go.Figure) -> None:
        """Add main chart indicators with enhanced styling."""
        for i, indicator in enumerate(self.main_indicators):
            # Auto-assign colors if not specified
            color = indicator['color'] or px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            
            trace = go.Scatter(
                x=self.df.index,
                y=self.df[indicator['column']],
                mode='lines',
                name=indicator['name'],
                line=dict(
                    color=color,
                    width=indicator['width'],
                    dash=indicator['line_style']
                ),
                opacity=indicator['opacity'],
                fill='tonexty' if indicator['fill'] else None,
                hovertemplate=f'<b>%{{x}}</b><br>{indicator["name"]}: %{{y:.4f}}<extra></extra>',
                **indicator['kwargs']
            )
            fig.add_trace(trace, row=1, col=1)
            
    def _add_subchart_indicators(self, fig: go.Figure) -> None:
        """Add subchart indicators and volume."""
        for i, subchart in enumerate(self.subcharts):
            row_idx = i + 2
            
            if subchart['type'] == 'indicator':
                config = subchart['config']
                trace = go.Scatter(
                    x=self.df.index,
                    y=self.df[config['column']],
                    mode='lines',
                    name=config['name'],
                    line=dict(
                        color=config['color'],
                        width=config['width'],
                        dash=config['line_style']
                    ),
                    opacity=config['opacity'],
                    fill='tonexty' if config['fill'] else None,
                    hovertemplate=f'<b>%{{x}}</b><br>{config["name"]}: %{{y:.4f}}<extra></extra>',
                    **config['kwargs']
                )
                fig.add_trace(trace, row=row_idx, col=1)
                
            elif subchart['type'] == 'volume':
                # Enhanced volume visualization
                volume_trace = go.Bar(
                    x=self.df.index,
                    y=self.df[subchart['column']],
                    name='Volume',
                    marker_color=self.default_colors['volume'],
                    opacity=0.6,
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
                )
                fig.add_trace(volume_trace, row=row_idx, col=1)
                
    def _add_enhanced_signals(self, fig: go.Figure) -> None:
        """Add enhanced signal markers with entry/exit distinction."""
        if not self.signal_config:
            return
            
        try:
            signals = self.df[self.signal_config['column']]
            prices = self.df['close']
            signal_type_column = self.signal_config.get('signal_type_column')
            
            # Define enhanced signal visualization
            signal_configs = {
                1: {  # Long
                    'name': 'Long Position',
                    'color': self.signal_config['colors'].get(1, self.default_colors['long']),
                    'symbol': 'triangle-up'
                },
                -1: {  # Short
                    'name': 'Short Position', 
                    'color': self.signal_config['colors'].get(-1, self.default_colors['short']),
                    'symbol': 'triangle-down'
                },
                0: {  # Neutral/Exit
                    'name': 'Neutral/Exit',
                    'color': self.signal_config['colors'].get(0, self.default_colors['neutral']),
                    'symbol': 'circle'
                }
            }
            
            # Add signals with enhanced entry/exit distinction
            for signal_value, config in signal_configs.items():
                if signal_value == 0 and not self.signal_config['show_neutral']:
                    continue
                    
                mask = signals == signal_value
                if not mask.any():
                    continue
                
                # Base signal trace
                signal_dates = self.df.index[mask]
                signal_prices = prices[mask]
                
                # Enhance with signal_type information if available
                if signal_type_column:
                    signal_types = self.df[signal_type_column][mask]
                    
                    # Separate entry and exit markers
                    for stype in ['entry', 'exit']:
                        type_mask = signal_types == stype
                        if not type_mask.any():
                            continue
                            
                        border_color = self.default_colors['entry_marker'] if stype == 'entry' else self.default_colors['exit_marker']
                        border_width = 2 if stype == 'entry' else 1
                        
                        trace = go.Scatter(
                            x=signal_dates[type_mask],
                            y=signal_prices[type_mask],
                            mode='markers',
                            name=f'{config["name"]} ({stype.title()})',
                            marker=dict(
                                symbol=config['symbol'],
                                size=self.signal_config['marker_size'],
                                color=config['color'],
                                line=dict(width=border_width, color=border_color)
                            ),
                            hovertemplate=f'<b>%{{x}}</b><br>{config["name"]} {stype.title()}<br>Price: %{{y:.4f}}<extra></extra>'
                        )
                        fig.add_trace(trace, row=1, col=1)
                else:
                    # Simple signal markers without entry/exit distinction
                    trace = go.Scatter(
                        x=signal_dates,
                        y=signal_prices,
                        mode='markers',
                        name=config['name'],
                        marker=dict(
                            symbol=config['symbol'],
                            size=self.signal_config['marker_size'],
                            color=config['color'],
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>%{{x}}</b><br>{config["name"]}<br>Price: %{{y:.4f}}<extra></extra>'
                    )
                    fig.add_trace(trace, row=1, col=1)
                    
        except Exception as e:
            logger.error(f"Error adding signal markers: {e}")
            warnings.warn("Failed to add signals to chart")
            
    def _style_chart(self, fig: go.Figure, theme: str = 'white') -> None:
        """Apply enhanced styling optimized for strategy analysis."""
        fig.update_layout(
            title=dict(
                text='Strategy Analysis Dashboard',
                font=dict(size=18, color='#2E86AB'),
                x=0.5
            ),
            height=self.chart_height,
            showlegend=True,
            hovermode='x unified',
            template=f'plotly_{theme}',
            xaxis_rangeslider_visible=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=1.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(r=120)  # Extra space for legend
        )
        
        # Enhanced x-axes styling
        fig.update_xaxes(
            title_text="Time",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showspikes=True,
            spikecolor="rgba(0,0,0,0.5)",
            spikethickness=1
        )
        
        # Enhanced y-axes styling
        fig.update_yaxes(
            title_text="Price",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showspikes=True,
            spikecolor="rgba(0,0,0,0.5)",
            spikethickness=1,
            row=1, col=1
        )
        
        # Style subchart axes
        for i, subchart in enumerate(self.subcharts):
            fig.update_yaxes(
                title_text=subchart.get('title', 'Value'),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                row=i+2, col=1
            )
            
    def plot(self,
             height: int = 800,
             show_stats: bool = True,
             theme: str = 'white',
             title: Optional[str] = None,
             return_fig: bool = False) -> Optional[go.Figure]:
        """
        Generate comprehensive strategy visualization.
        
        Parameters:
        -----------
        height : int
            Chart height in pixels
        show_stats : bool
            Display signal statistics
        theme : str
            Chart theme ('white', 'dark', or plotly theme)
        title : str, optional
            Custom chart title
        return_fig : bool
            Return figure object instead of displaying
            
        Returns:
        --------
        go.Figure or None
        """
        self.chart_height = height
        
        # Create and populate chart
        fig = self._create_base_chart()
        self._add_candlestick(fig)
        self._add_main_indicators(fig)
        self._add_subchart_indicators(fig)
        self._add_enhanced_signals(fig)
        self._style_chart(fig, theme)
        
        # Custom title
        if title:
            fig.update_layout(title=dict(text=title, x=0.5))
        
        # Add statistics
        if show_stats and self.signal_config:
            try:
                stats = self.calculate_signal_stats()
                self._add_stats_annotation(fig, stats)
            except Exception as e:
                logger.warning(f"Failed to calculate signal statistics: {e}")
        
        if return_fig:
            return fig
        else:
            fig.show()
            
    def _add_stats_annotation(self, fig: go.Figure, stats: Dict[str, Any]) -> None:
        """Add enhanced statistics annotation."""
        # Build statistics text
        stats_lines = [
            "<b>Signal Statistics</b>",
            f"Total Signals: {stats['total_signals']:,}",
            f"Long: {stats['long_signals']:,} ({stats['long_ratio']:.1f}%)",
            f"Short: {stats['short_signals']:,} ({stats['short_ratio']:.1f}%)",
            f"Frequency: {stats['signal_frequency']:.1f}%"
        ]
        
        # Add entry/exit stats if available
        if 'total_entries' in stats:
            stats_lines.extend([
                "",
                f"Entries: {stats['total_entries']:,}",
                f"Exits: {stats['total_exits']:,}",
                f"Long Entries: {stats.get('long_entries', 0):,}",
                f"Short Entries: {stats.get('short_entries', 0):,}"
            ])
        
        stats_text = "<br>".join(stats_lines)
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(46, 134, 171, 0.9)",  # Hummingbot blue
            bordercolor="white",
            borderwidth=2,
            font=dict(color="white", size=11),
            align="left"
        )
        
    @classmethod  
    def from_research_data(cls,
                          candles_df: pd.DataFrame,
                          signals: Optional[pd.Series] = None,
                          signal_types: Optional[pd.Series] = None,
                          indicators: Optional[Dict[str, pd.Series]] = None) -> 'StrategyVisualizer':
        """
        Factory method for creating visualizer from research data.
        
        Parameters:
        -----------
        candles_df : pd.DataFrame
            OHLCV candlestick data
        signals : pd.Series, optional
            Signal values (1, 0, -1)
        signal_types : pd.Series, optional
            Signal types ('entry', 'exit', 'hold')
        indicators : dict, optional
            Dictionary of indicator name -> values
            
        Returns:
        --------
        StrategyVisualizer instance
        """
        df = candles_df.copy()
        
        # Add signals
        if signals is not None:
            df['signals'] = signals
            
        if signal_types is not None:
            df['signal_types'] = signal_types
            
        # Add indicators
        if indicators:
            for name, values in indicators.items():
                df[name] = values
                
        visualizer = cls(df)
        
        # Configure signals if provided
        if signals is not None:
            signal_type_col = 'signal_types' if signal_types is not None else None
            visualizer.add_signals('signals', signal_type_column=signal_type_col)
            
        return visualizer
        
    def get_signal_summary(self) -> pd.DataFrame:
        """Get detailed signal summary DataFrame."""
        if not self.signal_config:
            raise ValueError("No signals configured")
            
        signal_col = self.signal_config['column']
        signal_type_col = self.signal_config.get('signal_type_column')
        
        # Base signal data
        signals = self.df[signal_col]
        mask = signals != 0 if not self.signal_config['show_neutral'] else signals.notna()
        
        if not mask.any():
            return pd.DataFrame()
            
        summary_data = {
            'datetime': self.df.index[mask],
            'signal': signals[mask], 
            'price': self.df['close'][mask]
        }
        
        # Add signal type if available
        if signal_type_col and signal_type_col in self.df.columns:
            summary_data['signal_type'] = self.df[signal_type_col][mask]
            
        summary_df = pd.DataFrame(summary_data)
        
        # Add descriptive signal names
        signal_names = {1: 'Long', -1: 'Short', 0: 'Neutral/Exit'}
        summary_df['signal_name'] = summary_df['signal'].map(signal_names)
        
        return summary_df.reset_index(drop=True)
        
    def reset(self) -> 'StrategyVisualizer':
        """Reset visualizer configuration."""
        self.subcharts = []
        self.main_indicators = []
        self.signal_config = {}
        self.signal_stats = {}
        
        # Re-add volume if it was there initially
        if self.show_volume and self._has_volume():
            self._add_volume_subchart()
            
        return self