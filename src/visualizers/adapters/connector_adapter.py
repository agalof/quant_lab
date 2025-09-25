"""
Connector Adapter for Research Phase Integration
==============================================

Seamlessly integrates CLOBDataSource with StrategyVisualizer for research workflows.
Provides data preparation, signal generation support, and indicator management.

Features:
- Direct integration with Hummingbot CLOBDataSource
- Real-time data streaming capabilities  
- Signal generation framework integration
- Technical indicator pipeline support
- Research notebook optimization

Author: Quants Lab Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import asyncio
import warnings

from core.data_sources.clob import CLOBDataSource
from core.data_structures.candles import Candles

logger = logging.getLogger(__name__)


class SignalGenerator(ABC):
    """
    Abstract base class for signal generation strategies.
    Implement this interface to create custom signal generation logic.
    """
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
        """
        Generate trading signals from OHLCV data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with indicators
            
        Returns:
        --------
        Tuple[pd.Series, Optional[pd.Series]]
            (signals, signal_types) where:
            - signals: 1=long, 0=neutral/exit, -1=short
            - signal_types: 'entry', 'exit', 'hold' (optional)
        """
        pass


class IndicatorEngine:
    """
    Technical indicator calculation engine with common indicators.
    Extensible framework for adding custom technical indicators.
    """
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = IndicatorEngine.sma(series, period)
        rolling_std = series.rolling(window=period).std()
        return {
            'bb_middle': sma,
            'bb_upper': sma + (rolling_std * std),
            'bb_lower': sma - (rolling_std * std)
        }
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator."""
        ema_fast = IndicatorEngine.ema(series, fast)
        ema_slow = IndicatorEngine.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorEngine.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }


class ConnectorAdapter:
    """
    Adapter class for integrating CLOBDataSource with StrategyVisualizer.
    
    Provides seamless research workflow integration with:
    - Data fetching and preparation
    - Technical indicator calculation
    - Signal generation framework
    - Real-time data streaming
    """
    
    def __init__(self, 
                 clob_source: Optional[CLOBDataSource] = None,
                 cache_data: bool = True,
                 auto_indicators: bool = True):
        """
        Initialize ConnectorAdapter.
        
        Parameters:
        -----------
        clob_source : CLOBDataSource, optional
            Data source instance. Creates new if None.
        cache_data : bool
            Whether to cache fetched data
        auto_indicators : bool
            Automatically add common indicators
        """
        self.clob_source = clob_source or CLOBDataSource()
        self.cache_data = cache_data
        self.auto_indicators = auto_indicators
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._indicator_cache: Dict[str, Dict[str, pd.Series]] = {}
        
        # Indicator engine
        self.indicator_engine = IndicatorEngine()
        
        # Signal generators registry
        self.signal_generators: Dict[str, SignalGenerator] = {}
        
        # Configuration
        self.default_indicators = [
            {'name': 'sma_20', 'func': 'sma', 'params': {'period': 20}},
            {'name': 'sma_50', 'func': 'sma', 'params': {'period': 50}},
            {'name': 'ema_12', 'func': 'ema', 'params': {'period': 12}},
            {'name': 'rsi_14', 'func': 'rsi', 'params': {'period': 14}}
        ]
        
        logger.info("ConnectorAdapter initialized")
    
    def get_available_connectors(self) -> List[str]:
        """Get list of available connector names."""
        return list(self.clob_source.connectors.keys())
    
    async def get_trading_pairs(self, connector_name: str) -> List[str]:
        """Get available trading pairs for a connector."""
        try:
            connector = self.clob_source.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector '{connector_name}' not found")
            
            # Get trading rules to extract pairs
            trading_rules = await self.clob_source.get_trading_rules(connector_name)
            return trading_rules.get_all_trading_pairs()
        
        except Exception as e:
            logger.error(f"Failed to get trading pairs for {connector_name}: {e}")
            return []
    
    async def load_data(self,
                       connector_name: str,
                       trading_pair: str,
                       interval: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load OHLCV data from connector.
        
        Parameters:
        -----------
        connector_name : str
            Name of the connector (e.g., 'binance_perpetual')
        trading_pair : str
            Trading pair (e.g., 'BTC-USDT')
        interval : str
            Time interval ('1m', '5m', '1h', '1d', etc.)
        start_time : datetime, optional
            Start time for data
        end_time : datetime, optional
            End time for data  
        limit : int, optional
            Maximum number of candles
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data with datetime index
        """
        cache_key = f"{connector_name}_{trading_pair}_{interval}"
        
        # Check cache first
        if self.cache_data and cache_key in self._data_cache:
            cached_data = self._data_cache[cache_key]
            logger.info(f"Using cached data for {cache_key}")
            
            # Filter by time range if specified
            if start_time or end_time:
                return self._filter_by_time_range(cached_data, start_time, end_time)
            return cached_data
        
        try:
            # Fetch data from connector
            logger.info(f"Fetching data: {connector_name} {trading_pair} {interval}")
            
            candles = await self.clob_source.get_candles(
                connector_name=connector_name,
                trading_pair=trading_pair,
                interval=interval,
                start_time=int(start_time.timestamp()) if start_time else None,
                end_time=int(end_time.timestamp()) if end_time else None,
                limit=limit
            )
            
            # Convert to DataFrame
            df = self._prepare_dataframe(candles)
            
            # Cache the data
            if self.cache_data:
                self._data_cache[cache_key] = df
                logger.info(f"Cached data for {cache_key}: {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {connector_name}: {e}")
            raise
    
    def _prepare_dataframe(self, candles: Candles) -> pd.DataFrame:
        """Convert Candles object to standardized DataFrame."""
        df = candles.data.copy()
        
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='s')
            df.index.name = 'datetime'
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No timestamp column found, using integer index")
        
        # Ensure standard column names
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
        
        return df.sort_index()
    
    def _filter_by_time_range(self, 
                             df: pd.DataFrame,
                             start_time: Optional[datetime],
                             end_time: Optional[datetime]) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        filtered_df = df.copy()
        
        if start_time:
            filtered_df = filtered_df[filtered_df.index >= start_time]
        
        if end_time:
            filtered_df = filtered_df[filtered_df.index <= end_time]
            
        return filtered_df
    
    def add_indicators(self, 
                      df: pd.DataFrame,
                      indicators: Optional[List[Dict]] = None,
                      use_defaults: bool = True) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
        indicators : list of dict, optional
            Custom indicator configurations
        use_defaults : bool
            Whether to add default indicators
            
        Returns:
        --------
        pd.DataFrame
            Data with added indicators
        """
        df_with_indicators = df.copy()
        
        # Add default indicators
        if use_defaults and self.auto_indicators:
            for indicator_config in self.default_indicators:
                df_with_indicators = self._add_single_indicator(
                    df_with_indicators, indicator_config
                )
        
        # Add custom indicators
        if indicators:
            for indicator_config in indicators:
                df_with_indicators = self._add_single_indicator(
                    df_with_indicators, indicator_config
                )
        
        return df_with_indicators
    
    def _add_single_indicator(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add a single technical indicator."""
        try:
            name = config['name']
            func_name = config['func']
            params = config.get('params', {})
            
            # Get indicator function
            if hasattr(self.indicator_engine, func_name):
                func = getattr(self.indicator_engine, func_name)
            else:
                logger.warning(f"Unknown indicator function: {func_name}")
                return df
            
            # Calculate indicator
            if func_name in ['bollinger_bands', 'macd', 'stochastic']:
                # Multi-output indicators
                if func_name == 'stochastic':
                    result = func(df['high'], df['low'], df['close'], **params)
                else:
                    result = func(df['close'], **params)
                
                # Add all outputs
                for key, values in result.items():
                    df[key] = values
            else:
                # Single-output indicators
                df[name] = func(df['close'], **params)
            
            logger.debug(f"Added indicator: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add indicator {config.get('name', 'unknown')}: {e}")
        
        return df
    
    def register_signal_generator(self, name: str, generator: SignalGenerator) -> None:
        """Register a custom signal generator."""
        self.signal_generators[name] = generator
        logger.info(f"Registered signal generator: {name}")
    
    def generate_signals(self,
                        df: pd.DataFrame,
                        generator_name: str,
                        **kwargs) -> Tuple[pd.Series, Optional[pd.Series]]:
        """
        Generate trading signals using registered generator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with indicators
        generator_name : str
            Name of registered signal generator
        **kwargs : dict
            Additional parameters for signal generator
            
        Returns:
        --------
        Tuple[pd.Series, Optional[pd.Series]]
            (signals, signal_types)
        """
        if generator_name not in self.signal_generators:
            raise ValueError(f"Signal generator '{generator_name}' not registered")
        
        generator = self.signal_generators[generator_name]
        
        try:
            signals, signal_types = generator.generate_signals(df)
            
            # Validate signal format
            self._validate_signals(signals, signal_types)
            
            logger.info(f"Generated signals using {generator_name}")
            return signals, signal_types
            
        except Exception as e:
            logger.error(f"Signal generation failed with {generator_name}: {e}")
            raise
    
    def _validate_signals(self, 
                         signals: pd.Series,
                         signal_types: Optional[pd.Series]) -> None:
        """Validate signal format."""
        # Check signal values
        valid_signals = {-1, 0, 1}
        unique_signals = set(signals.dropna().unique())
        invalid = unique_signals - valid_signals
        
        if invalid:
            warnings.warn(f"Invalid signal values found: {invalid}. Expected: {valid_signals}")
        
        # Check signal types if provided
        if signal_types is not None:
            valid_types = {'entry', 'exit', 'hold'}
            unique_types = set(signal_types.dropna().unique())
            invalid_types = unique_types - valid_types
            
            if invalid_types:
                warnings.warn(f"Invalid signal types found: {invalid_types}. Expected: {valid_types}")
    
    async def create_research_dataset(self,
                                     connector_name: str,
                                     trading_pair: str,
                                     interval: str,
                                     lookback_days: int = 30,
                                     include_indicators: bool = True,
                                     signal_generator: Optional[str] = None) -> Dict[str, Any]:
        """
        Create complete research dataset for strategy development.
        
        Parameters:
        -----------
        connector_name : str
            Connector name
        trading_pair : str
            Trading pair
        interval : str
            Data interval
        lookback_days : int
            Number of days of historical data
        include_indicators : bool
            Whether to add technical indicators
        signal_generator : str, optional
            Name of signal generator to apply
            
        Returns:
        --------
        dict
            Complete research dataset with data, indicators, and signals
        """
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Load base data
        logger.info(f"Creating research dataset: {trading_pair} ({lookback_days} days)")
        df = await self.load_data(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        # Add indicators
        if include_indicators:
            df = self.add_indicators(df)
        
        # Generate signals if requested
        signals = None
        signal_types = None
        if signal_generator:
            try:
                signals, signal_types = self.generate_signals(df, signal_generator)
            except Exception as e:
                logger.warning(f"Signal generation failed: {e}")
        
        dataset = {
            'data': df,
            'metadata': {
                'connector': connector_name,
                'trading_pair': trading_pair,
                'interval': interval,
                'start_time': start_time,
                'end_time': end_time,
                'total_periods': len(df),
                'indicators_included': include_indicators,
                'signal_generator': signal_generator
            }
        }
        
        if signals is not None:
            dataset['signals'] = signals
            
        if signal_types is not None:
            dataset['signal_types'] = signal_types
        
        logger.info(f"Research dataset created: {len(df)} records")
        return dataset
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about loaded data."""
        info = {
            'shape': df.shape,
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            },
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict(),
            'price_stats': {}
        }
        
        # Price statistics
        if 'close' in df.columns:
            close_prices = df['close']
            info['price_stats'] = {
                'min': close_prices.min(),
                'max': close_prices.max(),
                'mean': close_prices.mean(),
                'std': close_prices.std(),
                'volatility_pct': (close_prices.std() / close_prices.mean()) * 100
            }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear data cache."""
        self._data_cache.clear()
        self._indicator_cache.clear()
        logger.info("Data cache cleared")


# Example Signal Generators

class SimpleMovingAverageCrossover(SignalGenerator):
    """
    Simple moving average crossover signal generator.
    Generates signals when fast MA crosses above/below slow MA.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate MA crossover signals."""
        # Calculate MAs if not present
        fast_ma_col = f'sma_{self.fast_period}'
        slow_ma_col = f'sma_{self.slow_period}'
        
        if fast_ma_col not in df.columns:
            df[fast_ma_col] = IndicatorEngine.sma(df['close'], self.fast_period)
        
        if slow_ma_col not in df.columns:
            df[slow_ma_col] = IndicatorEngine.sma(df['close'], self.slow_period)
        
        fast_ma = df[fast_ma_col]
        slow_ma = df[slow_ma_col]
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signal_types = pd.Series('hold', index=df.index)
        
        # Crossover logic
        fast_above_slow = fast_ma > slow_ma
        crossover_up = fast_above_slow & ~fast_above_slow.shift(1)
        crossover_down = ~fast_above_slow & fast_above_slow.shift(1)
        
        # Set signals
        signals[crossover_up] = 1  # Long entry
        signals[crossover_down] = -1  # Short entry
        
        signal_types[crossover_up] = 'entry'
        signal_types[crossover_down] = 'entry'
        
        # Exit signals (opposite crossover)
        exit_long = crossover_down & (signals.shift(1) == 1)
        exit_short = crossover_up & (signals.shift(1) == -1)
        
        signals[exit_long] = 0
        signals[exit_short] = 0
        signal_types[exit_long | exit_short] = 'exit'
        
        return signals, signal_types


class RSIMeanReversion(SignalGenerator):
    """
    RSI-based mean reversion signal generator.
    Long when RSI < oversold, short when RSI > overbought.
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 exit_neutral: float = 50):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_neutral = exit_neutral
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate RSI mean reversion signals."""
        # Calculate RSI if not present
        rsi_col = f'rsi_{self.rsi_period}'
        
        if rsi_col not in df.columns:
            df[rsi_col] = IndicatorEngine.rsi(df['close'], self.rsi_period)
        
        rsi = df[rsi_col]
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signal_types = pd.Series('hold', index=df.index)
        
        # Entry conditions
        long_entry = rsi < self.oversold
        short_entry = rsi > self.overbought
        
        # Exit conditions (RSI returns to neutral)
        exit_condition = (rsi > self.exit_neutral - 5) & (rsi < self.exit_neutral + 5)
        
        # Apply entry signals
        signals[long_entry] = 1
        signals[short_entry] = -1
        signal_types[long_entry | short_entry] = 'entry'
        
        # Apply exit signals  
        signals[exit_condition] = 0
        signal_types[exit_condition] = 'exit'
        
        return signals, signal_types