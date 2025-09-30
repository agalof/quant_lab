"""
EWMAC Strategy Controller
File: ewmac_controller.py

Controller implementation for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.

This controller implements a directional trading strategy based on EMA crossovers:
- Calculates fast and slow EMAs from price data
- Detects crossover signals
- Creates long/short positions based on crossovers
- Exits positions on opposite crossover signals

Author: Quants Lab Team
Date: 2025-09-29
"""

from decimal import Decimal
from typing import List, Dict, Any
import logging

import pandas as pd
import pandas_ta as ta
from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, CreateExecutorAction, StopExecutorAction

from ewmac_config import EWMACControllerConfig


logger = logging.getLogger(__name__)


class EWMACController(DirectionalTradingControllerBase):
    """
    EWMAC (Exponentially Weighted Moving Average Crossover) Controller.
    
    This controller implements a trend-following strategy that:
    1. Calculates two EMAs (fast and slow) from price data
    2. Generates signals when EMAs cross:
       - Long signal (1): Fast EMA crosses above Slow EMA
       - Short signal (-1): Fast EMA crosses below Slow EMA
       - No signal (0): No crossover detected
    3. Opens positions when signals are generated
    4. Closes positions when opposite signals occur
    
    Signal Logic:
    - Long Entry: prev_fast <= prev_slow AND curr_fast > curr_slow
    - Short Entry: prev_fast >= prev_slow AND curr_fast < curr_slow
    - Exit: Opposite signal is generated
    
    Position Sizing:
    - Uses fixed quote amount per position
    - Calculates base amount dynamically based on current price
    
    Risk Management:
    - No stop-loss or take-profit (signal-based exits only)
    - Cooldown period prevents rapid position changes
    - Max executors per side limits exposure
    """

    def __init__(self, config: EWMACControllerConfig, *args, **kwargs):
        """
        Initialize the EWMAC controller.
        
        Args:
            config: EWMACControllerConfig instance with strategy parameters
            *args: Additional positional arguments for base class
            **kwargs: Additional keyword arguments for base class
        """
        self.config = config
        
        # Calculate required historical data length
        # Need at least 3x slow period for reliable EMA calculation
        # Plus buffer for crossover detection
        self.max_records = max(self.config.slow_ema_period * 3, 100)
        
        # Initialize candles configuration if not provided
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.candles_trading_pair,
                interval=self.config.interval,
                max_records=self.max_records
            )]
        
        super().__init__(config, *args, **kwargs)
        
        # Initialize state tracking
        self.previous_signal = 0
        self.last_fast_ema = None
        self.last_slow_ema = None
        
        logger.info(f"EWMAC Controller initialized: {self.config.fast_ema_period}/{self.config.slow_ema_period}")

    async def update_processed_data(self):
        """
        Update processed data with current market signals.
        
        This method is called on each tick to:
        1. Fetch latest OHLCV candles data
        2. Calculate fast and slow EMAs
        3. Detect crossover signals
        4. Calculate signal strength
        5. Store processed data for action generation
        
        The processed_data dictionary contains:
        - signal: Current signal value (-1, 0, or 1)
        - signal_strength: Percentage difference between EMAs
        - current_fast_ema: Current fast EMA value
        - current_slow_ema: Current slow EMA value
        - insufficient_data: Boolean indicating if we have enough data
        - timestamp: Current timestamp
        """
        # Get candles data
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        
        # Check if we have sufficient data
        if df is None or len(df) < self.config.slow_ema_period:
            self.processed_data = {
                "signal": 0,
                "signal_strength": 0.0,
                "current_fast_ema": None,
                "current_slow_ema": None,
                "insufficient_data": True,
                "timestamp": self.market_data_provider.time()
            }
            logger.warning(f"Insufficient data: need {self.config.slow_ema_period}, have {len(df) if df is not None else 0}")
            return
        
        # Calculate EMAs using pandas_ta
        df['fast_ema'] = ta.ema(df['close'], length=self.config.fast_ema_period)
        df['slow_ema'] = ta.ema(df['close'], length=self.config.slow_ema_period)
        
        # Drop NaN values from EMA calculation
        df = df.dropna()
        
        if len(df) < 2:
            self.processed_data = {
                "signal": 0,
                "signal_strength": 0.0,
                "current_fast_ema": None,
                "current_slow_ema": None,
                "insufficient_data": True,
                "timestamp": self.market_data_provider.time()
            }
            logger.warning("Insufficient data after EMA calculation")
            return
        
        # Detect crossover signal
        signal = self._detect_crossover(df, 'fast_ema', 'slow_ema')
        
        # Get current EMA values
        current_fast_ema = float(df['fast_ema'].iloc[-1])
        current_slow_ema = float(df['slow_ema'].iloc[-1])
        
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(current_fast_ema, current_slow_ema)
        
        # Update processed data
        self.processed_data = {
            "signal": signal,
            "signal_strength": signal_strength,
            "current_fast_ema": current_fast_ema,
            "current_slow_ema": current_slow_ema,
            "current_price": float(df['close'].iloc[-1]),
            "insufficient_data": False,
            "timestamp": self.market_data_provider.time(),
            "df": df  # Store for potential further analysis
        }
        
        # Update state tracking
        self.previous_signal = signal
        self.last_fast_ema = current_fast_ema
        self.last_slow_ema = current_slow_ema
        
        # Log signal changes
        if signal != 0:
            logger.info(f"Signal detected: {signal} | Strength: {signal_strength:.4f} | Fast: {current_fast_ema:.6f} | Slow: {current_slow_ema:.6f}")

    def _detect_crossover(self, df: pd.DataFrame, fast_col: str, slow_col: str) -> int:
        """
        Detect EMA crossover signals.
        
        This method examines the last two data points to detect crossovers:
        - Compares current and previous EMA values
        - Returns 1 for bullish crossover (fast crosses above slow)
        - Returns -1 for bearish crossover (fast crosses below slow)
        - Returns 0 for no crossover
        
        Args:
            df: DataFrame with EMA calculations
            fast_col: Fast EMA column name
            slow_col: Slow EMA column name
            
        Returns:
            int: Signal value (-1, 0, or 1)
        """
        if len(df) < 2:
            return 0
        
        # Current and previous EMA values
        curr_fast = df[fast_col].iloc[-1]
        curr_slow = df[slow_col].iloc[-1]
        prev_fast = df[fast_col].iloc[-2]
        prev_slow = df[slow_col].iloc[-2]
        
        # Detect crossovers
        # Long signal: fast EMA crosses above slow EMA
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return 1
        
        # Short signal: fast EMA crosses below slow EMA
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            return -1
        
        # No crossover detected
        return 0

    def _calculate_signal_strength(self, fast_ema: float, slow_ema: float) -> float:
        """
        Calculate signal strength based on EMA separation.
        
        Signal strength is the percentage difference between fast and slow EMAs:
        - Positive values indicate fast EMA above slow (bullish)
        - Negative values indicate fast EMA below slow (bearish)
        - Magnitude indicates strength of trend
        
        Args:
            fast_ema: Current fast EMA value
            slow_ema: Current slow EMA value
            
        Returns:
            float: Signal strength (normalized percentage difference)
        """
        if slow_ema == 0:
            return 0.0
        
        # Calculate percentage difference between EMAs
        strength = (fast_ema - slow_ema) / slow_ema
        return strength

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal) -> PositionExecutorConfig:
        """
        Get executor configuration for position creation.
        
        This creates a PositionExecutorConfig without triple barrier parameters,
        relying on signal-based exits instead. The position will remain open
        until an opposite crossover signal is generated.
        
        Args:
            trade_type: BUY (long) or SELL (short)
            price: Current market price
            amount: Position size in base currency
            
        Returns:
            PositionExecutorConfig: Configuration for PositionExecutor
        """
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            trading_pair=self.config.trading_pair,
            connector_name=self.config.connector_name,
            side=trade_type,
            amount=amount,
            # No triple barrier configuration
            # Exits are handled by opposite crossover signals
        )

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create position actions based on current signals.
        
        This method:
        1. Checks for valid signals in processed_data
        2. Verifies if we can create a new executor (cooldown, max executors)
        3. Calculates position size based on current price
        4. Creates a CreateExecutorAction if conditions are met
        
        Returns:
            List[ExecutorAction]: List of actions to execute (empty or single action)
        """
        create_actions = []
        
        # Get current signal
        signal = self.processed_data.get("signal", 0)
        signal_strength = abs(self.processed_data.get("signal_strength", 0.0))
        
        # Check if we have a valid signal
        if signal == 0:
            return create_actions
        
        # Check signal strength threshold
        if signal_strength < self.config.min_signal_strength:
            logger.debug(f"Signal strength {signal_strength:.4f} below threshold {self.config.min_signal_strength}")
            return create_actions
        
        # Check if we can create a new executor
        if not self.can_create_executor(signal):
            logger.debug(f"Cannot create executor: cooldown or max executors reached")
            return create_actions
        
        # Get current market price
        price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, 
            self.config.trading_pair,
            PriceType.MidPrice
        )
        
        # Calculate position size (fixed quote amount / current price)
        amount = self.config.position_size_quote / price
        
        # Determine trade direction
        trade_type = TradeType.BUY if signal > 0 else TradeType.SELL
        
        # Create executor action
        create_actions.append(CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=self.get_executor_config(trade_type, price, amount)
        ))
        
        logger.info(f"Creating {trade_type.name} position: {amount:.6f} @ {price:.6f}")
        
        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Propose stop actions based on opposite signals.
        
        In EWMAC strategy, positions are closed when opposite crossover signals occur:
        - Long positions closed on short signals (fast EMA crosses below slow EMA)
        - Short positions closed on long signals (fast EMA crosses above slow EMA)
        
        This overrides the base class to implement signal-based exits
        instead of using stop-loss or take-profit levels.
        
        Returns:
            List[ExecutorAction]: List of stop actions for active executors
        """
        stop_actions = []
        signal = self.processed_data.get("signal", 0)
        
        # No signal, no exits
        if signal == 0:
            return stop_actions
        
        # Get active executors
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active
        )
        
        for executor in active_executors:
            # Close long positions on short signals
            if executor.side == TradeType.BUY and signal == -1:
                stop_actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
                logger.info(f"Closing LONG position {executor.id} on SHORT signal")
            
            # Close short positions on long signals
            elif executor.side == TradeType.SELL and signal == 1:
                stop_actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
                logger.info(f"Closing SHORT position {executor.id} on LONG signal")
        
        return stop_actions

    def to_format_status(self) -> List[str]:
        """
        Format controller status for display.
        
        This provides a human-readable summary of the controller's current state,
        including signal status, EMA values, and active positions.
        
        Returns:
            List[str]: List of status lines for display
        """
        lines = []
        
        if not hasattr(self, 'processed_data') or self.processed_data.get('insufficient_data', True):
            lines.append("Status: Waiting for sufficient data...")
            lines.append(f"Required periods: {self.config.slow_ema_period}")
            return lines
        
        # Signal information
        signal = self.processed_data.get('signal', 0)
        signal_str = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "NEUTRAL")
        lines.append(f"Signal: {signal_str} ({signal})")
        lines.append(f"Signal Strength: {self.processed_data.get('signal_strength', 0):.4f}")
        
        # EMA values
        lines.append(f"Fast EMA ({self.config.fast_ema_period}): {self.processed_data.get('current_fast_ema', 0):.6f}")
        lines.append(f"Slow EMA ({self.config.slow_ema_period}): {self.processed_data.get('current_slow_ema', 0):.6f}")
        lines.append(f"Current Price: {self.processed_data.get('current_price', 0):.6f}")
        
        # Position information
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active
        )
        lines.append(f"Active Positions: {len(active_executors)}")
        
        return lines