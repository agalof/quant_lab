"""
EWMAC Strategy Controller
File: ewmac_controller.py

Controller implementation for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.
"""

from decimal import Decimal
from typing import List

import pandas as pd
import pandas_ta as ta
from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, CreateExecutorAction, StopExecutorAction

from ewmac_config_v01 import EWMACControllerConfig


class EWMACController(DirectionalTradingControllerBase):
    """
    EWMAC (Exponentially Weighted Moving Average Crossover) Controller.
    
    This controller implements a directional trading strategy based on EMA crossovers:
    - Generates long signals when fast EMA crosses above slow EMA
    - Generates short signals when fast EMA crosses below slow EMA
    - Uses simple position sizing in Phase 1
    - Exits positions on opposite crossover signals
    
    Signal Values:
    - 1: Long signal (fast EMA > slow EMA and crossover detected)
    - -1: Short signal (fast EMA < slow EMA and crossover detected)
    - 0: No signal (hold current position or no crossover)
    """

    def __init__(self, config: EWMACControllerConfig, *args, **kwargs):
        """
        Initialize the EWMAC controller.
        
        Args:
            config: EWMACControllerConfig instance
        """
        self.config = config
        
        # Calculate required historical data length
        # Need enough data for slow EMA + buffer for reliable signals
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

    async def update_processed_data(self):
        """
        Update processed data with current market signals.
        
        This method:
        1. Fetches OHLCV candles data
        2. Calculates fast and slow EMAs
        3. Detects crossover signals
        4. Updates processed_data with signal and features
        """
        # Get candles data
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        
        if df.empty or len(df) < self.config.slow_ema_period:
            # Insufficient data for signal generation
            self.processed_data = {
                "signal": 0, 
                "features": pd.DataFrame(),
                "insufficient_data": True
            }
            return
        
        # Calculate EMAs using pandas_ta
        df.ta.ema(length=self.config.fast_ema_period, append=True)
        df.ta.ema(length=self.config.slow_ema_period, append=True)
        
        # Get EMA column names
        fast_ema_col = f"EMA_{self.config.fast_ema_period}"
        slow_ema_col = f"EMA_{self.config.slow_ema_period}"
        
        # Extract current EMA values
        current_fast_ema = df[fast_ema_col].iloc[-1]
        current_slow_ema = df[slow_ema_col].iloc[-1]
        
        # Generate crossover signals
        signal = self._generate_crossover_signal(df, fast_ema_col, slow_ema_col)
        
        # Calculate signal strength (Phase 1: basic implementation)
        signal_strength = self._calculate_signal_strength(current_fast_ema, current_slow_ema)
        
        # Apply signal strength filter
        if abs(signal_strength) < self.config.min_signal_strength:
            signal = 0
        
        # Update processed data
        self.processed_data = {
            "signal": signal,
            "features": df,
            "current_fast_ema": current_fast_ema,
            "current_slow_ema": current_slow_ema,
            "signal_strength": signal_strength,
            "insufficient_data": False
        }
        
        # Update state for next iteration
        self.previous_signal = signal
        self.last_fast_ema = current_fast_ema
        self.last_slow_ema = current_slow_ema

    def _generate_crossover_signal(self, df: pd.DataFrame, fast_col: str, slow_col: str) -> int:
        """
        Generate crossover signals based on EMA relationships.
        
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
        
        This method creates a PositionExecutorConfig without triple barrier parameters,
        relying on signal-based exits instead.
        
        Args:
            trade_type: BUY or SELL
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
            # No triple barrier configuration - exits handled by opposite signals
            # This allows the position to run until opposite crossover occurs
        )

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create position actions based on current signals.
        
        Returns:
            List[ExecutorAction]: List of actions to execute
        """
        create_actions = []
        signal = self.processed_data.get("signal", 0)
        
        # Only create new positions on valid signals
        if signal != 0 and self.can_create_executor(signal):
            # Get current market price
            price = self.market_data_provider.get_price_by_type(
                self.config.connector_name, 
                self.config.trading_pair,
                PriceType.MidPrice
            )
            
            # Calculate position size (Phase 1: simple fixed amount)
            amount = self.config.position_size_quote / price
            
            # Determine trade direction
            trade_type = TradeType.BUY if signal > 0 else TradeType.SELL
            
            # Create executor action
            create_actions.append(CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=self.get_executor_config(trade_type, price, amount)
            ))
        
        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Propose stop actions based on opposite signals.
        
        In EWMAC strategy, positions are closed when opposite crossover signals occur.
        This overrides the base class to implement signal-based exits.
        
        Returns:
            List[ExecutorAction]: List of stop actions
        """
        stop_actions = []
        signal = self.processed_data.get("signal", 0)
        
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
            
            # Close short positions on long signals
            elif executor.side == TradeType.SELL and signal == 1:
                stop_actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
        
        return stop_actions