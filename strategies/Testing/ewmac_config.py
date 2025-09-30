"""
EWMAC Strategy Configuration
File: ewmac_config.py

Configuration class for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.

This configuration defines all parameters needed for the EWMAC strategy:
- EMA periods for signal generation
- Position sizing parameters
- Trading pair and connector settings
- Risk management parameters

Author: Quants Lab Team
Date: 2025-09-29
"""

from decimal import Decimal
from typing import List

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo


class EWMACControllerConfig(DirectionalTradingControllerConfigBase):
    """
    Configuration for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.
    
    The EWMAC strategy generates trading signals based on the crossover of two EMAs:
    - Long signal: Fast EMA crosses above Slow EMA
    - Short signal: Fast EMA crosses below Slow EMA
    - Exit: Opposite crossover signal
    
    Key Features:
    - Simple, interpretable signal generation
    - Fixed position sizing for consistent risk
    - Signal-based exits (no stop-loss/take-profit)
    - Suitable for trending markets
    
    Attributes:
        controller_name: Name identifier for the controller type
        fast_ema_period: Period for the fast EMA (default: 8)
        slow_ema_period: Period for the slow EMA (default: 32)
        position_size_quote: Fixed position size in quote currency
        min_signal_strength: Minimum signal strength threshold (0.0 = no filter)
        interval: Candle interval for data fetching
        candles_config: List of candles configurations
        candles_connector: Connector for candle data (auto-set if None)
        candles_trading_pair: Trading pair for candles (auto-set if None)
    """
    
    controller_name: str = "ewmac"
    
    # ========== Candles Configuration ==========
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(
        default="1m", 
        description="Candle interval (e.g., '1m', '5m', '1h')"
    )
    
    # ========== EMA Parameters ==========
    fast_ema_period: int = Field(
        default=8,
        ge=2,
        le=100,
        description="Fast EMA period (must be < slow EMA period)"
    )
    
    slow_ema_period: int = Field(
        default=32,
        ge=3,
        le=500,
        description="Slow EMA period (must be > fast EMA period)"
    )
    
    # ========== Position Sizing ==========
    position_size_quote: Decimal = Field(
        default=Decimal("100"),
        gt=0,
        description="Fixed position size in quote currency (e.g., USDT)"
    )
    
    # ========== Signal Filtering ==========
    min_signal_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum signal strength threshold (0.0 = accept all signals)"
    )
    
    # ========== Validators ==========
    
    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        """
        Auto-set candles connector to match trading connector if not specified.
        
        This ensures we fetch candles from the same exchange we're trading on
        unless explicitly configured otherwise.
        """
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        """
        Auto-set candles trading pair to match trading pair if not specified.
        
        This ensures we fetch candles for the correct trading pair
        unless explicitly configured otherwise.
        """
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v
    
    @field_validator("slow_ema_period")
    @classmethod
    def validate_ema_periods(cls, v, validation_info: ValidationInfo):
        """
        Ensure slow EMA period is greater than fast EMA period.
        
        This validation prevents configuration errors where the slow EMA
        would be faster than the fast EMA, which would break the strategy logic.
        """
        fast_period = validation_info.data.get("fast_ema_period")
        if fast_period is not None and v <= fast_period:
            raise ValueError(
                f"slow_ema_period ({v}) must be greater than fast_ema_period ({fast_period})"
            )
        return v

    def model_post_init(self, __context):
        """
        Additional validation and setup after model initialization.
        
        This ensures all derived parameters are properly configured.
        """
        super().model_post_init(__context)
        
        # Log configuration for debugging
        if hasattr(self, '_logger'):
            self._logger.info(f"EWMAC Config initialized: {self.fast_ema_period}/{self.slow_ema_period} EMA")
            self._logger.info(f"Position size: {self.position_size_quote} {self.trading_pair.split('-')[1]}")