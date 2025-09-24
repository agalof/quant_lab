"""
EWMAC Strategy Configuration
File: ewmac_config.py

Configuration class for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.
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
    
    This strategy generates signals based on the crossover of fast and slow EMAs:
    - Long signal: Fast EMA crosses above Slow EMA
    - Short signal: Fast EMA crosses below Slow EMA
    
    Phase 1 features:
    - Simple position sizing (fixed amount per trade)
    - Single pair support (will expand to multi-pair later)
    - Basic signal generation without strength filtering
    """
    controller_name: str = "ewmac"
    
    # Candles configuration
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(default="1m", description="Candle interval for signal generation")
    
    # EMA Parameters
    fast_ema_period: int = Field(default=8, description="Fast EMA period")
    slow_ema_period: int = Field(default=32, description="Slow EMA period")
    
    # Position sizing (Phase 1: Simple fixed amount)
    position_size_quote: Decimal = Field(
        default=Decimal("100"), 
        description="Fixed position size in quote currency (USD)"
    )
    
    # Signal filtering (Phase 1: Basic, will add strength filter later)
    min_signal_strength: float = Field(
        default=0.0, 
        description="Minimum signal strength threshold (0.0 = no filtering)"
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        """Auto-set candles connector to match trading connector if not specified."""
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        """Auto-set candles trading pair to match trading pair if not specified."""
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v
    
    @field_validator("slow_ema_period")
    @classmethod
    def validate_ema_periods(cls, v, validation_info: ValidationInfo):
        """Ensure slow EMA period is greater than fast EMA period."""
        fast_period = validation_info.data.get("fast_ema_period", 8)
        if v <= fast_period:
            raise ValueError(f"Slow EMA period ({v}) must be greater than fast EMA period ({fast_period})")
        return v