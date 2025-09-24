"""
EWMAC Strategy Script
File: ewmac_script.py

Main strategy script for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.
Phase 1: Single pair implementation with basic position sizing.
"""

from decimal import Decimal
from typing import Dict

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.strategy.strategy_v2_base import StrategyV2Base

from ewmac_config_v01 import EWMACControllerConfig
from ewmac_controller_v01 import EWMACController


class EWMACScript(StrategyV2Base):
    """
    EWMAC (Exponentially Weighted Moving Average Crossover) Strategy Script.
    
    Phase 1 Implementation Features:
    - Single trading pair support
    - EMA crossover signal generation (8/32 periods)
    - Simple fixed position sizing
    - Signal-based exits (no triple barriers)
    - Compatible with Hummingbot backtesting
    
    Usage:
        This script manages a single EWMAC controller for one trading pair.
        It can be easily extended to multi-pair in later phases.
    """
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: dict = None):
        """
        Initialize the EWMAC strategy script.
        
        Args:
            connectors: Dictionary of connector instances
            config: Strategy configuration dictionary
        """
        # Default configuration
        if config is None:
            config = {
                "trading_pair": "ETH-USDT",
                "connector_name": "binance_paper_trade",
                "fast_ema_period": 8,
                "slow_ema_period": 32,
                "position_size_quote": 100,
                "total_amount_quote": 1000,
                "max_executors_per_side": 1,
                "cooldown_time": 60,
                "interval": "1m",
                "min_signal_strength": 0.0
            }
        
        self.config = config
        self.trading_pair = config["trading_pair"]
        self.connector_name = config["connector_name"]
        
        # Initialize controller configuration
        controller_config = EWMACControllerConfig(
            id=f"ewmac_{self.trading_pair.replace('-', '_')}",
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            
            # EMA parameters
            fast_ema_period=config["fast_ema_period"],
            slow_ema_period=config["slow_ema_period"],
            
            # Position sizing
            position_size_quote=Decimal(str(config["position_size_quote"])),
            
            # Signal filtering
            min_signal_strength=config["min_signal_strength"],
            
            # Candles configuration
            interval=config["interval"],
            
            # Directional controller base parameters
            total_amount_quote=Decimal(str(config["total_amount_quote"])),
            max_executors_per_side=config["max_executors_per_side"],
            cooldown_time=config["cooldown_time"],
        )
        
        # Create controller instance
        self.controller = EWMACController(controller_config)
        
        # Initialize strategy base
        super().__init__(
            connectors=connectors,
            controllers=[self.controller]
        )

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        
        Args:
            clock: Hummingbot clock instance
            timestamp: Strategy start timestamp
        """
        self._last_timestamp = timestamp
        self.logger().info(f"EWMAC Strategy started at {timestamp}")
        self.logger().info(f"Trading pair: {self.trading_pair}")
        self.logger().info(f"Connector: {self.connector_name}")
        self.logger().info(f"EMA periods: {self.controller.config.fast_ema_period}/{self.controller.config.slow_ema_period}")

    @property
    def market_data_extra_info(self) -> str:
        """
        Provide additional market data information for display.
        
        Returns:
            str: Formatted string with current strategy state
        """
        if not hasattr(self.controller, 'processed_data'):
            return "Initializing EWMAC strategy..."
        
        data = self.controller.processed_data
        
        if data.get('insufficient_data', True):
            return "Waiting for sufficient market data..."
        
        # Format current state information
        lines = [
            f"=== EWMAC Strategy Status ===",
            f"Trading Pair: {self.trading_pair}",
            f"Current Signal: {data.get('signal', 0)}",
            f"Fast EMA ({self.controller.config.fast_ema_period}): {data.get('current_fast_ema', 0):.6f}",
            f"Slow EMA ({self.controller.config.slow_ema_period}): {data.get('current_slow_ema', 0):.6f}",
            f"Signal Strength: {data.get('signal_strength', 0):.4f}",
            f"Active Executors: {len(self.controller.executors_info)}",
        ]
        
        return "\n".join(lines)