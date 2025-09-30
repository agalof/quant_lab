"""
EWMAC Strategy Script
File: ewmac_script.py

Main strategy script for EWMAC (Exponentially Weighted Moving Average Crossover) strategy.

This script orchestrates the EWMAC strategy by:
- Initializing the controller with configuration
- Managing the strategy lifecycle
- Providing status information

Usage in Hummingbot:
    start --script ewmac_script.py

Usage in backtesting:
    See example notebooks in research_notebooks/strategies/ewmac/

Author: Quants Lab Team
Date: 2025-09-29
"""

from decimal import Decimal
from typing import Dict, Optional
import logging

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.strategy.strategy_v2_base import StrategyV2Base

from ewmac_config import EWMACControllerConfig
from ewmac_controller import EWMACController


logger = logging.getLogger(__name__)


class EWMACScript(StrategyV2Base):
    """
    EWMAC (Exponentially Weighted Moving Average Crossover) Strategy Script.
    
    This script implements a simple trend-following strategy using EMA crossovers:
    - Generates long signals when fast EMA crosses above slow EMA
    - Generates short signals when fast EMA crosses below slow EMA
    - Exits positions on opposite crossover signals
    
    Features:
    - Single trading pair support
    - Fixed position sizing for consistent risk
    - Signal-based entries and exits
    - Compatible with Hummingbot V2 framework
    - Suitable for backtesting and live trading
    
    Configuration Parameters:
    - trading_pair: Trading pair (e.g., "ETH-USDT")
    - connector_name: Exchange connector (e.g., "binance_paper_trade")
    - fast_ema_period: Fast EMA period (default: 8)
    - slow_ema_period: Slow EMA period (default: 32)
    - position_size_quote: Position size in quote currency (default: 100)
    - total_amount_quote: Total capital allocation (default: 1000)
    - max_executors_per_side: Max concurrent positions per side (default: 1)
    - cooldown_time: Seconds between position changes (default: 60)
    - interval: Candle interval (default: "1m")
    - min_signal_strength: Minimum signal strength threshold (default: 0.0)
    
    Example Configuration:
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
        script = EWMACScript(connectors, config)
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
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
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[Dict] = None):
        """
        Initialize the EWMAC strategy script.
        
        Args:
            connectors: Dictionary of connector instances keyed by connector name
            config: Strategy configuration dictionary (uses defaults if None)
        
        Raises:
            ValueError: If configuration validation fails
        """
        # Merge provided config with defaults
        if config is None:
            config = self.DEFAULT_CONFIG.copy()
        else:
            # Start with defaults and override with provided values
            full_config = self.DEFAULT_CONFIG.copy()
            full_config.update(config)
            config = full_config
        
        self.config = config
        self.trading_pair = config["trading_pair"]
        self.connector_name = config["connector_name"]
        
        logger.info(f"Initializing EWMAC Strategy for {self.trading_pair} on {self.connector_name}")
        
        # Create controller configuration
        controller_config = EWMACControllerConfig(
            # Strategy identification
            id=f"ewmac_{self.trading_pair.replace('-', '_').lower()}",
            
            # Exchange and trading pair
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
        
        # Initialize strategy base with controller
        super().__init__(
            connectors=connectors,
            controllers=[self.controller]
        )
        
        logger.info(f"EWMAC Strategy initialized with {controller_config.fast_ema_period}/{controller_config.slow_ema_period} EMA")

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        
        This method is called when the strategy begins execution.
        It logs the initial state and configuration.
        
        Args:
            clock: Hummingbot clock instance
            timestamp: Strategy start timestamp (Unix timestamp)
        """
        self._last_timestamp = timestamp
        
        logger.info("=" * 60)
        logger.info("EWMAC Strategy Started")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Trading Pair: {self.trading_pair}")
        logger.info(f"Connector: {self.connector_name}")
        logger.info(f"EMA Periods: {self.controller.config.fast_ema_period}/{self.controller.config.slow_ema_period}")
        logger.info(f"Position Size: {self.controller.config.position_size_quote} {self.trading_pair.split('-')[1]}")
        logger.info(f"Total Capital: {self.controller.config.total_amount_quote} {self.trading_pair.split('-')[1]}")
        logger.info(f"Interval: {self.controller.config.interval}")
        logger.info(f"Min Signal Strength: {self.controller.config.min_signal_strength}")
        logger.info("=" * 60)

    def stop(self, clock: Clock) -> None:
        """
        Stop the strategy.
        
        This method is called when the strategy is stopped.
        It logs the final state and can be extended for cleanup operations.
        
        Args:
            clock: Hummingbot clock instance
        """
        logger.info("=" * 60)
        logger.info("EWMAC Strategy Stopped")
        logger.info("=" * 60)
        
        # Log final statistics
        if hasattr(self.controller, 'executors_info'):
            total_executors = len(self.controller.executors_info)
            active_executors = len([e for e in self.controller.executors_info if e.is_active])
            logger.info(f"Total Executors Created: {total_executors}")
            logger.info(f"Active Executors: {active_executors}")
        
        logger.info("=" * 60)

    @property
    def market_data_extra_info(self) -> str:
        """
        Provide additional market data information for display.
        
        This property is used by the Hummingbot UI to show strategy status.
        It displays current signals, EMA values, and position information.
        
        Returns:
            str: Formatted string with current strategy state
        """
        if not hasattr(self.controller, 'processed_data'):
            return "Initializing EWMAC strategy..."
        
        data = self.controller.processed_data
        
        if data.get('insufficient_data', True):
            return f"Waiting for sufficient market data (need {self.controller.config.slow_ema_period} candles)..."
        
        # Format current state information
        signal = data.get('signal', 0)
        signal_str = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "NEUTRAL")
        
        lines = [
            "=" * 50,
            "EWMAC Strategy Status",
            "=" * 50,
            f"Trading Pair: {self.trading_pair}",
            f"Signal: {signal_str} ({signal})",
            f"Fast EMA ({self.controller.config.fast_ema_period}): {data.get('current_fast_ema', 0):.6f}",
            f"Slow EMA ({self.controller.config.slow_ema_period}): {data.get('current_slow_ema', 0):.6f}",
            f"Current Price: {data.get('current_price', 0):.6f}",
            f"Signal Strength: {data.get('signal_strength', 0):.4f}",
            f"Active Executors: {len([e for e in self.controller.executors_info if e.is_active])}/{self.controller.config.max_executors_per_side}",
            "=" * 50,
        ]
        
        return "\n".join(lines)

    def format_status(self) -> str:
        """
        Format comprehensive status information.
        
        This method provides detailed status information for logging and display,
        including configuration, signals, and performance metrics.
        
        Returns:
            str: Formatted status string
        """
        lines = [
            "\n" + "=" * 70,
            "EWMAC Strategy Status Report",
            "=" * 70,
            "",
            "Configuration:",
            f"  Trading Pair: {self.trading_pair}",
            f"  Connector: {self.connector_name}",
            f"  Fast EMA Period: {self.controller.config.fast_ema_period}",
            f"  Slow EMA Period: {self.controller.config.slow_ema_period}",
            f"  Position Size: {self.controller.config.position_size_quote}",
            f"  Interval: {self.controller.config.interval}",
            "",
        ]
        
        # Add controller status
        if hasattr(self.controller, 'processed_data'):
            data = self.controller.processed_data
            if not data.get('insufficient_data', True):
                signal = data.get('signal', 0)
                signal_str = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "NEUTRAL")
                
                lines.extend([
                    "Current Signal:",
                    f"  Status: {signal_str} ({signal})",
                    f"  Strength: {data.get('signal_strength', 0):.4f}",
                    f"  Fast EMA: {data.get('current_fast_ema', 0):.6f}",
                    f"  Slow EMA: {data.get('current_slow_ema', 0):.6f}",
                    f"  Price: {data.get('current_price', 0):.6f}",
                    "",
                ])
        
        # Add executor information
        if hasattr(self.controller, 'executors_info'):
            total_executors = len(self.controller.executors_info)
            active_executors = [e for e in self.controller.executors_info if e.is_active]
            
            lines.extend([
                "Executors:",
                f"  Total Created: {total_executors}",
                f"  Currently Active: {len(active_executors)}",
                f"  Max Per Side: {self.controller.config.max_executors_per_side}",
                "",
            ])
            
            if active_executors:
                lines.append("  Active Positions:")
                for executor in active_executors:
                    side_str = "LONG" if executor.side.name == "BUY" else "SHORT"
                    lines.append(f"    {executor.id}: {side_str}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# Example usage for testing
if __name__ == "__main__":
    # This section is for testing purposes only
    print("EWMAC Strategy Script")
    print("=" * 50)
    print("This script implements an EMA crossover strategy.")
    print("For usage with Hummingbot, run: start --script ewmac_script.py")
    print("For backtesting, use the Jupyter notebooks in research_notebooks/")
    print("=" * 50)