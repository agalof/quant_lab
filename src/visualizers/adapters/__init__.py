# quant_lab/src/visualizers/adapters/__init__.py
"""
Adapters Package for Data Source Integration
===========================================

Provides seamless integration between various data sources and the StrategyVisualizer.

Components:
- ConnectorAdapter: CLOBDataSource integration for research
- BacktestingAdapter: BacktestingResult integration (future)
- PerformanceAdapter: Live vs backtest comparison (future)

Signal Generators:
- SimpleMovingAverageCrossover: MA crossover strategy
- RSIMeanReversion: RSI mean reversion strategy
- Custom signal generator framework
"""

from .connector_adapter import (
    ConnectorAdapter,
    SignalGenerator, 
    IndicatorEngine,
    SimpleMovingAverageCrossover,
    RSIMeanReversion
)

__all__ = [
    'ConnectorAdapter',
    'SignalGenerator',
    'IndicatorEngine', 
    'SimpleMovingAverageCrossover',
    'RSIMeanReversion'
]