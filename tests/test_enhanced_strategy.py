import pytest
import pandas as pd
import numpy as np
from src.strategy.enhanced_strategy import EnhancedStrategy

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def strategy():
    config = {
        'lookback_period': 20,
        'atr_period': 14,
        'initial_capital': 100000,
        'max_risk_per_trade': 0.01,
        'signal_weights': {
            'ml_weight': 0.6,
            'traditional_weight': 0.4
        }
    }
    return EnhancedStrategy(config=config)

def test_strategy_initialization(strategy):
    assert strategy is not None
    assert strategy.config['lookback_period'] == 20
    assert strategy.config['atr_period'] == 14

def test_strategy_run(strategy, sample_data):
    results = strategy.run(sample_data)
    assert not results.empty
    assert 'ml_signal' in results.columns
    assert 'final_signal' in results.columns

def test_ml_model_training(strategy, sample_data):
    metrics = strategy.train_ml_model(sample_data)
    assert metrics is not None
    assert 'accuracy' in metrics

def test_signal_combination(strategy, sample_data):
    data = pd.DataFrame({
        'ml_signal': [1, -1, 0, 1, -1],
        'signal': [1, -1, 0, -1, 1]
    })
    combined = strategy._combine_signals(data)
    assert len(combined) == 5
    assert all(combined.isin([-1, 0, 1]))