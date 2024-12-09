import pytest
import pandas as pd
import numpy as np
from src.analysis.feature_engineering import FeatureEngineer

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
def feature_engineer():
    config = {
        'roc_periods': [5, 10, 20],
        'momentum_periods': [5, 10],
        'volume_ma_periods': [5, 10, 20]
    }
    return FeatureEngineer(config=config)

def test_feature_creation(feature_engineer, sample_data):
    features = feature_engineer.create_features(sample_data)
    assert not features.empty
    assert 'roc_5' in features.columns
    assert 'momentum_5' in features.columns
    assert 'volume_sma_5' in features.columns

def test_momentum_features(feature_engineer, sample_data):
    data = feature_engineer._add_momentum_features(sample_data)
    assert 'roc_5' in data.columns
    assert 'momentum_5' in data.columns
    
def test_volatility_features(feature_engineer, sample_data):
    data = feature_engineer._add_volatility_features(sample_data)
    assert 'bb_width' in data.columns
    assert 'atr_ratio' in data.columns

def test_volume_features(feature_engineer, sample_data):
    data = feature_engineer._add_volume_features(sample_data)
    assert 'volume_momentum' in data.columns
    assert 'volume_sma_5' in data.columns