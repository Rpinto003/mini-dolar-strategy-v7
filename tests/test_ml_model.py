import pytest
import pandas as pd
import numpy as np
from src.models.ml_model import MLModel

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'rsi': np.random.rand(len(dates)) * 100,
        'macd': np.random.randn(len(dates)),
        'volume_ratio': np.random.rand(len(dates)) * 2,
        'price_position': np.random.rand(len(dates)),
        'historical_volatility': np.random.rand(len(dates)) * 0.5,
        'atr': np.random.rand(len(dates)) * 2
    }, index=dates)
    return data

@pytest.fixture
def ml_model():
    config = {
        'probability_threshold': 0.7,
        'n_estimators': 100,
        'max_depth': 10
    }
    return MLModel(config=config)

def test_model_initialization(ml_model):
    assert ml_model is not None
    assert ml_model.model is None
    assert ml_model.feature_columns is not None

def test_feature_preparation(ml_model, sample_data):
    X, y = ml_model.prepare_features(sample_data)
    assert X is not None
    assert y is not None
    assert X.shape[1] == len(ml_model.feature_columns)

def test_model_training(ml_model, sample_data):
    metrics = ml_model.train(sample_data)
    assert metrics is not None
    assert 'accuracy' in metrics
    assert ml_model.model is not None

def test_prediction_generation(ml_model, sample_data):
    ml_model.train(sample_data)
    predictions = ml_model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert all(predictions.isin([-1, 0, 1]))