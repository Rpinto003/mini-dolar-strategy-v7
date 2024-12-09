from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.analysis.market_structure import MarketStructure
from src.analysis.volatility import VolatilityAnalyzer
from src.analysis.signals import SignalGenerator
from src.analysis.risk_management import RiskManager
from src.analysis.feature_engineering import FeatureEngineer
from src.models.metrics import MetricsCalculator
from src.models.ml_model import MLModel

class EnhancedStrategy:
    """Enhanced trading strategy implementation with machine learning capabilities"""
    
    def __init__(self, config: Dict = None):
        """Initialize strategy with configuration"""
        self.config = config or {}
        self.initialize_components()
        self.metrics = {}
        logger.info("Strategy initialized with ML capabilities")
    
    def initialize_components(self):
        """Initialize strategy components with configuration"""
        self.market_structure = MarketStructure(
            lookback_period=self.config.get('lookback_period', 20)
        )
        self.volatility_analyzer = VolatilityAnalyzer(
            atr_period=self.config.get('atr_period', 14)
        )
        self.signal_generator = SignalGenerator(
            risk_free_rate=self.config.get('risk_free_rate', 0.05)
        )
        self.risk_manager = RiskManager(
            initial_capital=self.config.get('initial_capital', 100000),
            max_risk_per_trade=self.config.get('max_risk_per_trade', 0.01)
        )
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize ML components
        self.feature_engineer = FeatureEngineer(config=self.config.get('feature_engineering', {}))
        self.ml_model = MLModel(config=self.config.get('ml_model', {}))
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute strategy pipeline with ML integration"""
        logger.info("Starting strategy execution with ML")
        
        try:
            # Market structure analysis
            data = self.market_structure.identify_structure(data)
            
            # Volatility analysis
            data = self.volatility_analyzer.calculate_volatility(data)
            
            # Feature engineering for ML
            data = self.feature_engineer.create_features(data)
            
            # Generate ML predictions
            ml_signals = self.ml_model.predict(data)
            data['ml_signal'] = ml_signals
            
            # Generate traditional signals
            data = self.signal_generator.generate_signals(data)
            
            # Combine ML and traditional signals
            data['final_signal'] = self._combine_signals(data)
            
            # Apply risk management
            data = self.risk_manager.apply_risk_management(data)
            
            # Calculate performance metrics
            self.calculate_strategy_metrics(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            return pd.DataFrame()
    
    def train_ml_model(self, training_data: pd.DataFrame) -> Dict:
        """Train the machine learning model"""
        try:
            # Prepare features
            data = self.market_structure.identify_structure(training_data)
            data = self.volatility_analyzer.calculate_volatility(data)
            data = self.feature_engineer.create_features(data)
            
            # Train model
            metrics = self.ml_model.train(data)
            logger.info("ML model training completed")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return {}
    
    def _combine_signals(self, data: pd.DataFrame) -> pd.Series:
        """Combine ML and traditional signals with weights"""
        # Get signal weights from config
        ml_weight = self.config.get('signal_weights', {}).get('ml_weight', 0.5)
        traditional_weight = self.config.get('signal_weights', {}).get('traditional_weight', 0.5)
        
        # Combine signals
        combined_signal = (
            ml_weight * data['ml_signal'] +
            traditional_weight * data['signal']
        )
        
        # Convert to discrete signals
        final_signal = pd.Series(0, index=data.index)
        signal_threshold = self.config.get('signal_threshold', 0.5)
        
        final_signal[combined_signal > signal_threshold] = 1
        final_signal[combined_signal < -signal_threshold] = -1
        
        return final_signal
    
    def calculate_strategy_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        try:
            # Calculate trade metrics
            trades = data[data['final_signal'] != 0].copy()
            trade_metrics = self.metrics_calculator.calculate_trade_metrics(trades)
            
            # Calculate portfolio metrics
            portfolio_metrics = self.metrics_calculator.calculate_portfolio_metrics(trades)
            
            # Calculate ML-specific metrics
            ml_metrics = self._calculate_ml_metrics(data)
            
            # Combine all metrics
            self.metrics = {
                **trade_metrics,
                **portfolio_metrics,
                **ml_metrics
            }
            
            self._log_performance_metrics()
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _calculate_ml_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate ML-specific performance metrics"""
        try:
            ml_metrics = {}
            
            # Calculate signal agreement rate
            agreement = (data['ml_signal'] == data['signal']).mean()
            ml_metrics['ml_traditional_agreement'] = agreement
            
            # Calculate ML signal contribution
            ml_correct = (
                (data['ml_signal'] == 1) & (data['returns'] > 0) |
                (data['ml_signal'] == -1) & (data['returns'] < 0)
            ).mean()
            ml_metrics['ml_accuracy'] = ml_correct
            
            return ml_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ML metrics: {str(e)}")
            return {}
    
    def _log_performance_metrics(self):
        """Log key performance metrics"""
        logger.info("Strategy Performance:")
        logger.info(f"Total Trades: {self.metrics.get('total_trades', 0)}")
        logger.info(f"Win Rate: {self.metrics.get('win_rate', 0):.2%}")
        logger.info(f"Total Return: {self.metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"ML Accuracy: {self.metrics.get('ml_accuracy', 0):.2%}")
        logger.info(f"ML-Traditional Agreement: {self.metrics.get('ml_traditional_agreement', 0):.2%}")
