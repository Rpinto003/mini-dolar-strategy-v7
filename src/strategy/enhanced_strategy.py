from typing import Dict
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
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initialize_components()
        self.metrics = {}
        logger.info("Strategy initialized with ML capabilities")
    
    def initialize_components(self):
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
        
        # ML components
        self.feature_engineer = FeatureEngineer(config=self.config.get('feature_engineering', {}))
        self.ml_model = MLModel(config=self.config.get('ml_model', {}))
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting strategy execution with ML")
        
        try:
            # Market analysis
            data = self.market_structure.identify_structure(data)
            data = self.volatility_analyzer.calculate_volatility(data)
            
            # ML pipeline
            data = self.feature_engineer.create_features(data)
            ml_signals = self.ml_model.predict(data)
            data['ml_signal'] = ml_signals
            
            # Traditional signals
            data = self.signal_generator.generate_signals(data)
            
            # Combine signals
            data['final_signal'] = self._combine_signals(data)
            
            # Risk management
            data = self.risk_manager.apply_risk_management(data)
            
            # Performance metrics
            self.calculate_strategy_metrics(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            return pd.DataFrame()
    
    def train_ml_model(self, training_data: pd.DataFrame) -> Dict:
        try:
            data = self.market_structure.identify_structure(training_data)
            data = self.volatility_analyzer.calculate_volatility(data)
            data = self.feature_engineer.create_features(data)
            
            metrics = self.ml_model.train(data)
            logger.info("ML model training completed")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return {}
    
    def _combine_signals(self, data: pd.DataFrame) -> pd.Series:
        ml_weight = self.config.get('signal_weights', {}).get('ml_weight', 0.5)
        trad_weight = self.config.get('signal_weights', {}).get('traditional_weight', 0.5)
        
        combined = (
            ml_weight * data['ml_signal'] +
            trad_weight * data['signal']
        )
        
        final = pd.Series(0, index=data.index)
        threshold = self.config.get('signal_threshold', 0.5)
        
        final[combined > threshold] = 1
        final[combined < -threshold] = -1
        
        return final
    
    def calculate_strategy_metrics(self, data: pd.DataFrame) -> Dict:
        try:
            trades = data[data['final_signal'] != 0].copy()
            trade_metrics = self.metrics_calculator.calculate_trade_metrics(trades)
            portfolio_metrics = self.metrics_calculator.calculate_portfolio_metrics(trades)
            ml_metrics = self._calculate_ml_metrics(data)
            
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
        try:
            ml_metrics = {}
            
            # Signal agreement between ML and traditional
            agreement = (data['ml_signal'] == data['signal']).mean()
            ml_metrics['ml_traditional_agreement'] = agreement
            
            # ML prediction accuracy
            returns = data['close'].pct_change()
            ml_correct = (
                ((data['ml_signal'] == 1) & (returns > 0)) |
                ((data['ml_signal'] == -1) & (returns < 0))
            ).mean()
            ml_metrics['ml_accuracy'] = ml_correct
            
            # ML contribution to final signals
            ml_contribution = (
                (data['final_signal'] == data['ml_signal']).sum() /
                (data['final_signal'] != 0).sum()
            )
            ml_metrics['ml_contribution'] = ml_contribution
            
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
        logger.info(f"ML Contribution: {self.metrics.get('ml_contribution', 0):.2%}")
