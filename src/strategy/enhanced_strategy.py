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
            max_risk_per_trade=self.config.get('max_risk_per_trade', 0.005)
        )
        self.metrics_calculator = MetricsCalculator()
        
        # ML components
        self.feature_engineer = FeatureEngineer(config=self.config.get('feature_engineering', {}))
        self.ml_model = MLModel(config=self.config.get('ml_model', {}))
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara os dados com todos os indicadores e features"""
        try:
            df = data.copy()
            
            # 1. Market Structure Analysis
            df = self.market_structure.identify_structure(df)
            logger.info("Market structure analysis completed")
            
            # 2. Volatility Analysis
            df = self.volatility_analyzer.calculate_volatility(df)
            logger.info("Volatility analysis completed")
            
            # 3. Signal Generation
            df = self.signal_generator.generate_signals(df)
            logger.info("Signal generation completed")
            
            # 4. Feature Engineering
            df = self.feature_engineer.create_features(df)
            logger.info("Feature engineering completed")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting strategy execution with ML")
        
        try:
            # 1. Preparação dos dados
            df = self.prepare_data(data)
            if df.empty:
                return pd.DataFrame()
            
            # 2. ML Predictions
            if self.ml_model is not None:
                ml_signals = self.ml_model.predict(df)
                df['ml_signal'] = ml_signals
            else:
                df['ml_signal'] = 0
            
            # 3. Signal Combination
            df['final_signal'] = self._combine_signals(df)
            
            # 4. Risk Management
            df = self.risk_manager.apply_risk_management(df)
            
            # 5. Calculate Metrics
            self.calculate_strategy_metrics(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            return pd.DataFrame()
    
    def train_ml_model(self, training_data: pd.DataFrame) -> Dict:
        try:
            # 1. Preparação dos dados
            df = self.prepare_data(training_data)
            if df.empty:
                return {}
                
            logger.info(f"Prepared {len(df)} samples for training")
            logger.info(f"Available features: {list(df.columns)}")
            
            # 2. Treinamento
            metrics = self.ml_model.train(df)
            logger.info("ML model training completed")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return {}
    
    def _combine_signals(self, data: pd.DataFrame) -> pd.Series:
        ml_weight = self.config.get('signal_weights', {}).get('ml_weight', 0.4)
        trad_weight = self.config.get('signal_weights', {}).get('traditional_weight', 0.6)
        
        combined = pd.Series(0, index=data.index)
        mask = (data['ml_signal'] != 0) | (data['signal'] != 0)
        
        if mask.any():
            combined.loc[mask] = (
                ml_weight * data.loc[mask, 'ml_signal'].astype(float) +
                trad_weight * data.loc[mask, 'signal'].astype(float)
            )
        
        final = pd.Series(0, index=data.index)
        threshold = self.config.get('signal_threshold', 0.6)
        
        final[combined > threshold] = 1
        final[combined < -threshold] = -1
        
        return final
    
    def calculate_strategy_metrics(self, data: pd.DataFrame) -> Dict:
        try:
            metrics = {}
            
            # Signal analysis
            signal_mask = (data['ml_signal'] != 0) & (data['signal'] != 0)
            if signal_mask.any():
                metrics['ml_traditional_agreement'] = (
                    data.loc[signal_mask, 'ml_signal'] == 
                    data.loc[signal_mask, 'signal']
                ).mean()
            else:
                metrics['ml_traditional_agreement'] = 0
            
            # ML accuracy
            ml_mask = data['ml_signal'] != 0
            if ml_mask.any():
                returns = data['close'].pct_change().shift(-1)
                metrics['ml_accuracy'] = (
                    ((data['ml_signal'] == 1) & (returns > 0)) |
                    ((data['ml_signal'] == -1) & (returns < 0))
                ).mean()
            else:
                metrics['ml_accuracy'] = 0
            
            # ML contribution
            final_mask = data['final_signal'] != 0
            if final_mask.any():
                metrics['ml_contribution'] = (
                    (data.loc[final_mask, 'final_signal'] == 
                     data.loc[final_mask, 'ml_signal']).sum() /
                    final_mask.sum()
                )
            else:
                metrics['ml_contribution'] = 0
            
            self.metrics = metrics
            self._log_performance_metrics()
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _log_performance_metrics(self):
        """Log key performance metrics"""
        logger.info("Strategy Performance:")
        if not self.metrics:
            logger.info("No metrics available")
            return
            
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value}")