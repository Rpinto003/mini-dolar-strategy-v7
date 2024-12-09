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
        self.performance_window = self.config.get('performance_window', 20)
        self.signal_performance = pd.DataFrame()
        logger.info("Strategy initialized with adaptive capabilities")
    
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
        try:
            df = data.copy()
            
            # Market Structure Analysis
            df = self.market_structure.identify_structure(df)
            
            # Volatility Analysis with Dynamic Windows
            df = self.volatility_analyzer.calculate_volatility(df)
            df['volatility_regime'] = self._determine_volatility_regime(df)
            
            # Enhanced Signal Generation
            df = self.signal_generator.generate_signals(df)
            df = self._add_volume_profile(df)
            
            # Feature Engineering
            df = self.feature_engineer.create_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()
    
    def _determine_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Determine market volatility regime"""
        vol = data['atr'] / data['close']
        vol_percentile = vol.rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        regime = pd.Series(index=data.index, data='normal')
        regime[vol_percentile > 0.8] = 'high'
        regime[vol_percentile < 0.2] = 'low'
        return regime
    
    def _add_volume_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile analysis"""
        df = data.copy()
        window = self.config.get('volume_window', 20)
        
        df['volume_ma'] = df['volume'].rolling(window=window).mean()
        df['vol_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['vol_ratio'] > self.config.get('high_volume_threshold', 1.5)
        
        return df
    
    def _calculate_dynamic_position_size(self, data: pd.DataFrame) -> pd.Series:
        """Calculate position size based on ATR and account size"""
        risk_per_trade = self.config.get('max_risk_per_trade', 0.01)
        account_size = self.risk_manager.get_current_capital()
        
        risk_amount = account_size * risk_per_trade
        position_size = risk_amount / (data['atr'] * self.config.get('atr_multiplier', 2))
        
        return position_size.round(2)
    
    def _update_signal_weights(self, data: pd.DataFrame):
        """Update signal weights based on recent performance"""
        if len(self.signal_performance) < self.performance_window:
            return self.config.get('signal_weights', {})
            
        ml_accuracy = self.signal_performance['ml_profit'].rolling(
            window=self.performance_window
        ).mean().iloc[-1]
        
        trad_accuracy = self.signal_performance['trad_profit'].rolling(
            window=self.performance_window
        ).mean().iloc[-1]
        
        total = ml_accuracy + trad_accuracy
        if total <= 0:
            return self.config.get('signal_weights', {})
            
        return {
            'ml_weight': ml_accuracy / total,
            'traditional_weight': trad_accuracy / total
        }
    
    def _combine_signals(self, data: pd.DataFrame) -> pd.Series:
        weights = self._update_signal_weights(data)
        ml_weight = weights.get('ml_weight', 0.4)
        trad_weight = weights.get('traditional_weight', 0.6)
        
        combined = pd.Series(0, index=data.index)
        mask = (data['ml_signal'] != 0) | (data['signal'] != 0)
        
        if mask.any():
            combined.loc[mask] = (
                ml_weight * data.loc[mask, 'ml_signal'].astype(float) +
                trad_weight * data.loc[mask, 'signal'].astype(float)
            )
        
        # Dynamic threshold based on volatility regime
        thresholds = {
            'high': self.config.get('high_vol_threshold', 0.7),
            'normal': self.config.get('signal_threshold', 0.6),
            'low': self.config.get('low_vol_threshold', 0.5)
        }
        
        final = pd.Series(0, index=data.index)
        for regime in thresholds:
            mask = data['volatility_regime'] == regime
            threshold = thresholds[regime]
            final.loc[mask & (combined > threshold)] = 1
            final.loc[mask & (combined < -threshold)] = -1
        
        return final
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting enhanced strategy execution")
        
        try:
            df = self.prepare_data(data)
            if df.empty:
                return pd.DataFrame()
            
            # ML Predictions with confidence
            if self.ml_model is not None:
                ml_signals, confidence = self.ml_model.predict_with_confidence(df)
                df['ml_signal'] = ml_signals
                df['ml_confidence'] = confidence
            else:
                df['ml_signal'] = 0
                df['ml_confidence'] = 0
            
            # Signal Combination and Position Sizing
            df['final_signal'] = self._combine_signals(df)
            df['position_size'] = self._calculate_dynamic_position_size(df)
            
            # Enhanced Risk Management
            df = self.risk_manager.apply_risk_management(df)
            
            # Update Performance Tracking
            self._update_performance_tracking(df)
            
            # Calculate Metrics
            self.calculate_strategy_metrics(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            return pd.DataFrame()
    
    def _update_performance_tracking(self, data: pd.DataFrame):
        """Track signal performance for weight updates"""
        returns = data['close'].pct_change().shift(-1)
        
        self.signal_performance = pd.concat([
            self.signal_performance,
            pd.DataFrame({
                'ml_profit': data['ml_signal'] * returns,
                'trad_profit': data['signal'] * returns
            })
        ]).tail(self.performance_window * 2)
    
    def calculate_strategy_metrics(self, data: pd.DataFrame) -> Dict:
        try:
            metrics = {}
            
            # Enhanced signal analysis
            signal_mask = (data['ml_signal'] != 0) & (data['signal'] != 0)
            if signal_mask.any():
                metrics['ml_traditional_agreement'] = (
                    data.loc[signal_mask, 'ml_signal'] == 
                    data.loc[signal_mask, 'signal']
                ).mean()
                
                # Add confidence-weighted metrics
                metrics['weighted_agreement'] = (
                    (data.loc[signal_mask, 'ml_signal'] == 
                     data.loc[signal_mask, 'signal']) *
                    data.loc[signal_mask, 'ml_confidence']
                ).mean()
            
            # Performance metrics by volatility regime
            for regime in ['high', 'normal', 'low']:
                mask = (data['volatility_regime'] == regime) & (data['final_signal'] != 0)
                if mask.any():
                    returns = data.loc[mask, 'close'].pct_change().shift(-1)
                    metrics[f'{regime}_vol_sharpe'] = (
                        returns.mean() / returns.std() * np.sqrt(252)
                    )
            
            self.metrics = metrics
            self._log_performance_metrics()
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _log_performance_metrics(self):
        logger.info("Enhanced Strategy Performance:")
        if not self.metrics:
            logger.info("No metrics available")
            return
            
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value}")
