import pandas as pd
import numpy as np
from typing import List
from loguru import logger

class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        logger.info("Initialized FeatureEngineer")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature set for machine learning model"""
        try:
            df = data.copy()
            
            # 1. Verificar features existentes
            required_features = [
                'rsi', 'macd', 'macd_hist', 'volume_ratio',
                'volume_trend', 'atr_ratio'
            ]
            
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            # 2. Features de momentum
            df = self._add_momentum_features(df)
            
            # 3. Features de volatilidade
            df = self._add_volatility_features(df)
            
            # 4. Features de volume
            df = self._add_volume_features(df)
            
            # 5. Features de tendência
            df = self._add_trend_features(df)
            
            logger.info(f"Created features: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return data
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators"""
        # ROC (Rate of Change)
        periods = [5, 10, 20]
        for p in periods:
            df[f'roc_{p}'] = df['close'].pct_change(periods=p)
        
        # RSI if not exists
        if 'rsi' not in df.columns:
            close_delta = df['close'].diff()
            gain = (close_delta.where(close_delta > 0, 0)).rolling(window=14).mean()
            loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD if not exists
        if 'macd' not in df.columns:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal_line']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # ATR ratio if not exists
        if 'atr_ratio' not in df.columns:
            tr = pd.DataFrame()
            tr['h-l'] = df['high'] - df['low']
            tr['h-pc'] = abs(df['high'] - df['close'].shift())
            tr['l-pc'] = abs(df['low'] - df['close'].shift())
            tr['tr'] = tr.max(axis=1)
            df['atr_ratio'] = tr['tr'].rolling(window=14).mean() / df['close']
        
        # Historical volatility
        returns = df['close'].pct_change()
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume ratio if not exists
        if 'volume_ratio' not in df.columns:
            vol_ma = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / vol_ma
        
        # Volume trend if not exists
        if 'volume_trend' not in df.columns:
            df['volume_trend'] = df['volume'].pct_change().rolling(window=5).mean()
        
        # Additional volume features
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        df['volume_roc'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=50).mean()
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        # Moving averages if not exist
        for period in [10, 20, 50]:
            if f'sma_{period}' not in df.columns:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Price relative to moving averages
        for period in [10, 20, 50]:
            df[f'price_sma{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Trend direction
        df['trend_short'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['trend_long'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Additional trend features
        df['price_momentum'] = df['close'] - df['close'].shift(10)
        df['trend_strength'] = df['trend_short'] * df['trend_long']
        
        return df