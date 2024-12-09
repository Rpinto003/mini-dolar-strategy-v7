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
            
            # Technical indicators
            df = self._add_momentum_features(df)
            df = self._add_volatility_features(df)
            df = self._add_volume_features(df)
            df = self._add_trend_features(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return data
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators"""
        # ROC (Rate of Change)
        df['roc_5'] = df['close'].pct_change(periods=5)
        df['roc_10'] = df['close'].pct_change(periods=10)
        df['roc_20'] = df['close'].pct_change(periods=20)
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # RSI variations
        df['rsi_divergence'] = df['rsi'] - df['rsi'].shift(5)
        df['rsi_trend'] = df['rsi'].rolling(window=10).mean()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Bollinger Bands
        window = 20
        std = df['close'].rolling(window=window).std()
        ma = df['close'].rolling(window=window).mean()
        df['bb_upper'] = ma + (2 * std)
        df['bb_lower'] = ma - (2 * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
        
        # ATR variations
        df['atr_ratio'] = df['atr'] / df['close']
        df['atr_ma_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume momentum
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        
        # Volume moving averages
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        
        # Volume price correlation
        df['volume_price_corr'] = df['close'].rolling(window=10).corr(df['volume'])
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        # Moving average crossovers
        df['sma_cross'] = np.where(
            df['sma_20'] > df['sma_50'],
            1,
            np.where(df['sma_20'] < df['sma_50'], -1, 0)
        )
        
        # Price relative to moving averages
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        
        # Trend strength
        df['adx'] = self._calculate_adx(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        adx_smooth = adx.ewm(alpha=1/period).mean()
        
        return adx_smooth