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
            df = self._add_momentum_features(df)
            df = self._add_volatility_features(df)
            df = self._add_volume_features(df)
            df = self._add_trend_features(df)
            return df.dropna()
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return data
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['roc_5'] = df['close'].pct_change(periods=5)
        df['roc_10'] = df['close'].pct_change(periods=10)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['rsi_divergence'] = df['rsi'] - df['rsi'].shift(5)
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        window = 20
        std = df['close'].rolling(window=window).std()
        ma = df['close'].rolling(window=window).mean()
        df['bb_width'] = (ma + (2 * std) - (ma - (2 * std))) / ma
        df['atr_ratio'] = df['atr'] / df['close']
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_price_corr'] = df['close'].rolling(window=10).corr(df['volume'])
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma_cross'] = np.where(
            df['sma_20'] > df['sma_50'], 1,
            np.where(df['sma_20'] < df['sma_50'], -1, 0)
        )
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['adx'] = self._calculate_adx(df)
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        return adx.ewm(alpha=1/period).mean()