import pandas as pd
import numpy as np

class MarketRegimeDetector:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        vol = data['atr'] / data['close']
        vol_rank = vol.rolling(window=self.lookback_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        trend = self._calculate_trend_strength(data)
        
        regime = pd.Series(index=data.index, data='ranging')
        regime[vol_rank > 0.8] = 'volatile'
        regime[(vol_rank < 0.2) & (trend > 0.7)] = 'trending_up'
        regime[(vol_rank < 0.2) & (trend < -0.7)] = 'trending_down'
        
        return regime
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        ma_fast = data['close'].rolling(window=20).mean()
        ma_slow = data['close'].rolling(window=50).mean()
        
        trend = (ma_fast - ma_slow) / data['close']
        return trend.rolling(window=self.lookback_period).mean()