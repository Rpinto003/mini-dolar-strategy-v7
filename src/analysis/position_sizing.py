import pandas as pd
from typing import Dict

class PositionSizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.default_risk = config.get('default_risk', 0.01)
        self.volatility_adjustments = {
            'volatile': 0.5,
            'ranging': 1.0,
            'trending_up': 1.5,
            'trending_down': 1.5
        }
    
    def calculate_position_size(self, capital: float, atr: float, regime: str) -> float:
        risk_amount = capital * self.default_risk
        vol_adj = self.volatility_adjustments.get(regime, 1.0)
        
        position_size = (risk_amount * vol_adj) / (atr * 2)
        return round(position_size, 2)
    
    def apply_position_sizing(self, data: pd.DataFrame, capital: float) -> pd.Series:
        return pd.Series([
            self.calculate_position_size(capital, row['atr'], row['market_regime'])
            for _, row in data.iterrows()
        ], index=data.index)