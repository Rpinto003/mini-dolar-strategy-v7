import pandas as pd
import numpy as np
from loguru import logger

class MarketStructure:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        logger.info(f"Initialized MarketStructure with lookback_period={lookback_period}")
    
    def identify_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure patterns and trends"""
        try:
            df = data.copy()
            
            # Calculate key moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate trend direction
            df['trend_direction'] = np.where(
                df['sma_20'] > df['sma_50'],
                1,  # Uptrend
                np.where(
                    df['sma_20'] < df['sma_50'],
                    -1,  # Downtrend
                    0  # Sideways
                )
            )
            
            # Identify swing highs and lows
            df['swing_high'] = self._identify_swing_highs(df)
            df['swing_low'] = self._identify_swing_lows(df)
            
            # Calculate price position relative to recent range
            df['price_position'] = self._calculate_price_position(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return data
    
    def _identify_swing_highs(self, df: pd.DataFrame) -> pd.Series:
        """Identify swing high points in the price series"""
        highs = pd.Series(0, index=df.index)
        for i in range(self.lookback_period, len(df)-self.lookback_period):
            if df['high'].iloc[i] == df['high'].iloc[i-self.lookback_period:i+self.lookback_period+1].max():
                highs.iloc[i] = 1
        return highs
    
    def _identify_swing_lows(self, df: pd.DataFrame) -> pd.Series:
        """Identify swing low points in the price series"""
        lows = pd.Series(0, index=df.index)
        for i in range(self.lookback_period, len(df)-self.lookback_period):
            if df['low'].iloc[i] == df['low'].iloc[i-self.lookback_period:i+self.lookback_period+1].min():
                lows.iloc[i] = 1
        return lows
    
    def _calculate_price_position(self, df: pd.DataFrame) -> pd.Series:
        """Calculate relative price position within recent range"""
        high = df['high'].rolling(window=self.lookback_period).max()
        low = df['low'].rolling(window=self.lookback_period).min()
        position = (df['close'] - low) / (high - low)
        return position