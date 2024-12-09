import pandas as pd
import numpy as np
from loguru import logger

class VolatilityAnalyzer:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        logger.info(f"Initialized VolatilityAnalyzer with atr_period={atr_period}")
    
    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volatility metrics"""
        try:
            df = data.copy()
            
            # Calculate ATR
            df = self._calculate_atr(df)
            
            # Calculate Historical Volatility
            df = self._calculate_historical_volatility(df)
            
            # Calculate Volatility Bands
            df = self._calculate_volatility_bands(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return data
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        df['atr'] = tr.rolling(window=self.atr_period).mean()
        return df
    
    def _calculate_historical_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Historical Volatility"""
        # Calculate daily returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate historical volatility (20-day standard deviation of returns)
        df['historical_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        return df
    
    def _calculate_volatility_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volatility Bands"""
        # Calculate basis (20-day moving average)
        df['basis'] = df['close'].rolling(window=20).mean()
        
        # Calculate upper and lower bands (2 ATR from basis)
        df['upper_band'] = df['basis'] + (2 * df['atr'])
        df['lower_band'] = df['basis'] - (2 * df['atr'])
        
        return df