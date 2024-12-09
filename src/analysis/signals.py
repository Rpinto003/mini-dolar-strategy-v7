import pandas as pd
import numpy as np
from loguru import logger

class SignalGenerator:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized SignalGenerator with risk_free_rate={risk_free_rate}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical analysis"""
        try:
            df = data.copy()
            
            # Calculate technical indicators
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_volume_ratio(df)
            
            # Generate signals based on combined indicators
            df['signal'] = self._combine_signals(df)
            
            # Add entry prices for trades
            df['entry_price'] = np.where(df['signal'] != 0, df['close'], np.nan)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return data
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        return df
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume ratio indicator"""
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.Series:
        """Combine multiple indicators to generate signals"""
        signals = pd.Series(0, index=df.index)
        
        # Long signals
        long_condition = (
            (df['rsi'] < 70) &
            (df['macd'] > df['signal_line']) &
            (df['volume_ratio'] > 1.2)
        )
        
        # Short signals
        short_condition = (
            (df['rsi'] > 30) &
            (df['macd'] < df['signal_line']) &
            (df['volume_ratio'] > 1.2)
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals