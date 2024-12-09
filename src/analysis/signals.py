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
            
            # Add entry prices and times for trades
            df['entry_price'] = np.nan
            df['entry_time'] = pd.NaT
            
            # Update entry information
            signal_mask = df['signal'] != 0
            df.loc[signal_mask, 'entry_price'] = df.loc[signal_mask, 'close']
            df.loc[signal_mask, 'entry_time'] = df.index[signal_mask]
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return data
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        close_delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (close_delta.where(close_delta > 0, 0)).rolling(window=period).mean()
        loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=12, adjust=False).mean()
        slow_ema = df['close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD and Signal line
        df['macd'] = fast_ema - slow_ema
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal_line']
        
        return df
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume ratio indicator"""
        # Calculate volume moving average
        vol_ma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma
        
        # Calculate volume trend
        df['volume_trend'] = df['volume'].rolling(window=5).mean().pct_change()
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.Series:
        """Combine multiple indicators to generate signals"""
        signals = pd.Series(0, index=df.index)
        
        # Define conditions for long signals
        long_condition = (
            (df['rsi'] < 40) &  # Oversold condition
            (df['macd'] > df['signal_line']) &  # MACD crossover
            (df['volume_ratio'] > 1.2) &  # Above average volume
            (df['volume_trend'] > 0)  # Increasing volume
        )
        
        # Define conditions for short signals
        short_condition = (
            (df['rsi'] > 60) &  # Overbought condition
            (df['macd'] < df['signal_line']) &  # MACD crossover
            (df['volume_ratio'] > 1.2) &  # Above average volume
            (df['volume_trend'] < 0)  # Decreasing volume
        )
        
        # Apply signals
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Add minimum spacing between signals
        min_bars_between_signals = 20
        last_signal_idx = -min_bars_between_signals - 1
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                if i - last_signal_idx <= min_bars_between_signals:
                    signals.iloc[i] = 0
                else:
                    last_signal_idx = i
        
        return signals