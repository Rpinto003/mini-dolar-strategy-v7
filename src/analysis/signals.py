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
            
            # 1. Cálculo dos indicadores técnicos
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_volume_ratio(df)
            df = self._calculate_trend(df)
            
            # 2. Geração de sinais
            df['signal'] = self._combine_signals(df)
            
            # 3. Registro dos preços de entrada
            df = self._register_entries(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return data
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        close_delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (close_delta.where(close_delta > 0, 0)).fillna(0)
        loss = (-close_delta.where(close_delta < 0, 0)).fillna(0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        # Calculate EMAs
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal_line']
        return df
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume ratio indicator"""
        # Volume moving average
        vol_ma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma
        
        # Volume trend (taxa de variação)
        df['volume_trend'] = df['volume'].pct_change().rolling(window=5).mean()
        
        return df
    
    def _calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        # ATR ratio (volatilidade relativa)
        if 'atr' in df.columns:
            df['atr_ratio'] = df['atr'] / df['close']
        else:
            # Calcular ATR se não existir
            high_low = df['high'] - df['low']
            high_pc = abs(df['high'] - df['close'].shift())
            low_pc = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            df['atr_ratio'] = atr / df['close']
        
        # Preço relativo às médias móveis
        if not 'sma_20' in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if not 'sma_50' in df.columns:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.Series:
        """Combine multiple indicators to generate signals"""
        signals = pd.Series(0, index=df.index)
        
        # Condições para compra
        long_condition = (
            (df['rsi'] < 40) &  # Sobrevendido
            (df['macd'] > df['signal_line']) &  # MACD cruzando para cima
            (df['volume_ratio'] > 1.2) &  # Volume acima da média
            (df['volume_trend'] > 0) &  # Volume crescente
            (df['price_sma20_ratio'] < 0.995)  # Preço abaixo da média
        )
        
        # Condições para venda
        short_condition = (
            (df['rsi'] > 60) &  # Sobrecomprado
            (df['macd'] < df['signal_line']) &  # MACD cruzando para baixo
            (df['volume_ratio'] > 1.2) &  # Volume acima da média
            (df['volume_trend'] < 0) &  # Volume decrescente
            (df['price_sma20_ratio'] > 1.005)  # Preço acima da média
        )
        
        # Aplicar sinais
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Espaçamento mínimo entre sinais
        min_bars = 20
        last_signal_idx = -min_bars - 1
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                if i - last_signal_idx <= min_bars:
                    signals.iloc[i] = 0
                else:
                    last_signal_idx = i
        
        return signals
    
    def _register_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Register entry prices and times for trades"""
        # Initialize entry columns
        df['entry_price'] = np.nan
        df['entry_time'] = pd.NaT
        
        # Update entry information for trades
        signal_mask = df['signal'] != 0
        df.loc[signal_mask, 'entry_price'] = df.loc[signal_mask, 'close']
        df.loc[signal_mask, 'entry_time'] = df.index[signal_mask]
        
        return df