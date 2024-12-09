import pandas as pd
import numpy as np
from loguru import logger

class RiskManager:
    def __init__(self, initial_capital: float = 100000, max_risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        logger.info(f"Initialized RiskManager with capital={initial_capital}, max_risk={max_risk_per_trade}")
    
    def apply_risk_management(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk management rules to trading signals"""
        try:
            df = data.copy()
            
            # Calculate position sizes
            df = self._calculate_position_sizes(df)
            
            # Apply stop losses
            df = self._apply_stop_losses(df)
            
            # Track equity curve
            df = self._track_equity(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in risk management: {str(e)}")
            return data
    
    def _calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on risk parameters"""
        # Only process rows with active signals
        signal_mask = df['signal'] != 0
        
        if signal_mask.any():
            # Calculate risk amount per trade
            risk_amount = self.current_capital * self.max_risk_per_trade
            
            # Calculate position size based on ATR for stop loss
            df.loc[signal_mask, 'position_size'] = risk_amount / (df.loc[signal_mask, 'atr'] * 2)
            
            # Adjust for remaining capital
            df.loc[signal_mask, 'position_size'] = df.loc[signal_mask, 'position_size'].clip(
                upper=self.current_capital / df.loc[signal_mask, 'close']
            )
        else:
            df['position_size'] = 0
        
        return df
    
    def _apply_stop_losses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stop loss levels to positions"""
        # Calculate stop loss levels
        df['stop_loss'] = np.nan
        
        # Long positions
        long_mask = df['signal'] == 1
        if long_mask.any():
            df.loc[long_mask, 'stop_loss'] = df.loc[long_mask, 'close'] - (2 * df.loc[long_mask, 'atr'])
            
        # Short positions
        short_mask = df['signal'] == -1
        if short_mask.any():
            df.loc[short_mask, 'stop_loss'] = df.loc[short_mask, 'close'] + (2 * df.loc[short_mask, 'atr'])
        
        return df
    
    def _track_equity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track equity curve and update capital"""
        # Initialize equity and trade tracking
        df['equity'] = self.initial_capital
        current_position = 0
        entry_price = 0
        current_size = 0
        
        # Create view for better performance
        equity_values = np.full(len(df), self.initial_capital, dtype=float)
        
        for i in range(1, len(df)):
            # Check for new positions
            if df['signal'].iloc[i] != 0 and current_position == 0:
                current_position = df['signal'].iloc[i]
                entry_price = df['close'].iloc[i]
                current_size = df['position_size'].iloc[i]
                equity_values[i] = equity_values[i-1]
            
            # Check for position exit
            elif current_position != 0:
                # Calculate P&L
                if current_position == 1:  # Long position
                    pnl = (df['close'].iloc[i] - entry_price) * current_size
                else:  # Short position
                    pnl = (entry_price - df['close'].iloc[i]) * current_size
                
                # Update equity
                equity_values[i] = equity_values[i-1] + pnl
                self.current_capital = equity_values[i]
                
                # Reset position if signal changes
                if df['signal'].iloc[i] != current_position:
                    current_position = 0
                    current_size = 0
            else:
                equity_values[i] = equity_values[i-1]
        
        # Update equity column efficiently
        df['equity'] = equity_values
        
        return df