import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger

class MetricsCalculator:
    def __init__(self):
        logger.info("Initialized MetricsCalculator")
    
    def calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level performance metrics"""
        try:
            metrics = {}
            
            # Basic trade metrics
            metrics['total_trades'] = len(trades)
            metrics['long_trades'] = len(trades[trades['final_signal'] == 1])
            metrics['short_trades'] = len(trades[trades['final_signal'] == -1])
            
            # Calculate trade returns
            trades['trade_return'] = trades.apply(
                lambda x: (
                    (x['close'] - x['entry_price']) / x['entry_price'] if x['final_signal'] == 1
                    else (x['entry_price'] - x['close']) / x['entry_price']
                ) if x['final_signal'] != 0 and pd.notnull(x['entry_price']) else 0,
                axis=1
            )
            
            # Calculate win rate
            winning_trades = trades[trades['trade_return'] > 0]
            metrics['winning_trades'] = len(winning_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            # Calculate return metrics
            metrics['avg_return'] = trades['trade_return'].mean() if len(trades) > 0 else 0
            metrics['median_return'] = trades['trade_return'].median() if len(trades) > 0 else 0
            metrics['std_return'] = trades['trade_return'].std() if len(trades) > 0 else 0
            
            # Calculate risk metrics
            metrics['max_drawdown'] = self._calculate_max_drawdown(trades)
            metrics['profit_factor'] = self._calculate_profit_factor(trades)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")
            return {}
    
    def calculate_portfolio_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate portfolio-level performance metrics"""
        try:
            metrics = {}
            
            if 'equity' not in trades.columns or len(trades) == 0:
                return metrics
            
            # Calculate equity curve
            equity_curve = trades['equity']
            
            # Calculate returns
            returns = equity_curve.pct_change().fillna(0)
            
            # Basic portfolio metrics
            metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            metrics['annualized_return'] = self._calculate_annualized_return(returns)
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            
            # Drawdown metrics
            metrics['max_drawdown'] = self._calculate_max_drawdown(equity_curve)
            metrics['avg_drawdown'] = self._calculate_avg_drawdown(equity_curve)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) == 0:
            return 0
            
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return abs(drawdown.min()) if not pd.isna(drawdown.min()) else 0
    
    def _calculate_avg_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown"""
        if len(equity_curve) == 0:
            return 0
            
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        negative_drawdowns = drawdowns[drawdowns < 0]
        return abs(negative_drawdowns.mean()) if len(negative_drawdowns) > 0 else 0
    
    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor"""
        if len(trades) == 0 or 'trade_return' not in trades.columns:
            return 0
            
        winning_trades = trades[trades['trade_return'] > 0]
        losing_trades = trades[trades['trade_return'] < 0]
        
        gross_profit = winning_trades['trade_return'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['trade_return'].sum()) if len(losing_trades) > 0 else 0
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
            
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return_std = excess_returns.std()
        
        if return_std == 0 or pd.isna(return_std):
            return 0
            
        return np.sqrt(252) * (excess_returns.mean() / return_std)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
            
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0 or pd.isna(downside_std):
            return 0
            
        return np.sqrt(252) * (excess_returns.mean() / downside_std)
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0
            
        total_return = (1 + returns).prod()
        years = len(returns) / 252  # Assuming 252 trading days per year
        
        if years == 0:
            return 0
            
        return (total_return ** (1 / years)) - 1