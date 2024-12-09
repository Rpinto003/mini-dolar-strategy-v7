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
            metrics['long_trades'] = len(trades[trades['signal'] == 1])
            metrics['short_trades'] = len(trades[trades['signal'] == -1])
            
            # Calculate trade returns
            trades['trade_return'] = trades.apply(
                lambda x: (
                    (x['close'] - x['entry_price']) / x['entry_price'] if x['signal'] == 1
                    else (x['entry_price'] - x['close']) / x['entry_price']
                ),
                axis=1
            )
            
            # Calculate win rate
            winning_trades = trades[trades['trade_return'] > 0]
            metrics['winning_trades'] = len(winning_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            # Calculate return metrics
            metrics['avg_return'] = trades['trade_return'].mean()
            metrics['median_return'] = trades['trade_return'].median()
            metrics['std_return'] = trades['trade_return'].std()
            
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
            
            # Calculate equity curve
            equity_curve = trades['equity']
            
            # Calculate returns
            returns = equity_curve.pct_change().dropna()
            
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
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_avg_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return drawdowns[drawdowns < 0].mean()
    
    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor"""
        winning_trades = trades[trades['trade_return'] > 0]
        losing_trades = trades[trades['trade_return'] < 0]
        
        gross_profit = winning_trades['trade_return'].sum()
        gross_loss = abs(losing_trades['trade_return'].sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        return np.sqrt(252) * (excess_returns.mean() / downside_std) if downside_std != 0 else 0
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_return = (1 + returns).prod()
        years = len(returns) / 252  # Assuming 252 trading days per year
        return (total_return ** (1 / years)) - 1