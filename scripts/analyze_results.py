import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

def analyze_strategy_results(results: pd.DataFrame):
    """Analyze and visualize strategy results"""
    # Create output directory
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Basic Performance Metrics
    logger.info("Calculating performance metrics...")
    
    # Trade statistics
    trades = results[results['final_signal'] != 0].copy()
    n_trades = len(trades)
    if n_trades > 0:
        winning_trades = trades[trades['trade_return'] > 0]
        win_rate = len(winning_trades) / n_trades
        avg_return = trades['trade_return'].mean()
        
        logger.info(f"Total Trades: {n_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Return per Trade: {avg_return:.2%}")
    
    # Equity curve analysis
    if 'equity' in results.columns:
        initial_equity = results['equity'].iloc[0]
        final_equity = results['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        logger.info(f"Total Return: {total_return:.2%}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['equity'])
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(output_dir / 'equity_curve.png')
        plt.close()
    
    # 2. Trade Analysis
    if n_trades > 0:
        # Trade return distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(trades['trade_return'], bins=50)
        plt.title('Trade Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'trade_distribution.png')
        plt.close()
        
        # Trade duration analysis
        if 'entry_time' in trades.columns:
            trades['duration'] = pd.to_datetime(trades.index) - pd.to_datetime(trades['entry_time'])
            avg_duration = trades['duration'].mean()
            logger.info(f"Average Trade Duration: {avg_duration}")
    
    # 3. Signal Analysis
    if 'ml_signal' in results.columns and 'signal' in results.columns:
        # Signal agreement analysis
        agreement = (results['ml_signal'] == results['signal']).mean()
        logger.info(f"ML-Traditional Signal Agreement: {agreement:.2%}")
        
        # Signal contribution
        signal_counts = pd.DataFrame({
            'ML': results['ml_signal'].value_counts(),
            'Traditional': results['signal'].value_counts(),
            'Final': results['final_signal'].value_counts()
        }).fillna(0)
        
        plt.figure(figsize=(10, 6))
        signal_counts.plot(kind='bar')
        plt.title('Signal Distribution')
        plt.xlabel('Signal Type')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'signal_distribution.png')
        plt.close()
    
    # 4. Market Analysis
    if all(col in results.columns for col in ['open', 'high', 'low', 'close']):
        # Daily volatility
        results['daily_returns'] = results['close'].pct_change()
        volatility = results['daily_returns'].std() * np.sqrt(252)
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        
        # Trading activity by hour
        results['hour'] = results.index.hour
        trade_by_hour = results[results['final_signal'] != 0].groupby('hour').size()
        
        plt.figure(figsize=(12, 6))
        trade_by_hour.plot(kind='bar')
        plt.title('Trading Activity by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Number of Trades')
        plt.savefig(output_dir / 'trade_by_hour.png')
        plt.close()
    
    # 5. Risk Analysis
    if 'equity' in results.columns:
        # Calculate drawdown
        rolling_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        drawdown.plot()
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.savefig(output_dir / 'drawdown.png')
        plt.close()
    
    logger.info(f"Analysis results saved to {output_dir}")

def main():
    # Configure logging
    logger.add(
        "logs/analysis_{time}.log",
        rotation="1 day",
        level="INFO"
    )
    
    # Load backtest results
    try:
        # You can modify this to load results from your preferred source
        results = pd.read_csv('backtest_results.csv', index_col='time', parse_dates=True)
        analyze_strategy_results(results)
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")

if __name__ == "__main__":
    main()