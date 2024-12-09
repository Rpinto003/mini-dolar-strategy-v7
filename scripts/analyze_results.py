import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

def find_latest_results() -> tuple:
    """Find the latest results files"""
    output_dir = Path("output")
    if not output_dir.exists():
        return None, None
    
    # Find all results files
    results_files = list(output_dir.glob("test_backtest_results_*.csv"))
    metrics_files = list(output_dir.glob("test_backtest_metrics_*.csv"))
    
    if not results_files or not metrics_files:
        return None, None
    
    # Get latest files
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
    
    return latest_results, latest_metrics

def analyze_strategy_results(results_file: Path, metrics_file: Path):
    """Analyze and visualize strategy results"""
    # Create output directory for plots
    plots_dir = Path("analysis_output")
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    results = pd.read_csv(results_file, index_col=0, parse_dates=True)
    metrics = pd.read_csv(metrics_file, index_col=0)
    
    logger.info("Loaded results data:")
    logger.info(f"Period: {results.index[0]} to {results.index[-1]}")
    logger.info(f"Number of records: {len(results)}")
    
    # 1. Performance Metrics
    logger.info("\nPerformance Metrics:")
    for col in metrics.columns:
        value = metrics[col].iloc[0]
        if isinstance(value, (int, float)):
            logger.info(f"{col}: {value:.2%}")
        else:
            logger.info(f"{col}: {value}")
    
    # 2. Trade Analysis
    trades = results[results['final_signal'] != 0].copy()
    if len(trades) > 0:
        logger.info(f"\nTrade Analysis:")
        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"Average Trade Duration: {pd.Timedelta(trades.index.to_series().diff().mean())}")
        
        # Plot trade distribution
        plt.figure(figsize=(12, 6))
        plt.hist(trades['trade_return'] if 'trade_return' in trades.columns else [], bins=50)
        plt.title('Trade Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(plots_dir / 'trade_distribution.png')
        plt.close()
        
        # Plot trade timing
        plt.figure(figsize=(15, 6))
        plt.scatter(trades.index, trades['final_signal'], alpha=0.6)
        plt.title('Trade Entry Points')
        plt.xlabel('Date')
        plt.ylabel('Signal Direction')
        plt.grid(True)
        plt.savefig(plots_dir / 'trade_timing.png')
        plt.close()
    
    # 3. Equity Curve Analysis
    if 'equity' in results.columns:
        # Plot equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results['equity'])
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(plots_dir / 'equity_curve.png')
        plt.close()
        
        # Calculate and plot drawdown
        rolling_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - rolling_max) / rolling_max
        
        plt.figure(figsize=(15, 6))
        plt.plot(results.index, drawdown)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.savefig(plots_dir / 'drawdown.png')
        plt.close()
    
    # 4. Signal Analysis
    if all(col in results.columns for col in ['ml_signal', 'signal', 'final_signal']):
        # Signal agreement analysis
        signal_agreement = (results['ml_signal'] == results['signal']).mean()
        logger.info(f"\nSignal Analysis:")
        logger.info(f"ML-Traditional Signal Agreement: {signal_agreement:.2%}")
        
        # Plot signal distribution
        plt.figure(figsize=(10, 6))
        signal_counts = pd.DataFrame({
            'ML': results['ml_signal'].value_counts(),
            'Traditional': results['signal'].value_counts(),
            'Final': results['final_signal'].value_counts()
        }).fillna(0)
        
        signal_counts.plot(kind='bar')
        plt.title('Signal Distribution')
        plt.xlabel('Signal Type')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'signal_distribution.png')
        plt.close()
    
    # 5. Intraday Analysis
    if isinstance(results.index, pd.DatetimeIndex):
        # Trading activity by hour
        results['hour'] = results.index.hour
        trades_by_hour = trades.groupby('hour').size()
        
        plt.figure(figsize=(12, 6))
        trades_by_hour.plot(kind='bar')
        plt.title('Trading Activity by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Number of Trades')
        plt.grid(True)
        plt.savefig(plots_dir / 'trades_by_hour.png')
        plt.close()
    
    logger.info(f"\nAnalysis plots saved to {plots_dir}")

def main():
    # Configure logging
    logger.add(
        "logs/analysis_{time}.log",
        rotation="1 day",
        level="INFO"
    )
    
    try:
        # Find latest results
        results_file, metrics_file = find_latest_results()
        
        if results_file is None or metrics_file is None:
            logger.error("No results files found in output directory")
            return
        
        logger.info(f"Analyzing results from:")
        logger.info(f"Results: {results_file.name}")
        logger.info(f"Metrics: {metrics_file.name}")
        
        # Analyze results
        analyze_strategy_results(results_file, metrics_file)
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")

if __name__ == "__main__":
    main()