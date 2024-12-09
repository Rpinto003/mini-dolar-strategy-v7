import yaml
from pathlib import Path
from loguru import logger
import pandas as pd

from src.strategy.enhanced_strategy import EnhancedStrategy
from src.data.loaders.market_data import MarketDataLoader

def load_config() -> dict:
    """Load strategy configuration from YAML file"""
    config_path = Path("config/strategy_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_results(results: pd.DataFrame, metrics: dict, prefix: str = ""):
    """Save backtest results and metrics"""
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Ensure results have datetime index
        if not isinstance(results.index, pd.DatetimeIndex):
            results.index = pd.to_datetime(results.index)
        
        # Save results with date in filename
        date_str = results.index[0].strftime("%Y%m%d")
        results_file = output_dir / f"{prefix}backtest_results_{date_str}.csv"
        metrics_file = output_dir / f"{prefix}backtest_metrics_{date_str}.csv"
        
        # Save results
        results.to_csv(results_file)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_file)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Results file: {results_file.name}")
        logger.info(f"Metrics file: {metrics_file.name}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def main():
    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "strategy_{time}.log",
        rotation="1 day",
        level="INFO"
    )
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded")
        
        # Initialize components
        data_loader = MarketDataLoader(
            db_path=config['data']['db_path'],
            table_name=config['data']['table_name']
        )
        
        strategy = EnhancedStrategy(config=config['strategy'])
        
        # Load data
        data = data_loader.get_minute_data(
            interval=config['data']['interval'],
            start_date=config['backtest']['start_date'],
            end_date=config['backtest']['end_date']
        )
        logger.info(f"Loaded {len(data)} data points")
        
        if len(data) < 100:
            logger.error("Insufficient data for analysis")
            return
        
        # Split data for training and testing
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Train ML model
        logger.info("Training ML model...")
        training_metrics = strategy.train_ml_model(train_data)
        if training_metrics:
            logger.info(f"Training metrics: {training_metrics}")
            save_results(train_data, training_metrics, prefix="train_")
        
        # Run strategy on test data
        logger.info("Running strategy on test data...")
        results = strategy.run(test_data)
        
        if not results.empty and strategy.metrics:
            # Log performance metrics
            logger.info("\nStrategy Performance:")
            for key, value in strategy.metrics.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.2%}")
                else:
                    logger.info(f"{key}: {value}")
            
            # Save results
            save_results(results, strategy.metrics, prefix="test_")
        else:
            logger.error("No results generated")
        
    except Exception as e:
        logger.error(f"Error in strategy execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()