import yaml
from pathlib import Path
from loguru import logger

from src.strategy.enhanced_strategy import EnhancedStrategy
from src.data.loaders.market_data import MarketDataLoader

def load_config() -> dict:
    """Load strategy configuration from YAML file"""
    config_path = Path("config/strategy_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    # Configure logging
    logger.add(
        "logs/strategy_{time}.log",
        rotation="1 day",
        level="INFO"
    )
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        data_loader = MarketDataLoader(
            db_path=config['data']['db_path'],
            table_name=config['data']['table_name']
        )
        
        strategy = EnhancedStrategy(config=config['strategy'])
        
        # Load training data
        training_data = data_loader.get_minute_data(
            interval=config['data']['interval'],
            start_date=config['backtest']['start_date'],
            end_date=config['backtest']['end_date']
        )
        logger.info(f"Loaded {len(training_data)} data points for training")
        
        # Train ML model
        training_metrics = strategy.train_ml_model(training_data)
        logger.info(f"ML model training metrics: {training_metrics}")
        
        # Run strategy
        results = strategy.run(training_data)
        logger.info("Strategy execution completed successfully")
        
        # Log final metrics
        logger.info("Final Performance Metrics:")
        for key, value in strategy.metrics.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()