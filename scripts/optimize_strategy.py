import yaml
from pathlib import Path
from loguru import logger
import pandas as pd

from src.strategy.enhanced_strategy import EnhancedStrategy
from src.data.loaders.market_data import MarketDataLoader
from src.optimization.hyperparameters import HyperparameterOptimizer

def load_config() -> dict:
    """Load strategy configuration from YAML file"""
    config_path = Path("config/strategy_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_optimization_results(params: dict, metrics: dict):
    """Save optimization results"""
    # Create output directory
    output_dir = Path("output/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters
    params_df = pd.DataFrame([params])
    params_df.to_csv(output_dir / "best_parameters.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "best_metrics.csv")
    
    logger.info(f"Optimization results saved to {output_dir}")

def update_config_file(best_params: dict):
    """Update strategy configuration file with best parameters"""
    config_path = Path("config/strategy_config.yaml")
    
    # Load current config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update with best parameters
    for param_name, value in best_params.items():
        group, param = param_name.split('.')
        if group not in config:
            config[group] = {}
        
        if '.' in param:  # For nested parameters
            subgroup, subparam = param.split('.')
            if subgroup not in config[group]:
                config[group][subgroup] = {}
            config[group][subgroup][subparam] = value
        else:
            config[group][param] = value
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration file updated with best parameters")

def main():
    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "optimization_{time}.log",
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
        
        if len(data) < 1000:
            logger.error("Insufficient data for optimization")
            return
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer()
        
        # Run optimization
        logger.info("Starting hyperparameter optimization...")
        best_params, best_metrics = optimizer.optimize(
            strategy=strategy,
            data=data,
            metric='sharpe_ratio'
        )
        
        if best_params:
            logger.info("Optimization completed successfully")
            logger.info("\nBest Parameters:")
            for param, value in best_params.items():
                logger.info(f"{param}: {value}")
            
            logger.info("\nBest Metrics:")
            for metric, value in best_metrics.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.2%}")
                else:
                    logger.info(f"{metric}: {value}")
            
            # Save results
            save_optimization_results(best_params, best_metrics)
            
            # Update configuration
            update_config_file(best_params)
        else:
            logger.error("Optimization failed to find better parameters")
        
    except Exception as e:
        logger.error(f"Error in optimization process: {str(e)}")
        raise

if __name__ == "__main__":
    main()