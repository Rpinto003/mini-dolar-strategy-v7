import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(start_date='2024-01-01', end_date='2024-12-31', freq='1min'):
    """Generate sample market data for testing"""
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    trading_dates = dates[dates.dayofweek < 5]  # Monday to Friday
    trading_dates = trading_dates[(trading_dates.hour >= 9) & (trading_dates.hour < 18)]  # Trading hours
    
    # Generate price data
    np.random.seed(42)  # For reproducibility
    price = 100 + np.random.randn(len(trading_dates)).cumsum() * 0.02
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': trading_dates,
        'open': price + np.random.randn(len(trading_dates)) * 0.01,
        'high': price + np.abs(np.random.randn(len(trading_dates)) * 0.015),
        'low': price - np.abs(np.random.randn(len(trading_dates)) * 0.015),
        'close': price + np.random.randn(len(trading_dates)) * 0.01,
        'volume': np.random.randint(100, 1000, len(trading_dates))
    })
    
    return data

def create_database(data, db_path='data/market_data.db', table_name='mini_dollar_futures'):
    """Create SQLite database with sample data"""
    # Create directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    
    # Save data to database
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Created database at {db_path} with {len(data)} records")

def main():
    # Generate sample data
    data = generate_sample_data()
    
    # Create database
    create_database(data)

if __name__ == '__main__':
    main()