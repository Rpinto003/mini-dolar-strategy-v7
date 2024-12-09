import pandas as pd
import sqlite3
from datetime import datetime
from loguru import logger

class MarketDataLoader:
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        logger.info(f"Initialized MarketDataLoader with db={db_path}, table={table_name}")
    
    def get_minute_data(self, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data from SQLite database"""
        try:
            query = self._build_query(interval, start_date, end_date)
            
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(query, conn)
            
            logger.info(f"Loaded {len(data)} rows of market data")
            return self._process_data(data)
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()
    
    def _build_query(self, interval: str, start_date: str, end_date: str) -> str:
        """Build SQL query for data retrieval"""
        return f"""
        SELECT 
            datetime,
            open,
            high,
            low,
            close,
            volume
        FROM {self.table_name}
        WHERE 
            datetime BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY datetime ASC
        """
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and validate market data"""
        # Convert datetime string to datetime object
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        # Ensure all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Validate data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any remaining invalid data
        data = data.dropna()
        
        return data