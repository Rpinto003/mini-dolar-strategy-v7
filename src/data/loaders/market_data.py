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
        """Carregar dados do mercado do banco SQLite"""
        try:
            query = self._build_query(interval, start_date, end_date)
            
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(query, conn)
            
            # Converte a coluna time para datetime
            data['time'] = pd.to_datetime(data['time'])
            data.set_index('time', inplace=True)
            
            logger.info(f"Carregados {len(data)} registros de {start_date} até {end_date}")
            return self._process_data(data)
            
        except Exception as e:
            logger.error(f"Erro carregando dados: {str(e)}")
            return pd.DataFrame()
    
    def _build_query(self, interval: str, start_date: str, end_date: str) -> str:
        """Construir query SQL para buscar dados"""
        return f"""
        SELECT 
            time,
            open,
            high,
            low,
            close,
            volume
        FROM {self.table_name}
        WHERE 
            time BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY time ASC
        """
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processar e validar dados do mercado"""
        # Verificar colunas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Colunas faltando: {missing_columns}")
            return pd.DataFrame()
        
        # Remover linhas com valores ausentes
        data = data.dropna()
        
        # Validar tipos de dados
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remover dados inválidos restantes
        data = data.dropna()
        
        # Ordenar por índice temporal
        data = data.sort_index()
        
        return data