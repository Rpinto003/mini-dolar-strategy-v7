import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

def explore_database(db_path: str, table_name: str):
    """Explorar os dados do banco SQLite"""
    try:
        # Conectar ao banco
        conn = sqlite3.connect(db_path)
        
        # Verificar schema da tabela
        schema = pd.read_sql(f"PRAGMA table_info({table_name})", conn)
        logger.info("\nEsquema da tabela:")
        print(schema)
        
        # Verificar primeiras linhas
        data = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        logger.info("\nPrimeiras linhas:")
        print(data)
        
        # Verificar período dos dados
        period = pd.read_sql(f"""
            SELECT 
                MIN(time) as inicio,
                MAX(time) as fim,
                COUNT(*) as total_registros
            FROM {table_name}
        """, conn)
        logger.info("\nPeríodo dos dados:")
        print(period)
        
        # Verificar estatísticas básicas
        stats = pd.read_sql(f"""
            SELECT 
                MIN(close) as preco_min,
                MAX(close) as preco_max,
                AVG(close) as preco_medio,
                MIN(volume) as volume_min,
                MAX(volume) as volume_max,
                AVG(volume) as volume_medio
            FROM {table_name}
        """, conn)
        logger.info("\nEstatísticas básicas:")
        print(stats)
        
        # Verificar quantidade de registros por dia
        daily_count = pd.read_sql(f"""
            SELECT 
                date(time) as data,
                COUNT(*) as registros
            FROM {table_name}
            GROUP BY date(time)
            ORDER BY date(time)
            LIMIT 5
        """, conn)
        logger.info("\nRegistros por dia (primeiros 5 dias):")
        print(daily_count)
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Erro explorando dados: {str(e)}")

def main():
    # Configurar logging
    logger.add(
        "logs/explore_data_{time}.log",
        rotation="1 day",
        level="INFO"
    )
    
    # Parâmetros do banco
    db_path = "src/data/candles.db"
    table_name = "candles"
    
    logger.info(f"Explorando banco de dados: {db_path}")
    explore_database(db_path, table_name)

if __name__ == "__main__":
    main()