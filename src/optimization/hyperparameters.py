import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

class HyperparameterOptimizer:
    def __init__(self):
        # Define os espaços de busca para cada parâmetro
        self.param_grid = {
            # Parâmetros do ML Model
            'ml_model': {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 8],
                'min_samples_leaf': [50, 100, 200],
                'probability_threshold': [0.6, 0.7, 0.8]
            },
            
            # Parâmetros de indicadores técnicos
            'indicators': {
                'rsi_period': [9, 14, 21],
                'macd_fast': [8, 12, 16],
                'macd_slow': [21, 26, 34],
                'volume_ma_period': [10, 20, 30]
            },
            
            # Parâmetros da estratégia
            'strategy': {
                'signal_weights.ml_weight': [0.3, 0.4, 0.5],
                'signal_weights.traditional_weight': [0.5, 0.6, 0.7],
                'signal_threshold': [0.5, 0.6, 0.7]
            },
            
            # Parâmetros de risk management
            'risk': {
                'max_risk_per_trade': [0.003, 0.005, 0.008],
                'position_size_atr_multiple': [1.0, 1.5, 2.0],
                'min_spacing_bars': [15, 20, 30]
            }
        }
    
    def optimize(self, strategy, data: pd.DataFrame, metric: str = 'sharpe_ratio') -> Tuple[Dict, Dict]:
        """Otimiza hiperparâmetros usando grid search"""
        try:
            # Divide os dados em treino e validação
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            val_data = data[train_size:]
            
            best_score = float('-inf')
            best_params = None
            best_metrics = None
            
            # Gera todas as combinações de parâmetros
            param_combinations = self._generate_param_combinations()
            total_combinations = len(param_combinations)
            
            logger.info(f"Starting grid search with {total_combinations} combinations")
            
            for i, params in enumerate(param_combinations, 1):
                try:
                    # Atualiza configuração
                    strategy = self._update_strategy_config(strategy, params)
                    
                    # Treina modelo
                    train_metrics = strategy.train_ml_model(train_data)
                    
                    # Avalia no conjunto de validação
                    results = strategy.run(val_data)
                    val_metrics = strategy.metrics
                    
                    # Calcula score
                    if metric in val_metrics:
                        current_score = val_metrics[metric]
                    else:
                        current_score = self._calculate_composite_score(val_metrics)
                    
                    logger.info(f"Combination {i}/{total_combinations} - Score: {current_score:.4f}")
                    
                    # Atualiza melhor resultado
                    if current_score > best_score:
                        best_score = current_score
                        best_params = params.copy()
                        best_metrics = val_metrics.copy()
                        
                        logger.info("New best parameters found!")
                        logger.info(f"Parameters: {best_params}")
                        logger.info(f"Score: {best_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating combination {i}: {str(e)}")
                    continue
            
            return best_params, best_metrics
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {}, {}
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Gera todas as combinações possíveis de parâmetros"""
        combinations = []
        
        # Para cada grupo de parâmetros
        for group, params in self.param_grid.items():
            group_combinations = []
            param_names = list(params.keys())
            param_values = list(params.values())
            
            # Gera combinações para este grupo
            for instance in itertools.product(*param_values):
                param_dict = {}
                for name, value in zip(param_names, instance):
                    param_dict[f"{group}.{name}"] = value
                group_combinations.append(param_dict)
            
            combinations.extend(group_combinations)
        
        return combinations
    
    def _update_strategy_config(self, strategy, params: Dict) -> object:
        """Atualiza configuração da estratégia com novos parâmetros"""
        try:
            for param_name, value in params.items():
                group, name = param_name.split('.')
                
                if group not in strategy.config:
                    strategy.config[group] = {}
                
                if '.' in name:  # Para nested dicts como signal_weights
                    subgroup, subname = name.split('.')
                    if subgroup not in strategy.config[group]:
                        strategy.config[group][subgroup] = {}
                    strategy.config[group][subgroup][subname] = value
                else:
                    strategy.config[group][name] = value
            
            # Reinicializa componentes com novos parâmetros
            strategy.initialize_components()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error updating strategy config: {str(e)}")
            return strategy
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calcula um score composto combinando várias métricas"""
        score = 0.0
        
        if 'sharpe_ratio' in metrics:
            score += metrics['sharpe_ratio'] * 0.4
        
        if 'win_rate' in metrics:
            score += metrics['win_rate'] * 0.3
        
        if 'profit_factor' in metrics:
            score += metrics['profit_factor'] * 0.2
        
        if 'ml_accuracy' in metrics:
            score += metrics['ml_accuracy'] * 0.1
        
        return score