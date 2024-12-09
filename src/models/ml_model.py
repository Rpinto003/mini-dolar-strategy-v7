import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from loguru import logger

class MLModel:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        
        # Define feature columns
        self.feature_columns = [
            'rsi', 'macd', 'macd_hist', 'volume_ratio',
            'volume_trend', 'atr_ratio'
        ]
        
        # Model parameters
        self.n_estimators = self.config.get('n_estimators', 200)
        self.max_depth = self.config.get('max_depth', 8)
        self.probability_threshold = self.config.get('probability_threshold', 0.75)
        
        logger.info("Initialized MLModel")
    
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for model training/prediction"""
        try:
            # 1. Verificação inicial
            if len(data) < 100:
                logger.error("Dataset too small for analysis")
                return None, None
            
            # 2. Seleção de features
            available_features = [col for col in self.feature_columns if col in data.columns]
            if not available_features:
                logger.error("No valid features found in data")
                return None, None
            
            # 3. Preparação dos dados
            X = data[available_features].copy()
            X = X.ffill().fillna(0)
            
            # 4. Remoção de outliers
            for col in X.columns:
                q1 = X[col].quantile(0.01)
                q3 = X[col].quantile(0.99)
                X[col] = X[col].clip(lower=q1, upper=q3)
            
            # 5. Normalização
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # 6. Criação do target
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            threshold = volatility.mean()
            
            y = pd.Series(0, index=returns.index)
            y[returns > threshold] = 1
            y[returns < -threshold] = -1
            y = y.shift(-1)  # Próximo período
            
            # 7. Remove última linha e NaNs
            valid_idx = ~(y.isna() | X.isna().any(axis=1))
            X = X[valid_idx]
            y = y[valid_idx]
            
            logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None, None
    
    def train(self, data: pd.DataFrame) -> dict:
        """Train the machine learning model"""
        try:
            # 1. Preparação dos dados
            X, y = self.prepare_features(data)
            if X is None or y is None or len(X) < 100:
                logger.error("Insufficient data for training")
                return {}
            
            # 2. Inicialização do modelo
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # 3. Cross-validação
            n_splits = min(5, len(X) // 200)
            if n_splits < 2:
                n_splits = 2
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            metrics = []
            
            # 4. Treinamento e avaliação
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_test)
                fold_metrics = classification_report(y_test, predictions, output_dict=True)
                metrics.append(fold_metrics)
            
            # 5. Cálculo das métricas finais
            avg_metrics = self._average_metrics(metrics)
            logger.info(f"Model training completed. Accuracy: {avg_metrics['accuracy']:.2f}")
            
            # 6. Treinamento final
            self.model.fit(X, y)
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {}
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data"""
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return pd.Series(0, index=data.index)
            
            # 1. Preparação dos dados
            X = data[self.feature_columns].copy()
            X = X.ffill().fillna(0)
            
            # 2. Remoção de outliers
            for col in X.columns:
                q1 = X[col].quantile(0.01)
                q3 = X[col].quantile(0.99)
                X[col] = X[col].clip(lower=q1, upper=q3)
            
            # 3. Normalização
            X = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # 4. Previsão
            probabilities = self.model.predict_proba(X)
            
            # 5. Geração de sinais
            signals = pd.Series(0, index=data.index)
            confidence_threshold = self.probability_threshold
            
            up_prob = probabilities[:, 1]
            down_prob = probabilities[:, 0]
            
            # Long signals
            signals[up_prob > confidence_threshold] = 1
            # Short signals
            signals[down_prob > confidence_threshold] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _average_metrics(self, metrics_list: list) -> dict:
        """Average metrics across cross-validation folds"""
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['weighted avg']['precision'] for m in metrics_list]),
            'recall': np.mean([m['weighted avg']['recall'] for m in metrics_list]),
            'f1-score': np.mean([m['weighted avg']['f1-score'] for m in metrics_list])
        }
        return avg_metrics