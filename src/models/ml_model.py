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
            'volume_trend', 'price_sma20_ratio', 'atr_ratio',
            'bb_width'
        ]
        
        # Model parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.probability_threshold = self.config.get('probability_threshold', 0.7)
        
        logger.info("Initialized MLModel")
    
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for model training/prediction"""
        try:
            # Select features
            X = data[self.feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(0)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Create target - predict next period's direction
            returns = data['close'].pct_change().shift(-1)
            y = np.where(returns > 0, 1, 0)[:-1]
            X = X[:-1]  # Remove last row as we don't have next period's return
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None, None
    
    def train(self, data: pd.DataFrame) -> dict:
        """Train the machine learning model"""
        try:
            # Prepare features
            X, y = self.prepare_features(data)
            if X is None or y is None:
                return {}
            
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                class_weight='balanced'
            )
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            metrics = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                predictions = self.model.predict(X_test)
                metrics.append(classification_report(y_test, predictions, output_dict=True))
            
            # Calculate average metrics
            avg_metrics = self._average_metrics(metrics)
            logger.info(f"Model training completed. Accuracy: {avg_metrics['accuracy']:.2f}")
            
            # Train final model on all data
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
            
            # Prepare features
            X = data[self.feature_columns].copy()
            X = X.fillna(method='ffill').fillna(0)
            X = self.scaler.transform(X)
            
            # Get probabilities
            probabilities = self.model.predict_proba(X)
            
            # Convert to signals
            signals = pd.Series(0, index=data.index)
            
            # Long signals when probability of up move is high
            signals[probabilities[:, 1] > self.probability_threshold] = 1
            
            # Short signals when probability of down move is high
            signals[probabilities[:, 0] > self.probability_threshold] = -1
            
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