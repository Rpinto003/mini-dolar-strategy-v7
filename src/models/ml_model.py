import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from loguru import logger
from src.analysis.feature_analysis import FeatureAnalyzer

class MLModel:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_analyzer = FeatureAnalyzer()
        
        # Features mais importantes para previsão de mercado
        self.feature_columns = [
            'price_sma10_ratio','price_sma20_ratio', 'price_sma50_ratio',
            'price_position','rsi',
            'roc_5', 'roc_10', 'roc_20',
            'returns','price_momentum'
            #'open', 'high', 'low', 'close', 'volume', 'g_high', 'swing_low', 'price_position', 'atr', 'returns',
            #'historical_volatility', 'basis', 'upper_band', 'lower_band', 'rsi', 'macd', 'signal_line', 'maatr_ratio', 
            #'price_sma20_ratio', 'price_sma50_ratio', 'signal', 'entry_price', 'entry_time', 'roc_5', 'roc_10', 'roc_20', 'volatility_5',
            #'volume_ma_ratio', 'sma_10', 'price_sma10_ratio', 'trend_short', 'trend_long', 'price_momentum', 'trend_strength'
            #'rsi', 'macd', 'macd_hist',         # Momentum
            #'volume_ratio', 'volume_trend',      # Volume
            #'atr_ratio', 'volatility_10',        # Volatilidade
            #'trend_short', 'trend_long',         # Tendência
            #'price_sma20_ratio', 'price_momentum' # Preço relativo
        ]
        
        # Parâmetros do modelo
        self.n_estimators = self.config.get('n_estimators', 500)
        self.max_depth = self.config.get('max_depth', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 100)
        self.probability_threshold = self.config.get('probability_threshold', 0.7)
        
        logger.info("Initialized MLModel")
    
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for model training/prediction"""
        try:
            # 1. Verificação inicial
            if len(data) < 1000:  # Precisamos de mais dados para treinar
                logger.error("Dataset too small for analysis")
                return None, None
            
            # 2. Seleção de features
            available_features = [col for col in self.feature_columns if col in data.columns]
            if len(available_features) < 6:  # Precisamos de pelo menos 6 features
                logger.error("Insufficient features available")
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
            
            # 6. Target dinâmico baseado na volatilidade
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            threshold = volatility * 1.5  # 1.5 desvios padrão
            
            # Classificação: 1 (up), 0 (sideways), -1 (down)
            y = pd.Series(0, index=returns.index)
            y[returns > threshold] = 1
            y[returns < -threshold] = -1
            y = y.shift(-1)  # Próximo período
            
            # 7. Remoção de NaNs e últimas linhas
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
            if X is None or y is None or len(X) < 1000:
                logger.error("Insufficient data for training")
                return {}
            
            # 2. Inicialização do modelo
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # 3. Cross-validação
            tscv = TimeSeriesSplit(n_splits=5, test_size=1000)
            metrics = []
            
            # 4. Treinamento e avaliação
            fold = 0
            for train_idx, test_idx in tscv.split(X):
                fold += 1
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Treina modelo
                self.model.fit(X_train, y_train)
                
                # Feature analysis for this fold
                logger.info(f"\nAnalyzing features for fold {fold}...")
                self.feature_analyzer.analyze_features(
                    self.model,
                    X_test,
                    y_test
                )
                
                # Avalia
                predictions = self.model.predict(X_test)
                fold_metrics = classification_report(y_test, predictions, output_dict=True)
                metrics.append(fold_metrics)
            
            # 5. Métricas finais
            avg_metrics = self._average_metrics(metrics)
            logger.info(f"Model training completed. Accuracy: {avg_metrics['accuracy']:.2f}")
            
            # 6. Treinamento final e análise
            self.model.fit(X, y)
            logger.info("\nFinal feature analysis...")
            self.feature_analyzer.analyze_features(self.model, X, y)
            
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
            
            # 4. Previsão com probabilidades
            probabilities = self.model.predict_proba(X)
            
            # 5. Geração de sinais com threshold mais alto
            signals = pd.Series(0, index=data.index)
            confidence_threshold = self.probability_threshold
            
            # Sinal de compra se prob > threshold
            buy_prob = probabilities[:, np.where(self.model.classes_ == 1)[0][0]]
            signals[buy_prob > confidence_threshold] = 1
            
            # Sinal de venda se prob > threshold
            sell_prob = probabilities[:, np.where(self.model.classes_ == -1)[0][0]]
            signals[sell_prob > confidence_threshold] = -1
            
            # 6. Análise das features para predição
            logger.info("\nAnalyzing features for prediction...")
            if len(X) > 0:
                self.feature_analyzer.analyze_features(self.model, X, signals)
            
            return signals.astype(int)
            
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