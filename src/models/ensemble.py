import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class EnsembleModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        }
        self.weights = {'rf': 0.5, 'gb': 0.5}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        metrics = {}
        
        for name, model in self.models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            
            metrics[f'{name}_accuracy'] = accuracy_score(y, y_pred)
            metrics[f'{name}_precision'] = precision_score(y, y_pred, average='weighted')
            metrics[f'{name}_recall'] = recall_score(y, y_pred, average='weighted')
        
        return metrics
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        predictions = pd.DataFrame()
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)
            predictions[name] = pd.Series(
                [1 if p[1] > 0.5 else -1 if p[0] > 0.5 else 0
                 for p in pred_proba]
            )
        
        final_pred = pd.Series(0, index=X.index)
        confidence = pd.Series(0.0, index=X.index)
        
        for name, weight in self.weights.items():
            final_pred += predictions[name] * weight
            confidence += model.predict_proba(X).max(axis=1) * weight
        
        return final_pred, confidence