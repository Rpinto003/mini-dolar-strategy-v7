import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import shap
from sklearn.inspection import permutation_importance
from pathlib import Path

class FeatureAnalyzer:
    def __init__(self):
        self.output_dir = Path("output/feature_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized FeatureAnalyzer")
    
    def analyze_features(self, model, X: pd.DataFrame, y: pd.Series):
        """Perform comprehensive feature analysis"""
        try:
            # 1. Random Forest Feature Importance
            self._plot_feature_importance(model, X)
            
            # 2. SHAP Values
            self._plot_shap_values(model, X)
            
            # 3. Feature Correlations
            self._plot_correlation_matrix(X)
            
            # 4. Permutation Importance
            self._plot_permutation_importance(model, X, y)
            
            logger.info(f"Feature analysis plots saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")
    
    def _plot_feature_importance(self, model, X: pd.DataFrame):
        """Plot Random Forest feature importance"""
        try:
            # Get feature importance
            importance = model.feature_importances_
            
            # Create DataFrame for plotting
            feat_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp['feature'], feat_imp['importance'])
            plt.title('Feature Importance (Random Forest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def _plot_shap_values(self, model, X: pd.DataFrame):
        """Plot SHAP values for feature importance"""
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[1] if isinstance(shap_values, list) else shap_values,
                X,
                plot_type="bar",
                show=False
            )
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_importance.png')
            plt.close()
            
            # Detailed SHAP plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[1] if isinstance(shap_values, list) else shap_values,
                X,
                show=False
            )
            plt.title('SHAP Feature Impact')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_impact.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP values: {str(e)}")
    
    def _plot_correlation_matrix(self, X: pd.DataFrame):
        """Plot feature correlation matrix"""
        try:
            # Calculate correlations
            corr = X.corr()
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                corr,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f'
            )
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
    
    def _plot_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series):
        """Plot permutation feature importance"""
        try:
            # Calculate permutation importance
            result = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Create DataFrame for plotting
            perm_imp = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=True)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                perm_imp['importance_mean'],
                perm_imp['feature'],
                xerr=perm_imp['importance_std'],
                fmt='o'
            )
            plt.title('Permutation Feature Importance')
            plt.xlabel('Mean Importance (+/- std)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'permutation_importance.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting permutation importance: {str(e)}")
