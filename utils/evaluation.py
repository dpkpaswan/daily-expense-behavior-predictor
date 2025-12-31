"""
Evaluation and Statistical Testing Module
Computes metrics and performs statistical validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation and statistical testing
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize evaluator
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.results = {}
        
    def compute_regression_metrics(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Correlation': correlation,
        }
        
        logger.info(f"Regression Metrics: MAE={mae:.4f}, R2={r2:.4f}, RMSE={rmse:.4f}")
        
        return metrics
    
    def compute_classification_metrics(self, y_true: np.ndarray,
                                      y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics for stress prediction
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities or binary labels
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Binarize predictions if probabilities
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        y_true_binary = y_true.flatten()
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
        }
        
        logger.info(f"Classification Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return metrics
    
    def ks_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test
        Tests if two distributions are different
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            KS statistic and p-value
        """
        ks_stat, p_value = stats.ks_2samp(y_true, y_pred)
        
        logger.info(f"KS Test: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        
        return ks_stat, p_value
    
    def paired_ttest(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test
        Tests if mean difference is significantly different from zero
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            t-statistic and p-value
        """
        t_stat, p_value = stats.ttest_rel(y_true, y_pred)
        
        logger.info(f"Paired t-test: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
        
        return t_stat, p_value
    
    def anova_test(self, groups: list) -> Tuple[float, float]:
        """
        One-way ANOVA test
        Tests if means across multiple groups are significantly different
        
        Args:
            groups: List of arrays for each group
            
        Returns:
            F-statistic and p-value
        """
        f_stat, p_value = stats.f_oneway(*groups)
        
        logger.info(f"ANOVA Test: F-statistic={f_stat:.4f}, p-value={p_value:.4f}")
        
        return f_stat, p_value
    
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d effect size
        Measures standardized difference between two groups
        
        Args:
            group1: First group
            group2: Second group
            
        Returns:
            Cohen's d value
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        # Cohen's d
        d = (mean1 - mean2) / (pooled_std + 1e-8)
        
        logger.info(f"Cohen's d: {d:.4f}")
        
        return d
    
    def cross_validate_models(self, models: Dict, X: np.ndarray,
                             y: np.ndarray, k_folds: int = 5) -> Dict[str, list]:
        """
        Perform k-fold cross-validation on multiple models
        
        Args:
            models: Dictionary of models
            X: Features
            y: Targets
            k_folds: Number of folds
            
        Returns:
            Dictionary with CV scores for each model
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = {name: [] for name in models.keys()}
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for name, model in models.items():
                try:
                    # Train
                    model.fit(X_train, y_train)
                    # Score
                    score = model.score(X_test, y_test)
                    cv_results[name].append(score)
                except Exception as e:
                    logger.warning(f"Error in CV for {name}: {e}")
        
        # Print summary
        for name, scores in cv_results.items():
            logger.info(f"{name}: Mean CV Score = {np.mean(scores):.4f} "
                       f"(±{np.std(scores):.4f})")
        
        return cv_results
    
    def residual_analysis(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, any]:
        """
        Analyze residuals for model diagnostics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual analysis
        """
        residuals = y_true - y_pred.flatten()
        
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
        
        # Test for heteroscedasticity (using fitted values vs residuals)
        fitted_values = y_pred.flatten()
        abs_residuals = np.abs(residuals)
        
        analysis = {
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'residuals_min': np.min(residuals),
            'residuals_max': np.max(residuals),
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'normality_test_passed': shapiro_p > self.alpha,
        }
        
        logger.info(f"Residual Analysis: Mean={analysis['residuals_mean']:.4f}, "
                   f"Std={analysis['residuals_std']:.4f}")
        logger.info(f"Normality Test p-value: {shapiro_p:.4f}")
        
        return analysis
    
    def model_comparison_report(self, results_dict: Dict) -> pd.DataFrame:
        """
        Create comprehensive model comparison report
        
        Args:
            results_dict: Dictionary with model results
            
        Returns:
            Comparison DataFrame
        """
        report_data = []
        
        for model_name, metrics in results_dict.items():
            report_data.append({
                'Model': model_name,
                **metrics
            })
        
        report_df = pd.DataFrame(report_data)
        
        logger.info("\nModel Comparison Report:")
        logger.info(report_df.to_string())
        
        return report_df
    
    def statistical_significance_test(self, y_true1: np.ndarray,
                                     y_pred1: np.ndarray,
                                     y_true2: np.ndarray,
                                     y_pred2: np.ndarray) -> Dict[str, any]:
        """
        Compare two models statistically
        
        Args:
            y_true1: True values for model 1
            y_pred1: Predictions for model 1
            y_true2: True values for model 2
            y_pred2: Predictions for model 2
            
        Returns:
            Statistical comparison results
        """
        # Compute errors
        errors1 = np.abs(y_true1 - y_pred1.flatten())
        errors2 = np.abs(y_true2 - y_pred2.flatten())
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(errors1, errors2)
        
        # Cohen's d
        d = self.cohens_d(errors1, errors2)
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(errors1, errors2)
        
        comparison = {
            'mean_error_model1': np.mean(errors1),
            'mean_error_model2': np.mean(errors2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'significant_difference': p_value < self.alpha,
            'u_statistic': u_stat,
            'u_p_value': u_p_value,
        }
        
        logger.info(f"Model Comparison: p-value={p_value:.4f}, "
                   f"Cohen's d={d:.4f}")
        
        return comparison
    
    def generate_evaluation_report(self, y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model_name: str = "Model") -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: True values
            y_pred: Predictions
            model_name: Name of model
            
        Returns:
            Complete evaluation report
        """
        report = {
            'model_name': model_name,
            'regression_metrics': self.compute_regression_metrics(y_true, y_pred),
            'ks_test': {'statistic': self.ks_test(y_true, y_pred)[0]},
            'paired_ttest': {'t_statistic': self.paired_ttest(y_true, y_pred)[0]},
            'residual_analysis': self.residual_analysis(y_true, y_pred),
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Report for {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Regression Metrics: {report['regression_metrics']}")
        logger.info(f"Residual Analysis: {report['residual_analysis']}")
        logger.info(f"{'='*50}\n")
        
        return report
