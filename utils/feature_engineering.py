"""
Feature Engineering Module
Computes behavioral risk scores and advanced features
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for behavioral analysis
    """
    
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        
    def compute_behavioral_risk_score(self, df: pd.DataFrame,
                                     impulse_freq_col: str = 'impulse_frequency',
                                     overspend_col: str = 'total_spending',
                                     mood_var_col: str = 'mood_variance',
                                     income_col: str = 'daily_income') -> pd.DataFrame:
        """
        Compute Behavioral Risk Score (BRS) using weighted formula
        
        BRS_t = 0.42 * ImpulseFreq_t + 0.35 * (Overspend_t / Income_t) + 0.23 * MoodVar_t
        
        Args:
            df: Input DataFrame
            impulse_freq_col: Column name for impulse frequency
            overspend_col: Column name for overspending ratio
            mood_var_col: Column name for mood variance
            income_col: Column name for income
            
        Returns:
            DataFrame with BRS column added
        """
        df_copy = df.copy()
        
        # Handle potential NaN values
        impulse_freq = df_copy.get(impulse_freq_col, pd.Series(0, index=df_copy.index)).fillna(0)
        overspend = df_copy.get(overspend_col, pd.Series(0, index=df_copy.index)).fillna(0)
        income = df_copy.get(income_col, pd.Series(1, index=df_copy.index)).fillna(1)
        mood_var = df_copy.get(mood_var_col, pd.Series(0, index=df_copy.index)).fillna(0)

        # Overspend term as specified: Overspend / Income
        overspend_over_income = overspend / (income + 1e-8)

        # Normalize each component to [0, 1] to keep BRS bounded and comparable
        def _minmax(series: pd.Series) -> pd.Series:
            return (series - series.min()) / (series.max() - series.min() + 1e-8)

        impulse_freq_norm = _minmax(impulse_freq.astype(float))
        overspend_norm = _minmax(overspend_over_income.astype(float))
        mood_var_norm = _minmax(mood_var.astype(float))

        brs = (0.42 * impulse_freq_norm + 0.35 * overspend_norm + 0.23 * mood_var_norm)
        
        # Clip to [0, 1]
        df_copy['behavioral_risk_score'] = np.clip(brs, 0, 1)
        
        logger.info(f"BRS computed. Mean: {df_copy['behavioral_risk_score'].mean():.4f}, "
                   f"Std: {df_copy['behavioral_risk_score'].std():.4f}")
        
        return df_copy
    
    def compute_rolling_features(self, df: pd.DataFrame,
                                window_size: int = None) -> pd.DataFrame:
        """
        Compute rolling statistical features for time-series data
        
        Args:
            df: Input DataFrame
            window_size: Window size for rolling calculations
            
        Returns:
            DataFrame with rolling features added
        """
        if window_size is None:
            window_size = self.window_size
        
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['date', 'behavioral_persona']:
                # Rolling statistics
                df_copy[f'{col}_rolling_mean'] = df_copy[col].rolling(
                    window=window_size, min_periods=1).mean()
                df_copy[f'{col}_rolling_std'] = df_copy[col].rolling(
                    window=window_size, min_periods=1).std(ddof=0)
                df_copy[f'{col}_rolling_max'] = df_copy[col].rolling(
                    window=window_size, min_periods=1).max()
                df_copy[f'{col}_rolling_min'] = df_copy[col].rolling(
                    window=window_size, min_periods=1).min()
        
        logger.info(f"Rolling features computed. Total features: {df_copy.shape[1]}")
        return df_copy
    
    def compute_lagged_features(self, df: pd.DataFrame,
                               lags: list = [1, 7, 14, 30],
                               cols_to_lag: list = None) -> pd.DataFrame:
        """
        Create lagged features for temporal dependencies
        
        Args:
            df: Input DataFrame
            lags: List of lag values
            cols_to_lag: Columns to create lags for
            
        Returns:
            DataFrame with lagged features
        """
        df_copy = df.copy()
        
        if cols_to_lag is None:
            cols_to_lag = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in ['date', 'behavioral_persona']]
        
        for col in cols_to_lag:
            for lag in lags:
                df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        # Drop NaN rows created by lagging
        df_copy = df_copy.dropna()
        
        logger.info(f"Lagged features created with lags {lags}. New shape: {df_copy.shape}")
        return df_copy
    
    def compute_interaction_features(self, df: pd.DataFrame,
                                    interactions: Dict[str, Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features between selected columns
        
        Args:
            df: Input DataFrame
            interactions: Dictionary of interaction pairs
            
        Returns:
            DataFrame with interaction features
        """
        df_copy = df.copy()
        
        if interactions is None:
            interactions = {
                'impulse_mood_interaction': ('impulse_frequency', 'mood_score'),
                'income_spending_interaction': ('daily_income', 'total_spending'),
                'stress_impulse_interaction': ('stress_level', 'impulse_purchases'),
            }
        
        for feature_name, (col1, col2) in interactions.items():
            if col1 in df_copy.columns and col2 in df_copy.columns:
                df_copy[feature_name] = df_copy[col1] * df_copy[col2]
        
        logger.info(f"Interaction features created: {len(interactions)}")
        return df_copy
    
    def compute_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on behavioral insights
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with domain-specific features
        """
        df_copy = df.copy()
        
        # Financial stress indicator
        if 'spending_ratio' in df_copy.columns:
            df_copy['high_spending_flag'] = (df_copy['spending_ratio'] > 0.5).astype(int)
        
        # Impulse buying risk
        if 'impulse_purchases' in df_copy.columns and 'total_spending' in df_copy.columns:
            df_copy['impulse_ratio'] = df_copy['impulse_purchases'] / (df_copy['total_spending'] + 1e-8)
        
        # Mood-based spending pattern
        if 'mood_score' in df_copy.columns and 'entertainment_expense' in df_copy.columns:
            df_copy['mood_based_spending'] = (df_copy['mood_score'] * 
                                             df_copy['entertainment_expense'])
        
        # Sleep deprivation impact on spending
        if 'sleep_hours' in df_copy.columns and 'impulse_purchases' in df_copy.columns:
            df_copy['sleep_deprivation_risk'] = (10 - df_copy['sleep_hours']) * df_copy['impulse_purchases']
        
        # Social activity spending correlation
        if 'social_activities' in df_copy.columns and 'entertainment_expense' in df_copy.columns:
            df_copy['social_spending'] = df_copy['social_activities'] * df_copy['entertainment_expense']
        
        logger.info(f"Domain-specific features created. Total features: {df_copy.shape[1]}")
        return df_copy
    
    def engineer_features_full_pipeline(self, df: pd.DataFrame,
                                       window_size: int = 7,
                                       lags: list = [1, 7, 14]) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            window_size: Window size for rolling calculations
            lags: List of lag values
            
        Returns:
            Engineered DataFrame
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Behavioral Risk Score
        df = self.compute_behavioral_risk_score(df)
        
        # Rolling features
        df = self.compute_rolling_features(df, window_size=window_size)
        
        # Lagged features
        df = self.compute_lagged_features(df, lags=lags)
        
        # Interaction features
        df = self.compute_interaction_features(df)
        
        # Domain-specific features
        df = self.compute_domain_specific_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def get_feature_importance_info(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get information about engineered features
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with feature statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            'total_features': len(numeric_cols),
            'behavioral_risk_score': 'behavioral_risk_score' in df.columns,
            'rolling_features': sum(1 for col in numeric_cols if 'rolling' in col),
            'lagged_features': sum(1 for col in numeric_cols if 'lag' in col),
            'interaction_features': sum(1 for col in numeric_cols if 'interaction' in col),
        }
