"""
Utility Functions and Helpers
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Handle model saving and loading
    """
    
    @staticmethod
    def save_model(model: Any, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str) -> Any:
        """Load model from disk"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


class DataVisualizationHelper:
    """
    Helper functions for data visualization
    """
    
    @staticmethod
    def prepare_prediction_plot_data(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    dates: List[str] = None) -> Dict:
        """
        Prepare data for prediction vs actual plot
        
        Args:
            y_true: True values
            y_pred: Predictions
            dates: Date labels
            
        Returns:
            Dictionary with plot data
        """
        if dates is None:
            dates = list(range(len(y_true)))
        
        return {
            'dates': dates,
            'actual': y_true.flatten().tolist(),
            'predicted': y_pred.flatten().tolist(),
        }
    
    @staticmethod
    def prepare_residual_plot_data(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict:
        """
        Prepare data for residual plot
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Dictionary with plot data
        """
        residuals = (y_true - y_pred.flatten()).flatten()
        fitted = y_pred.flatten()
        
        return {
            'fitted': fitted.tolist(),
            'residuals': residuals.tolist(),
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals)),
        }


class AlertSystem:
    """
    Financial stress alert system
    """
    
    ALERT_LEVELS = {
        'low': {'threshold': 0.3, 'color': 'green'},
        'medium': {'threshold': 0.6, 'color': 'yellow'},
        'high': {'threshold': 1.0, 'color': 'red'},
    }
    
    @staticmethod
    def get_stress_level(brs: float) -> str:
        """
        Determine stress level from BRS
        
        Args:
            brs: Behavioral Risk Score (0-1)
            
        Returns:
            Stress level string
        """
        if brs <= AlertSystem.ALERT_LEVELS['low']['threshold']:
            return 'low'
        elif brs <= AlertSystem.ALERT_LEVELS['medium']['threshold']:
            return 'medium'
        else:
            return 'high'
    
    @staticmethod
    def generate_alert_message(brs: float, daily_spending: float,
                              daily_income: float) -> str:
        """
        Generate alert message
        
        Args:
            brs: Behavioral Risk Score
            daily_spending: Daily spending
            daily_income: Daily income
            
        Returns:
            Alert message
        """
        level = AlertSystem.get_stress_level(brs)
        spending_ratio = daily_spending / (daily_income + 1e-8)
        
        if level == 'low':
            message = f"Financial status: HEALTHY | Risk: {brs:.2%}"
        elif level == 'medium':
            message = f"Financial status: MODERATE | Risk: {brs:.2%} | Spending: {spending_ratio:.2%}"
        else:
            message = f"Financial status: CRITICAL | Risk: {brs:.2%} | Spending: {spending_ratio:.2%}"
        
        return message
    
    @staticmethod
    def get_recommendations(brs: float, spending_ratio: float) -> List[str]:
        """
        Generate personalized recommendations
        
        Args:
            brs: Behavioral Risk Score
            spending_ratio: Spending to income ratio
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if brs > 0.7:
            recommendations.append("Set daily spending limits to reduce financial stress")
            recommendations.append("Track impulse purchases and identify triggers")
        
        if spending_ratio > 0.7:
            recommendations.append("Reduce variable expenses by 20-30%")
            recommendations.append("Prioritize essential expenses only")
        
        if brs > 0.5 and spending_ratio > 0.5:
            recommendations.append("Implement a detailed budget and stick to it")
            recommendations.append("Consider consulting a financial advisor")
        
        if not recommendations:
            recommendations.append("Continue your current financial management strategy")
        
        return recommendations


class ReportGenerator:
    """
    Generate comprehensive reports
    """
    
    @staticmethod
    def generate_daily_report(date: str, brs: float, daily_spending: float,
                            daily_income: float, predicted_stress: float,
                            recommendations: List[str]) -> Dict:
        """
        Generate daily financial report
        
        Args:
            date: Report date
            brs: Behavioral Risk Score
            daily_spending: Daily spending
            daily_income: Daily income
            predicted_stress: Predicted stress level
            recommendations: List of recommendations
            
        Returns:
            Report dictionary
        """
        spending_ratio = daily_spending / (daily_income + 1e-8)
        surplus = daily_income - daily_spending
        
        report = {
            'date': date,
            'behavioral_risk_score': float(brs),
            'daily_income': float(daily_income),
            'daily_spending': float(daily_spending),
            'spending_ratio': float(spending_ratio),
            'daily_surplus': float(surplus),
            'predicted_stress': float(predicted_stress),
            'stress_level': AlertSystem.get_stress_level(brs),
            'alert_message': AlertSystem.generate_alert_message(brs, daily_spending, daily_income),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
        }
        
        return report
    
    @staticmethod
    def generate_monthly_report(daily_reports: List[Dict]) -> Dict:
        """
        Generate monthly summary report
        
        Args:
            daily_reports: List of daily reports
            
        Returns:
            Monthly report dictionary
        """
        if not daily_reports:
            return {}
        
        brs_scores = [r['behavioral_risk_score'] for r in daily_reports]
        spending_ratios = [r['spending_ratio'] for r in daily_reports]
        daily_incomes = [r['daily_income'] for r in daily_reports]
        daily_spendings = [r['daily_spending'] for r in daily_reports]
        surplus_values = [r['daily_surplus'] for r in daily_reports]
        
        report = {
            'period': f"{daily_reports[0]['date']} to {daily_reports[-1]['date']}",
            'total_days': len(daily_reports),
            'total_income': float(np.sum(daily_incomes)),
            'total_spending': float(np.sum(daily_spendings)),
            'total_surplus': float(np.sum(surplus_values)),
            'average_daily_income': float(np.mean(daily_incomes)),
            'average_daily_spending': float(np.mean(daily_spendings)),
            'average_spending_ratio': float(np.mean(spending_ratios)),
            'average_brs': float(np.mean(brs_scores)),
            'max_brs': float(np.max(brs_scores)),
            'min_brs': float(np.min(brs_scores)),
            'high_stress_days': int(sum(1 for r in daily_reports 
                                         if AlertSystem.get_stress_level(r['behavioral_risk_score']) == 'high')),
            'medium_stress_days': int(sum(1 for r in daily_reports 
                                          if AlertSystem.get_stress_level(r['behavioral_risk_score']) == 'medium')),
            'low_stress_days': int(sum(1 for r in daily_reports 
                                       if AlertSystem.get_stress_level(r['behavioral_risk_score']) == 'low')),
            'savings_rate': float(np.sum(surplus_values) / (np.sum(daily_incomes) + 1e-8)),
            'timestamp': datetime.now().isoformat(),
        }
        
        return report


class TimeSeriesHelper:
    """
    Time series utility functions
    """
    
    @staticmethod
    def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            data: Input data
            seq_length: Sequence length
            
        Returns:
            Sequences and targets
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def add_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Add date-based features
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with date features
        """
        df_copy = df.copy()
        
        if date_col in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy[date_col])
            df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
            df_copy['day_of_month'] = df_copy['date'].dt.day
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['quarter'] = df_copy['date'].dt.quarter
            df_copy['week_of_year'] = df_copy['date'].dt.isocalendar().week
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        
        return df_copy


class ConfigManager:
    """
    Manage configuration files
    """
    
    @staticmethod
    def save_config(config: Dict, path: str):
        """Save configuration to JSON"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict:
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config = json.load(f)
        logger.info(f"Config loaded from {path}")
        return config

