"""
Data Pipeline Module
Handles data loading, preprocessing, and clustering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Complete data pipeline for loading, preprocessing, and clustering
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.gmm_model = None
        self.feature_names = None
        
    def generate_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic expense behavior dataset
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(self.random_state)
        
        data = {
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'daily_income': np.random.uniform(50, 200, n_samples),
            'grocery_expense': np.random.uniform(10, 50, n_samples),
            'entertainment_expense': np.random.uniform(0, 30, n_samples),
            'utilities_expense': np.random.uniform(20, 100, n_samples),
            'impulse_purchases': np.random.uniform(0, 50, n_samples),
            'mood_score': np.random.uniform(1, 10, n_samples),
            'stress_level': np.random.uniform(1, 10, n_samples),
            'sleep_hours': np.random.uniform(4, 10, n_samples),
            'social_activities': np.random.randint(0, 5, n_samples),
            'impulse_frequency': np.random.uniform(0, 1, n_samples),
            'overspending_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['total_spending'] = (df['grocery_expense'] + df['entertainment_expense'] + 
                                df['utilities_expense'] + df['impulse_purchases'])
        df['spending_ratio'] = df['total_spending'] / df['daily_income']
        df['mood_variance'] = df['mood_score'].rolling(window=7, min_periods=1).std()
        
        logger.info(f"Generated synthetic dataset with shape {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using imputation
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        logger.info(f"Missing values handled. Final shape: {df.shape}")
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          feature_cols: list = None, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize numeric features
        
        Args:
            df: Input DataFrame
            feature_cols: List of columns to normalize
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with normalized features
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'date' in feature_cols:
                feature_cols.remove('date')
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        logger.info(f"Features normalized. Columns: {len(feature_cols)}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders
            
        Returns:
            DataFrame with encoded categorical features
        """
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'date']
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info(f"Categorical features encoded: {len(categorical_cols)}")
        return df
    
    def perform_gmm_clustering(self, df: pd.DataFrame, 
                              feature_cols: list = None,
                              n_components: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform Gaussian Mixture Model clustering for behavioral personas
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to use for clustering
            n_components: Number of clusters
            
        Returns:
            DataFrame with cluster labels, cluster labels array
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'date' in feature_cols:
                feature_cols.remove('date')
        
        X = df[feature_cols].values
        
        self.gmm_model = GaussianMixture(n_components=n_components, 
                                        random_state=self.random_state,
                                        n_init=10)
        clusters = self.gmm_model.fit_predict(X)
        
        df['behavioral_persona'] = clusters
        
        logger.info(f"GMM Clustering completed. Components: {n_components}")
        logger.info(f"Cluster distribution:\n{pd.Series(clusters).value_counts()}")
        
        return df, clusters
    
    def preprocess_full_pipeline(self, df: pd.DataFrame = None,
                                n_synthetic_samples: int = 1000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute complete preprocessing pipeline
        
        Args:
            df: Input DataFrame (if None, generates synthetic data)
            n_synthetic_samples: Number of synthetic samples to generate
            
        Returns:
            Preprocessed DataFrame and metadata dictionary
        """
        def _try_convert_transaction_log(df_in: pd.DataFrame) -> pd.DataFrame | None:
            """Convert a transaction-log dataset into a daily aggregated dataset.

            Supports datasets with columns like:
              - Date/Category/Amount/Type
              - date/category/amount/type
            Where Type indicates income vs expense.
            """
            if not isinstance(df_in, pd.DataFrame) or df_in.empty:
                return None

            # Normalize column names for detection
            col_lut = {str(c).strip().lower(): c for c in df_in.columns}
            date_c = col_lut.get("date")
            cat_c = col_lut.get("category")
            amt_c = col_lut.get("amount")
            type_c = col_lut.get("type")

            if date_c is None or amt_c is None or type_c is None:
                return None

            tx = df_in.copy()
            tx[date_c] = pd.to_datetime(tx[date_c], errors="coerce")
            tx[amt_c] = pd.to_numeric(tx[amt_c], errors="coerce")
            tx[type_c] = tx[type_c].astype(str).str.strip().str.lower()
            if cat_c is not None:
                tx[cat_c] = tx[cat_c].astype(str).str.strip().str.lower()
            else:
                tx["__category"] = "other"
                cat_c = "__category"

            tx = tx.dropna(subset=[date_c, amt_c])
            if tx.empty:
                return None

            income_mask = tx[type_c].str.contains("income", na=False)
            expense_mask = tx[type_c].str.contains("expense", na=False)
            if not income_mask.any() and not expense_mask.any():
                # If Type isn't informative, can't safely convert.
                return None

            tx_income = tx[income_mask].copy()
            tx_exp = tx[expense_mask].copy()

            daily_income = tx_income.groupby(tx_income[date_c].dt.date)[amt_c].sum().rename("daily_income")

            def _bucket(cat: str) -> str:
                if "grocery" in cat or "food" in cat:
                    return "grocery_expense"
                if "util" in cat or "rent" in cat or "mortgage" in cat:
                    return "utilities_expense"
                if "entertain" in cat:
                    return "entertainment_expense"
                return "impulse_purchases"

            if not tx_exp.empty:
                tx_exp["__bucket"] = tx_exp[cat_c].map(_bucket)
                daily_exp = tx_exp.groupby([tx_exp[date_c].dt.date, "__bucket"])[amt_c].sum().unstack(fill_value=0.0)
            else:
                daily_exp = pd.DataFrame()

            out = pd.concat([daily_income, daily_exp], axis=1).fillna(0.0).reset_index()
            # The date column after reset_index can be named "index" or inherit the original
            # date column name (e.g., "Date"). Normalize it to "date".
            if "date" not in out.columns:
                date_like_cols = [c for c in out.columns if str(c).strip().lower() == "date"]
                if len(date_like_cols) > 0:
                    out = out.rename(columns={date_like_cols[0]: "date"})
                else:
                    out = out.rename(columns={out.columns[0]: "date"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce")

            # Fill daily_income if it's not recorded every day
            out = out.sort_values("date").reset_index(drop=True)
            inc = pd.to_numeric(out["daily_income"], errors="coerce").fillna(0.0)
            nonzero = inc.mask(inc == 0.0, np.nan)
            est = nonzero.ffill()
            median_income = float(np.nanmedian(nonzero.values)) if np.isfinite(nonzero.values).any() else 0.0
            est = est.fillna(median_income)
            out["daily_income"] = pd.to_numeric(est, errors="coerce").fillna(0.0)

            # Ensure expected expense columns exist
            for col in ["grocery_expense", "utilities_expense", "entertainment_expense", "impulse_purchases"]:
                if col not in out.columns:
                    out[col] = 0.0

            # Add neutral defaults for behavioral inputs if missing
            out["impulse_frequency"] = 0.2
            out["mood_score"] = 6.0
            return out

        # Load or generate data
        if df is None:
            df = self.generate_synthetic_dataset(n_synthetic_samples)
        else:
            df = df.copy()

        # If the incoming DF is a transaction log, convert it to daily schema
        if isinstance(df, pd.DataFrame) and 'daily_income' not in df.columns:
            converted = _try_convert_transaction_log(df)
            if converted is not None:
                df = converted

        # Ensure a date column exists
        if 'date' not in df.columns:
            df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
        else:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].isna().all():
                    df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
            except Exception:
                df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')

        # Validate minimum required columns for meaningful predictions
        if 'daily_income' not in df.columns:
            raise ValueError("Missing required column: daily_income")

        # Derive total_spending if possible
        if 'total_spending' not in df.columns:
            expense_cols = [c for c in ['grocery_expense', 'entertainment_expense', 'utilities_expense', 'impulse_purchases'] if c in df.columns]
            if len(expense_cols) > 0:
                df['total_spending'] = df[expense_cols].sum(axis=1)
            else:
                raise ValueError("Missing required column: total_spending (or expense-category columns)")

        # Derive spending_ratio if missing
        if 'spending_ratio' not in df.columns:
            df['spending_ratio'] = df['total_spending'] / (df['daily_income'] + 1e-8)

        # Ensure mood_score exists; if not, create a neutral default
        if 'mood_score' not in df.columns:
            df['mood_score'] = 6.0

        # Derive mood_variance if missing
        if 'mood_variance' not in df.columns:
            df['mood_variance'] = df['mood_score'].rolling(window=7, min_periods=1).std(ddof=0)

        # Preserve raw columns for downstream features (e.g., exact BRS formula)
        raw_cols = [
            'daily_income',
            'total_spending',
            'grocery_expense',
            'entertainment_expense',
            'utilities_expense',
            'impulse_purchases',
            'impulse_frequency',
            'mood_score',
        ]
        for col in raw_cols:
            if col in df.columns:
                df[f'{col}_raw'] = df[col]

        if 'mood_score' in df.columns:
            df['mood_variance_raw'] = df['mood_score'].rolling(window=7, min_periods=1).std(ddof=0)
        
        # Store original feature names
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' in numeric_cols:
            numeric_cols.remove('date')
        self.feature_names = numeric_cols
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Normalize features (exclude raw preserved columns)
        cols_to_scale = [c for c in numeric_cols if not c.endswith('_raw') and c != 'mood_variance_raw']
        df = self.normalize_features(df, feature_cols=cols_to_scale)
        
        # Perform clustering (exclude raw preserved columns)
        cluster_cols = [c for c in cols_to_scale if c in df.columns]
        df, clusters = self.perform_gmm_clustering(df, feature_cols=cluster_cols)
        
        metadata = {
            'n_samples': len(df),
            'feature_names': cols_to_scale,
            'n_clusters': len(np.unique(clusters)),
            'cluster_distribution': pd.Series(clusters).value_counts().to_dict()
        }
        
        logger.info(f"Pipeline completed. Metadata: {metadata}")
        return df, metadata
    
    def split_train_test(self, df: pd.DataFrame, 
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-series aware)
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test data
            
        Returns:
            Training and testing DataFrames
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Data split: Train {len(train_df)}, Test {len(test_df)}")
        return train_df, test_df
